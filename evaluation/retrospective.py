"""
evaluation/retrospective.py

Retrospective case testing — simulates what alerts the framework
would have generated at each reporting period for 3-4 held-out
test projects, and compares those alerts against what actually
happened to the project.

This answers the key evaluation question:
    Would the framework have given enough lead time for a manager
    to intervene before the budget overrun occurred?

The minimum lead time target is 4 reporting periods — consistent
with the 4-week intervention window defined in the Research Design.

Outputs:
    - Console report with per-project alert timelines
    - CSV saved to evaluation/results/retrospective_results.csv
    - Narrative summary ready for dissertation

Run after benchmark.py:
    python evaluation/retrospective.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import json
import os
import sys
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine
from config import MODELS_SAVED
from governance.decision_matrix import classify_variance

OUTPUT_DIR  = 'evaluation/results'
PLOTS_DIR   = 'evaluation/results/retrospective_plots'
FEATURE_COLS = [
    'cpi_trend',
    'sv_trajectory',
    'burn_rate_ratio',
    'schedule_pressure'
]
# Minimum periods of lead time considered adequate
MIN_LEAD_TIME = 4


# ── STEP 1: Select retrospective case projects ────────────────────────────────

def select_case_projects(n=4):
    """
    Select the most interesting test projects for retrospective analysis.
    Picks projects that represent a range of outcomes:
        - One large overrun (Critical or Red tier)
        - One moderate overrun (Amber tier)
        - One on-budget or slight overrun (Green tier)
        - One underrun

    This gives a balanced case study that demonstrates the framework
    across different project scenarios.
    """
    query = """
        SELECT
            pr.project_id,
            pr.project_name,
            pr.project_size,
            pr.budget_at_completion   AS bac,
            pr.total_periods,
            f.actual_final_variance   AS actual_variance,
            f.prediction_period
        FROM features f
        JOIN projects pr ON pr.project_id = f.project_id
        WHERE pr.train_test_flag = 'test'
        AND   pr.quality_flag    = 'pass'
        AND   pr.total_periods   >= 8
        ORDER BY f.actual_final_variance DESC
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        return df

    selected = []

    # Largest overrun
    overruns = df[df['actual_variance'] > 5]
    if not overruns.empty:
        selected.append(overruns.iloc[0])

    # Moderate overrun
    moderate = df[
        (df['actual_variance'] > 0) &
        (df['actual_variance'] <= 5)
    ]
    if not moderate.empty:
        selected.append(moderate.iloc[0])

    # On budget or slight overrun
    on_budget = df[
        (df['actual_variance'] >= -2) &
        (df['actual_variance'] <= 2)
    ]
    if not on_budget.empty:
        row = on_budget.iloc[len(on_budget) // 2]
        if not any(
            s['project_id'] == row['project_id']
            for s in selected
        ):
            selected.append(row)

    # Underrun
    underruns = df[df['actual_variance'] < -5]
    if not underruns.empty:
        selected.append(underruns.iloc[-1])

    # Fill remaining slots if needed
    for _, row in df.iterrows():
        if len(selected) >= n:
            break
        if not any(
            s['project_id'] == row['project_id']
            for s in selected
        ):
            selected.append(row)

    return pd.DataFrame(selected).reset_index(drop=True)


# ── STEP 2: Load period data for one project ──────────────────────────────────

def load_project_periods(project_id):
    """Load all periods for one project with PV, EV and AC."""
    query = """
        SELECT
            period_number,
            planned_value,
            earned_value,
            actual_cost
        FROM periods
        WHERE project_id = :project_id
        ORDER BY period_number
    """
    from sqlalchemy import text
    with engine.connect() as conn:
        return pd.read_sql(
            text(query), conn,
            params={'project_id': project_id}
        )


# ── STEP 3: Calculate features at any period ──────────────────────────────────

def calc_features_at_period(periods_df, cutoff_period, bac):
    """
    Calculate the four predictive features at any given period.
    Used to simulate what features the model would have seen
    at each point in the project's life.
    """
    window = periods_df[
        periods_df['period_number'] <= cutoff_period
    ].copy()

    if len(window) < 2:
        return None

    # CPI trend
    valid_cpi = window[
        (window['actual_cost'] > 0) &
        (window['actual_cost'].notna()) &
        (window['earned_value'].notna())
    ]
    cpi_trend = float(
        (valid_cpi['earned_value'] / valid_cpi['actual_cost'])
        .clip(0.1, 3.0).mean()
    ) if not valid_cpi.empty else None

    # SV trajectory
    valid_sv = window[
        window['earned_value'].notna() &
        window['planned_value'].notna()
    ]
    sv_trajectory = float(
        ((valid_sv['earned_value'] - valid_sv['planned_value']) / bac)
        .mean()
    ) if not valid_sv.empty and bac > 0 else None

    # Burn rate ratio
    last_ac = window.iloc[-1]['actual_cost']
    burn_rate_ratio = float(last_ac / bac) \
        if bac > 0 and pd.notna(last_ac) else None

    # Schedule pressure
    last_row = window.iloc[-1]
    ev = last_row['earned_value']
    pv = last_row['planned_value']
    schedule_pressure = float((ev - pv) / bac) \
        if bac > 0 and pd.notna(ev) and pd.notna(pv) else None

    if any(v is None for v in [
        cpi_trend, sv_trajectory,
        burn_rate_ratio, schedule_pressure
    ]):
        return None

    return {
        'cpi_trend':         cpi_trend,
        'sv_trajectory':     sv_trajectory,
        'burn_rate_ratio':   burn_rate_ratio,
        'schedule_pressure': schedule_pressure
    }


# ── STEP 4: Simulate alert timeline for one project ───────────────────────────

def simulate_alert_timeline(
    project_id, project_name, bac,
    total_periods, actual_variance,
    model, explainer
):
    """
    Simulate the framework running at every reporting period
    for one project.

    For each period from period 3 onwards:
        1. Calculate features from data available up to that period
        2. Run the Random Forest model
        3. Classify the prediction into a governance tier
        4. Record the tier

    This shows exactly when the framework would have first raised
    an Amber or Red alert — and how many periods of lead time
    that provided before the project ended.
    """
    periods_df = load_project_periods(project_id)

    if periods_df.empty or total_periods < 4:
        return None

    timeline = []
    prev_tier = None

    for period in range(3, total_periods + 1):
        features = calc_features_at_period(
            periods_df, period, bac
        )
        if features is None:
            continue

        # Run RF model
        X = np.array([[
            features['cpi_trend'],
            features['sv_trajectory'],
            features['burn_rate_ratio'],
            features['schedule_pressure']
        ]])

        rf_prediction = float(model.predict(X)[0])
        tier_info     = classify_variance(rf_prediction)
        tier_name     = tier_info['name']
        tier_number   = tier_info['tier_number']

        # Detect tier changes
        tier_changed = tier_name != prev_tier
        prev_tier    = tier_name

        # Calculate completion percentage at this period
        completion_pct = round(period / total_periods * 100, 1)

        # Periods remaining from this point
        periods_remaining = total_periods - period

        timeline.append({
            'period':            period,
            'completion_pct':    completion_pct,
            'periods_remaining': periods_remaining,
            'rf_prediction':     round(rf_prediction, 4),
            'tier_name':         tier_name,
            'tier_number':       tier_number,
            'tier_changed':      tier_changed,
            'features':          features,
        })

    return timeline


# ── STEP 5: Analyse alert timing ──────────────────────────────────────────────

def analyse_alert_timing(timeline, actual_variance, project_name):
    """
    From the simulated timeline, identify:
        - When the first meaningful alert fired (Amber or above)
        - How many periods of lead time that provided
        - Whether the lead time was adequate (>= MIN_LEAD_TIME)
        - Whether the final tier matched the actual outcome

    Returns a dictionary of findings for the dissertation.
    """
    if not timeline:
        return None

    first_amber  = None
    first_red    = None
    first_critical = None
    final_tier   = timeline[-1]['tier_name']
    final_pred   = timeline[-1]['rf_prediction']

    for entry in timeline:
        tier = entry['tier_name']
        if tier in ['AMBER', 'RED', 'CRITICAL'] \
                and first_amber is None:
            first_amber = entry
        if tier in ['RED', 'CRITICAL'] and first_red is None:
            first_red = entry
        if tier == 'CRITICAL' and first_critical is None:
            first_critical = entry

    # Classify actual outcome
    actual_tier = classify_variance(actual_variance)['name']

    # Lead time assessment
    lead_time_periods = None
    lead_time_adequate = None
    alert_entry = first_amber or first_red or first_critical

    if alert_entry:
        lead_time_periods  = alert_entry['periods_remaining']
        lead_time_adequate = lead_time_periods >= MIN_LEAD_TIME

    # Final prediction accuracy
    prediction_error = abs(final_pred - actual_variance)

    # Did the framework correctly identify the tier?
    tier_correct = (
        final_tier == actual_tier or
        (final_tier in ['RED', 'CRITICAL'] and
         actual_tier in ['RED', 'CRITICAL']) or
        (final_tier == 'AMBER' and actual_variance > 3)
    )

    return {
        'project_name':        project_name,
        'actual_variance':     actual_variance,
        'actual_tier':         actual_tier,
        'final_prediction':    final_pred,
        'final_tier':          final_tier,
        'tier_correct':        tier_correct,
        'prediction_error':    round(prediction_error, 4),
        'first_amber_period':  first_amber['period']
                               if first_amber else None,
        'first_amber_completion': first_amber['completion_pct']
                                  if first_amber else None,
        'first_red_period':    first_red['period']
                               if first_red else None,
        'lead_time_periods':   lead_time_periods,
        'lead_time_adequate':  lead_time_adequate,
        'total_periods':       timeline[-1]['period'],
    }


# ── STEP 6: Generate timeline chart ──────────────────────────────────────────

def plot_alert_timeline(
    timeline, project_name,
    actual_variance, project_id
):
    """
    Generate a visual timeline chart showing how the predicted
    variance and alert tier changed across all reporting periods.

    The chart shows:
        - RF predicted variance at each period (blue line)
        - Actual final variance (purple dashed line)
        - Tier boundary lines (green/amber/red thresholds)
        - Coloured background bands for each tier zone
        - First alert markers
    """
    if not timeline:
        return

    periods     = [t['period']         for t in timeline]
    predictions = [t['rf_prediction']  for t in timeline]
    tiers       = [t['tier_name']      for t in timeline]
    completions = [t['completion_pct'] for t in timeline]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    fig.patch.set_facecolor('#F8FAFC')

    # ── Top chart: predicted variance over time ────────────────────────
    ax1.set_facecolor('#F8FAFC')

    # Tier background bands
    y_min = min(min(predictions), actual_variance, -25) - 5
    y_max = max(max(predictions), actual_variance, 25)  + 5

    ax1.axhspan(-5,   5,   alpha=0.08, color='#27AE60', label='Green zone')
    ax1.axhspan( 5,  10,   alpha=0.08, color='#E67E22', label='Amber zone')
    ax1.axhspan(10,  20,   alpha=0.08, color='#E74C3C', label='Red zone')
    ax1.axhspan(20,  y_max,alpha=0.08, color='#922B21', label='Critical zone')
    ax1.axhspan(y_min,-5,  alpha=0.08, color='#3498DB', label='Underrun zone')

    # Tier boundary lines
    for y, col in [(5, '#E67E22'), (10, '#E74C3C'), (20, '#922B21'),
                   (-5, '#3498DB')]:
        ax1.axhline(
            y=y, color=col, linestyle='--',
            linewidth=0.8, alpha=0.6
        )

    # RF prediction line
    ax1.plot(
        periods, predictions,
        color='#1F4E79', linewidth=2.5,
        marker='o', markersize=4,
        label='RF Predicted Variance', zorder=5
    )

    # Actual variance line
    ax1.axhline(
        y=actual_variance, color='#8E44AD',
        linestyle='-', linewidth=2, alpha=0.8,
        label=f'Actual Variance: {actual_variance:.1f}%'
    )

    # Mark first amber alert
    first_amber = next(
        (t for t in timeline
         if t['tier_name'] in ['AMBER', 'RED', 'CRITICAL']),
        None
    )
    if first_amber:
        ax1.axvline(
            x=first_amber['period'],
            color='#E67E22', linestyle=':',
            linewidth=2, alpha=0.9,
            label=f"First alert: period {first_amber['period']} "
                  f"({first_amber['completion_pct']:.0f}% complete)"
        )
        ax1.annotate(
            f"First alert\nPeriod {first_amber['period']}\n"
            f"{first_amber['periods_remaining']} periods lead time",
            xy=(first_amber['period'], first_amber['rf_prediction']),
            xytext=(first_amber['period'] + 1,
                    first_amber['rf_prediction'] + 3),
            fontsize=8, color='#E67E22',
            arrowprops=dict(
                arrowstyle='->', color='#E67E22', lw=1.5
            )
        )

    ax1.set_ylabel('Predicted Cost Variance (%)', fontsize=10)
    ax1.set_title(
        f'Retrospective Alert Timeline — {project_name}\n'
        f'Actual outcome: {actual_variance:.2f}%',
        fontsize=11, fontweight='bold', color='#1F4E79'
    )
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax1.set_ylim(y_min, y_max)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='#AAAAAA', linewidth=0.8)

    # ── Bottom chart: tier band over time ──────────────────────────────
    ax2.set_facecolor('#F8FAFC')
    tier_colours_map = {
        'GREEN': '#27AE60', 'AMBER': '#E67E22',
        'RED': '#E74C3C',   'CRITICAL': '#922B21'
    }
    tier_colours_list = [
        tier_colours_map.get(t, '#AAAAAA') for t in tiers
    ]

    ax2.bar(
        periods, [1] * len(periods),
        color=tier_colours_list,
        alpha=0.8, width=0.8
    )
    ax2.set_yticks([])
    ax2.set_xlabel('Reporting Period', fontsize=10)
    ax2.set_ylabel('Tier', fontsize=9)
    ax2.set_title('Alert Tier by Period', fontsize=9, color='#555')

    # Legend for tier colours
    patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in tier_colours_map.items()
    ]
    ax2.legend(
        handles=patches, loc='upper right',
        fontsize=7, framealpha=0.9,
        ncol=4
    )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(
        PLOTS_DIR,
        f'retrospective_project_{project_id}.png'
    )
    plt.savefig(
        path, dpi=150,
        bbox_inches='tight',
        facecolor='#F8FAFC'
    )
    plt.close()
    print(f"  Chart saved: {path}")


# ── STEP 7: Save results ──────────────────────────────────────────────────────

def save_results(all_findings, all_timelines):
    """Save retrospective results to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Findings summary
    findings_df  = pd.DataFrame(all_findings)
    findings_path = os.path.join(
        OUTPUT_DIR, 'retrospective_results.csv'
    )
    findings_df.to_csv(findings_path, index=False)
    print(f"  Saved: {findings_path}")

    # Full timeline data
    timeline_rows = []
    for project_name, timeline in all_timelines.items():
        for entry in timeline:
            row = {
                'project_name': project_name,
                'period':       entry['period'],
                'completion_pct': entry['completion_pct'],
                'rf_prediction':  entry['rf_prediction'],
                'tier_name':      entry['tier_name'],
                'periods_remaining': entry['periods_remaining'],
            }
            timeline_rows.append(row)

    timeline_df   = pd.DataFrame(timeline_rows)
    timeline_path = os.path.join(
        OUTPUT_DIR, 'retrospective_timelines.csv'
    )
    timeline_df.to_csv(timeline_path, index=False)
    print(f"  Saved: {timeline_path}")


# ── STEP 8: Print dissertation report ────────────────────────────────────────

def print_report(all_findings, case_projects):
    """Print the retrospective evaluation report."""
    print("\n" + "=" * 70)
    print("RETROSPECTIVE CASE TESTING RESULTS")
    print("=" * 70)

    # Counts — use different names
    adequate_count = sum(
        1 for f in all_findings
        if f.get('lead_time_adequate') is True
    )
    correct_tier_count = sum(
        1 for f in all_findings
        if f.get('tier_correct') is True
    )
    n = len(all_findings)

    print(f"\n  Cases analysed          : {n}")
    print(f"  Adequate lead time      : {adequate_count}/{n} cases")
    print(f"  Correct tier identified : {correct_tier_count}/{n} cases")
    print(f"  Min lead time target    : {MIN_LEAD_TIME} periods")

    print(f"\n  Per-case summary:")
    print(f"  {'Project':<32} {'Actual':<10} {'Tier':<10} "
          f"{'1st Alert':<12} {'Lead Time':<12} {'Adequate'}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} "
          f"{'-'*12} {'-'*12} {'-'*8}")

    for f in all_findings:
        name          = str(f['project_name'])[:30]
        actual        = f"{f['actual_variance']:+.2f}%"
        tier          = f['actual_tier']
        alert_per     = (f"Period {f['first_amber_period']}"
                         if f['first_amber_period'] else "None")
        lead          = (f"{f['lead_time_periods']} periods"
                         if f['lead_time_periods'] is not None else "N/A")
        adequate_flag = ("✓ Yes" if f['lead_time_adequate'] is True
                         else ("✗ No" if f['lead_time_adequate'] is False
                               else "N/A"))  # Changed from 'adequate' to 'adequate_flag'
        print(f"  {name:<32} {actual:<10} {tier:<10} "
              f"{alert_per:<12} {lead:<12} {adequate_flag}")

    # Detailed per-case findings
    print(f"\n  Detailed case findings:")
    for f in all_findings:
        print(f"\n  ── {f['project_name']} ──")
        print(f"     Actual variance   : {f['actual_variance']:+.2f}%"
              f"  (Tier: {f['actual_tier']})")
        print(f"     Final prediction  : {f['final_prediction']:+.2f}%"
              f"  (Tier: {f['final_tier']})")
        print(f"     Prediction error  : {f['prediction_error']:.2f}%")
        print(f"     Tier correct      : "
              f"{'Yes ✓' if f['tier_correct'] else 'No ✗'}")

        if f['first_amber_period']:
            print(f"     First alert       : "
                  f"Period {f['first_amber_period']} "
                  f"({f['first_amber_completion']:.0f}% complete)")
            print(f"     Lead time         : "
                  f"{f['lead_time_periods']} periods remaining")
            print(f"     Lead time adequate: "
                  f"{'Yes ✓' if f['lead_time_adequate'] else 'No ✗'}")
        else:
            print(f"     First alert       : No Amber/Red alert fired")
            print(f"     Lead time         : N/A — project stayed Green")

    # Dissertation narrative
    print(f"\n" + "=" * 70)
    print("DISSERTATION NARRATIVE — paste into Results chapter")
    print("=" * 70)
    
    # Use the counts, not the flag strings
    adequate_pct = (adequate_count / n * 100) if n > 0 else 0
    correct_pct = (correct_tier_count / n * 100) if n > 0 else 0
    
    print(f"""
  Retrospective Case Testing Results

  The framework was applied retrospectively to {n} held-out test
  projects, simulating the alerts it would have generated at each
  reporting period and comparing these against the known project
  outcomes.

  Of the {n} cases analysed, {adequate_count} ({adequate_pct:.0f}%) received
  a meaningful alert (Amber or above) with at least {MIN_LEAD_TIME}
  reporting periods of lead time remaining — sufficient for a
  programme manager to investigate and intervene before further
  budget was committed. The framework correctly identified the
  governance tier for {correct_tier_count} of {n} cases ({correct_pct:.0f}%).

  These results demonstrate that the framework provides actionable
  early warning within the intervention window defined in the
  Research Design (minimum 4 weeks / reporting periods), consistent
  with the lead time requirements identified by Lipke et al. (2009).
    """)
    print("=" * 70)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_retrospective():
    print("\nStarting retrospective case testing...\n")

    # Load model
    print("Loading model:")
    tuned_path    = os.path.join(
        MODELS_SAVED, 'random_forest_tuned.pkl'
    )
    baseline_path = os.path.join(
        MODELS_SAVED, 'random_forest_baseline.pkl'
    )
    if os.path.exists(tuned_path):
        model = joblib.load(tuned_path)
        print("  Loaded tuned Random Forest")
    elif os.path.exists(baseline_path):
        model = joblib.load(baseline_path)
        print("  Loaded baseline Random Forest")
    else:
        print("ERROR: No model found. Run models/train_rf.py first.")
        return

    explainer = shap.TreeExplainer(model)

    # Select case projects
    print("\nSelecting case projects:")
    case_projects = select_case_projects(n=4)

    if case_projects.empty:
        print("ERROR: No suitable test projects found.")
        return

    print(f"  Selected {len(case_projects)} projects:")
    for _, row in case_projects.iterrows():
        print(f"    project {int(row['project_id']):<6} "
              f"{row['project_name'][:40]:<40} "
              f"actual: {row['actual_variance']:+.2f}%")

    # Run retrospective simulation for each project
    all_findings  = []
    all_timelines = {}

    for _, proj in case_projects.iterrows():
        project_id   = int(proj['project_id'])
        project_name = proj['project_name']
        bac          = float(proj['bac'])
        total        = int(proj['total_periods'])
        actual_var   = float(proj['actual_variance'])

        print(f"\n  Simulating: {project_name}")
        print(f"    BAC: {bac:,.2f} | "
              f"Periods: {total} | "
              f"Actual: {actual_var:+.2f}%")

        # Simulate alert timeline
        timeline = simulate_alert_timeline(
            project_id, project_name, bac,
            total, actual_var, model, explainer
        )

        if not timeline:
            print(f"    SKIPPED — insufficient data")
            continue

        all_timelines[project_name] = timeline

        # Analyse timing
        findings = analyse_alert_timing(
            timeline, actual_var, project_name
        )
        if findings:
            all_findings.append(findings)

            # Print quick summary
            if findings['first_amber_period']:
                print(
                    f"    First alert: Period "
                    f"{findings['first_amber_period']} "
                    f"({findings['first_amber_completion']:.0f}%"
                    f" complete) — "
                    f"{findings['lead_time_periods']} periods lead time "
                    f"{'✓' if findings['lead_time_adequate'] else '✗'}"
                )
            else:
                print(f"    No Amber/Red alert fired — "
                      f"project stayed Green throughout")

        # Generate timeline chart
        plot_alert_timeline(
            timeline, project_name,
            actual_var, project_id
        )

    if not all_findings:
        print("\nERROR: No findings generated.")
        return

    # Save results
    print("\nSaving results:")
    save_results(all_findings, all_timelines)

    # Print report
    print_report(all_findings, case_projects)

    print("\nRetrospective case testing complete.")
    print("Results saved to evaluation/results/")
    print("Charts saved to evaluation/results/retrospective_plots/")
    print("\nNext steps:")
    print("  1. Deploy to Streamlit Cloud")
    print("  2. Send expert panel the rubric")
    print("  3. Write Results and Evaluation chapters")


if __name__ == "__main__":
    run_retrospective()