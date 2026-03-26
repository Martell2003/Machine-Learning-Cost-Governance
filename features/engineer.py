"""
features/engineer.py

Calculates the four predictive features for each project at the 20%
completion mark and saves them to the features table in PostgreSQL.

The four features are:
    1. CPI Trend          — average EV/AC over the first 20% of periods
    2. SV Trajectory      — average EV-PV over the first 20% of periods
    3. Burn-Rate Ratio    — AC at 20% mark divided by BAC
    4. Schedule Pressure  — SV at 20% mark divided by BAC

Target variable:
    Final Cost Variance % — how much the project went over/under budget

"""

import pandas as pd
import numpy as np
from sqlalchemy import text
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine


# ── STEP 1: Load data ─────────────────────────────────────────────────────────

def load_data():
    """
    Load all projects and their periods from the database.
    Includes both train and test projects — features are calculated
    for all of them. The split flag is used later during model training.
    Excludes projects that failed the quality gate.
    """
    query = """
        SELECT
            pr.project_id,
            pr.project_name,
            pr.budget_at_completion   AS bac,
            pr.total_periods,
            pr.train_test_flag,
            pr.project_size,
            p.period_id,
            p.period_number,
            p.planned_value,
            p.earned_value,
            p.actual_cost
        FROM projects pr
        JOIN periods p ON pr.project_id = p.project_id
        WHERE pr.quality_flag = 'pass'
        ORDER BY pr.project_id, p.period_number
    """
    df = pd.read_sql(query, engine)
    n_projects = df['project_id'].nunique()
    print(f"Loaded {len(df)} period rows across {n_projects} projects")
    return df


# ── STEP 2: Find the 20% completion period ────────────────────────────────────

def get_prediction_period(group):
    """
    Find the period number that represents the 20% completion mark but ensure
    a minimum of 3 periods to ensure features are calculated from enough data points.

    Why 20%:
    Research by Lipke et al. (2009) shows that the Cost Performance Index
    stabilises after the first 20% of a project's duration — meaning the
    CPI at 20% is a reliable predictor of the final CPI. Predicting earlier
    than this gives unstable signals; waiting longer reduces the lead time
    available for management intervention.
    """
    total_periods = group['period_number'].max()
    cutoff = max(3, round(total_periods * 0.20))
    return cutoff


# ── STEP 3: Calculate individual features ─────────────────────────────────────

def calc_cpi_trend(window):
    """
    CPI Trend — average Cost Performance Index across the prediction window.

    Formula per period: EV / AC
    Final value: mean of all per-period CPIs in the window.

    A CPI below 1.0 means the project is earning less value than it is
    spending. The trend (average) is more stable than a single-period
    snapshot and less sensitive to one-off cost spikes.
    """
    # Avoid division by zero — skip periods where AC is zero or null
    valid = window[(window['actual_cost'] > 0) & (window['actual_cost'].notna()) & (window['earned_value'].notna())]
    if valid.empty:
        return None
    cpi_per_period = (valid['earned_value'] / valid['actual_cost']).clip(0.1, 3.0) # Cap extreme CPI values to prevent skewing the average
    return round(float(cpi_per_period.mean()), 6)


def calc_sv_trajectory(window, bac):
    """
    Schedule Variance Trajectory — average SV normalised by BAC across the prediction window.

    Formula per period: (EV - PV) / BAC per period, then averaged
    Final value: mean of all per-period SVs in the window.

    Normalising by BAC makes the SV comparable across projects of 
    different sizes. Without this, a £5m SV on a £100m project 
    looks far worse to the model than a £50k SV on a £200k project,
    even though both represent a 5% schedule slippage.

    """
    if not bac or bac == 0:
        return None
    valid = window[window['earned_value'].notna() & window['planned_value'].notna()]
    if valid.empty:
        return None
    sv_per_period = (valid['earned_value'] - valid['planned_value']) / bac
    return round(float(sv_per_period.mean()), 6)


def calc_burn_rate_ratio(window, bac):
    """
    Burn-Rate Ratio — cumulative AC at the 20% mark divided by BAC.

    Formula: AC_at_cutoff / BAC
    Uses the AC value at the last period in the window (cumulative spend).

    A burn-rate of 0.25 at the 20% mark means 25% of the total budget
    has already been spent by the time 20% of the project is complete —
    a warning sign that the project will exhaust its budget too early.
    """
    if not bac or bac == 0:
        return None
    # Take the AC at the last period of the window (cumulative)
    last_ac = window.iloc[-1]['actual_cost']
    if last_ac is None or pd.isna(last_ac):
        return None
    return round(float(last_ac / bac), 6)


def calc_schedule_pressure(window, bac):
    """
    Schedule Pressure Index — SV at the 20% mark normalised by BAC.

    Formula: (EV - PV) at the cutoff period / BAC
    Uses the SV at the last period in the window.

    Dividing by BAC makes schedule pressure comparable across projects
    of different budget sizes. A £100,000 schedule slippage means very
    different things on a £200,000 project versus a £20,000,000 project.
    """
    if not bac or bac == 0:
        return None
    last_row = window.iloc[-1]
    ev = last_row['earned_value']
    pv = last_row['planned_value']
    if pd.isna(ev) or pd.isna(pv):
        return None
    sv = ev - pv
    return round(float(sv / bac), 6)


def calc_final_variance(group, bac):
    """
    Final Cost Variance % — the target variable the model will learn to predict.

    Formula: (Final AC - BAC) / BAC * 100

    A positive value means the project went over budget.
    A negative value means it came in under budget.
    This is calculated from the last period of the project — the actual
    final outcome — which is only available because DSLIB contains
    completed projects with known outcomes.
    """
    if not bac or bac == 0:
        return None
    final_ac = group.iloc[-1]['actual_cost']
    if pd.isna(final_ac):
        return None
    variance_pct = ((final_ac - bac) / bac) * 100

    return round(float(variance_pct), 4)


# ── STEP 4: Engineer features for one project ─────────────────────────────────

def engineer_project(project_id, group, bac):
    """
    Run the full feature engineering pipeline for one project.
    Returns a dictionary of features or None if the project cannot
    be processed due to insufficient data.
    """
    group = group.sort_values('period_number').reset_index(drop=True)
    total_periods = len(group)

    # Need at least 8 periods to calculate meaningful features
    if total_periods < 8:
        print(f"  SKIPPED project {project_id} — only {total_periods} periods (minimum 8)")
        return None

    # Find the 20% cutoff period
    cutoff = get_prediction_period(group)
    window = group[group['period_number'] <= cutoff]

    if window.empty:
        print(f"  SKIPPED project {project_id} — empty prediction window")
        return None

    # Calculate all four features
    cpi_trend         = calc_cpi_trend(window)
    sv_trajectory     = calc_sv_trajectory(window, bac)
    burn_rate_ratio   = calc_burn_rate_ratio(window, bac)
    schedule_pressure = calc_schedule_pressure(window, bac)
    final_variance    = calc_final_variance(group, bac)

    # Check we have enough valid features to be useful
    features = [cpi_trend, sv_trajectory, burn_rate_ratio, schedule_pressure]
    null_count = sum(1 for f in features if f is None)
    if null_count > 1:
        print(f"  WARNING project {project_id} — {null_count} features are null")

    if final_variance is None:
        print(f"  SKIPPED project {project_id} — could not calculate final variance (no target)")
        return None
    
    # ── NEW: skip projects where final AC equals BAC exactly ──────────
    # These are data entry errors — the final cost was never recorded
    # and defaulted to the planned budget value
    final_ac = group.iloc[-1]['actual_cost']
    if abs(final_ac - bac) < 1.0:
        print(f"  SKIPPED project {project_id} — final AC equals BAC (unrecorded outcome)")
        return None

    return {
        'project_id':              project_id,
        'prediction_period':       int(cutoff),
        'total_periods_used':      int(total_periods),
        'cpi_trend':               cpi_trend,
        'sv_trajectory':           sv_trajectory,
        'burn_rate_ratio':         burn_rate_ratio,
        'schedule_pressure':       schedule_pressure,
        'actual_final_variance':   final_variance,
    }


# ── STEP 5: Save features to PostgreSQL ───────────────────────────────────────

def save_features(features_list):
    """
    Save all engineered feature rows to the features table.
    Clears any existing features first to avoid duplicates on re-runs.
    """
    if not features_list:
        print("  Nothing to save — features list is empty")
        return

    df = pd.DataFrame(features_list)

    with engine.connect() as conn:
        # Clear existing features to allow clean re-runs
        conn.execute(text("DELETE FROM features"))
        conn.commit()

    df.to_sql(
        name='features',
        con=engine,
        if_exists='append',
        index=False
    )
    print(f"  Saved {len(df)} feature rows to the features table")


# ── STEP 6: Verify features saved correctly ───────────────────────────────────

def verify_features():
    """Read back the features table and return a summary DataFrame."""
    query = """
        SELECT
            f.feature_id,
            f.project_id,
            pr.project_name,
            pr.train_test_flag,
            pr.project_size,
            f.prediction_period,
            f.total_periods_used,
            f.cpi_trend,
            f.sv_trajectory,
            f.burn_rate_ratio,
            f.schedule_pressure,
            f.actual_final_variance
        FROM features f
        JOIN projects pr ON f.project_id = pr.project_id
        ORDER BY pr.train_test_flag, f.project_id
    """
    return pd.read_sql(query, engine)


# ── STEP 7: Summary report ─────────────────────────────────────────────────────

def print_summary(features_df):
    """Print a clear summary of what was engineered."""
    print("\n" + "=" * 75)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 75)

    train_df = features_df[features_df['train_test_flag'] == 'train']
    test_df  = features_df[features_df['train_test_flag'] == 'test']

    print(f"\n  Total projects with features : {len(features_df)}")
    print(f"  Training set                 : {len(train_df)}")
    print(f"  Test set                     : {len(test_df)}")

    print(f"\n  Feature statistics (training set):")
    if not train_df.empty:
        stats = train_df[['cpi_trend', 'sv_trajectory',
                           'burn_rate_ratio', 'schedule_pressure',
                           'actual_final_variance']].describe().round(4)
        print(stats.to_string())

    print(f"\n  All projects:")
    print(f"\n  {'ID':<6} {'Name':<30} {'Set':<8} {'Cut':<6} "
          f"{'CPI':<8} {'SV':<12} {'Burn':<8} {'Pressure':<12} {'Variance%'}")
    print(f"  {'-'*6} {'-'*30} {'-'*8} {'-'*6} "
          f"{'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")

    for _, row in features_df.iterrows():
        pid      = int(row['project_id'])
        name     = str(row['project_name'])[:28]
        flag     = str(row['train_test_flag'])
        cutoff   = int(row['prediction_period'])
        cpi      = f"{row['cpi_trend']:.4f}"      if row['cpi_trend']         is not None else 'NULL'
        sv       = f"{row['sv_trajectory']:.2f}"  if row['sv_trajectory']     is not None else 'NULL'
        burn     = f"{row['burn_rate_ratio']:.4f}" if row['burn_rate_ratio']   is not None else 'NULL'
        pressure = f"{row['schedule_pressure']:.6f}" if row['schedule_pressure'] is not None else 'NULL'
        variance = f"{row['actual_final_variance']:.2f}%" if row['actual_final_variance'] is not None else 'NULL'
        print(f"  {pid:<6} {name:<30} {flag:<8} {cutoff:<6} "
              f"{cpi:<8} {sv:<12} {burn:<8} {pressure:<12} {variance}")

    # Quick sanity check
    print(f"\n  Sanity checks:")
    if not features_df.empty:
        over_budget  = (features_df['actual_final_variance'] > 0).sum()
        under_budget = (features_df['actual_final_variance'] < 0).sum()
        on_budget    = (features_df['actual_final_variance'] == 0).sum()
        avg_variance = features_df['actual_final_variance'].mean()
        print(f"    Over budget   : {over_budget} projects")
        print(f"    Under budget  : {under_budget} projects")
        print(f"    On budget     : {on_budget} projects")
        print(f"    Average variance : {avg_variance:.2f}%")

        null_features = features_df[['cpi_trend', 'sv_trajectory',
                                      'burn_rate_ratio', 'schedule_pressure']].isna().sum()
        if null_features.sum() > 0:
            print(f"\n  WARNING — null features found:")
            print(null_features[null_features > 0].to_string())
        else:
            print(f"    No null features — all good")

    print("=" * 75)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_feature_engineering():
    print("\nStarting feature engineering...\n")

    df = load_data()

    if df.empty:
        print("ERROR: No data loaded.")
        return

    print("\nEngineering features:")
    features_list = []
    skipped = 0

    for project_id, group in df.groupby('project_id'):
        bac          = group['bac'].iloc[0]
        project_name = group['project_name'].iloc[0]
        flag         = group['train_test_flag'].iloc[0]

        result = engineer_project(project_id, group, bac)

        if result is None:
            skipped += 1
            continue

        features_list.append(result)
        cutoff   = result['prediction_period']
        cpi      = result['cpi_trend']
        variance = result['actual_final_variance']
        print(f"  [{flag}] project {project_id} ({project_name[:25]}) — "
              f"cutoff: period {cutoff} | CPI: {cpi} | variance: {variance:.2f}%")

    print(f"\n  Engineered: {len(features_list)} projects")
    print(f"  Skipped   : {skipped} projects")

    if not features_list:
        print("\nERROR: No features were generated.")
        return

    # ── Remove duplicate projects based on identical feature vectors ──────────
    before = len(features_list)
    seen   = set()
    unique = []
    for f in features_list:
        sig = (
            round(f['cpi_trend'] or 0, 3),
            round(f['sv_trajectory'] or 0, 3),
            round(f['burn_rate_ratio'] or 0, 3),
            round(f['schedule_pressure'] or 0, 3)
        )
        if sig not in seen:
            seen.add(sig)
            unique.append(f)
        else:
            print(f"  REMOVED duplicate project {f['project_id']} "
                  f"— identical features to an existing project")

    features_list = unique
    after = len(features_list)
    if before != after:
        print(f"  Duplicates removed: {before - after}")

    # cap extreme target outliers to prevent skewing the model
    print("\nFinal target cleanup:")
    save_features(features_list)  # Save before cleanup to preserve original values in the database

    # Save to database
    print("\nSaving features:")
    save_features(features_list)

    # Verify and report
    features_df = verify_features()
    print_summary(features_df)

    print("\nFeature engineering complete. Run models/train_rf.py next.")


if __name__ == "__main__":
    run_feature_engineering()