"""
governance/decision_matrix.py

Translates every ML forecast into a specific management action.
This is the governance layer — the component that makes the framework
operationally useful rather than just analytically interesting.

A forecast percentage on its own tells a manager nothing actionable.
The decision matrix maps every forecast to:
    - A risk tier (Green / Amber / Red / Critical)
    - A named responsible role
    - A specific prescribed action
    - A defined response window
    - An automated alert record in the database

Grounded in MSP and PRINCE2 governance frameworks (OGC, 2011)
and EVM variance analysis literature (PMI, 2021).

Run after shap_engine.py:
    python governance/decision_matrix.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime, date
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine


# ── Governance tier definitions ───────────────────────────────────────────────
# Each tier defines the threshold, colour, responsible role,
# prescribed action and response window.
# Thresholds are grounded in EVM variance analysis literature (PMI, 2021)

TIERS = {
    1: {
        'name':            'GREEN',
        'colour':          '#27AE60',
        'label':           'Monitor',
        'min_variance':    -5.0,
        'max_variance':     5.0,
        'owner':           'IT Programme Manager',
        'action':          (
            'No intervention required. Continue standard reporting cycle. '
            'Review forecast at next scheduled gate.'
        ),
        'window':          'Next scheduled gate review',
        'escalation':      None,
        'description':     (
            'Project is performing within acceptable tolerance. '
            'Cost variance is within ±5% of budget.'
        )
    },
    2: {
        'name':            'AMBER',
        'colour':          '#E67E22',
        'label':           'PMO Review',
        'min_variance':     5.0,
        'max_variance':    10.0,
        'owner':           'PMO Lead',
        'action':          (
            'Schedule PMO review meeting within 14 days. '
            'IT Programme Manager to prepare variance analysis report. '
            'Review SHAP output to identify root cause cost drivers. '
            'Update risk register.'
        ),
        'window':          'Within 14 days',
        'escalation':      'PMO Lead → Programme Board if variance increases',
        'description':     (
            'Cost variance is trending between 5% and 10%. '
            'Early management attention required before variance grows.'
        )
    },
    3: {
        'name':            'RED',
        'colour':          '#E74C3C',
        'label':           'Escalate',
        'min_variance':    10.0,
        'max_variance':    20.0,
        'owner':           'PMO Lead',
        'action':          (
            'Formal escalation to programme board within 7 days. '
            'Initiate change control process. '
            'PMO Lead to present recovery options with cost implications. '
            'Mandatory root cause analysis before next reporting period.'
        ),
        'window':          'Within 7 days',
        'escalation':      'Mandatory programme board presentation',
        'description':     (
            'Cost variance is between 10% and 20%. '
            'Formal governance response required immediately.'
        )
    },
    4: {
        'name':            'CRITICAL',
        'colour':          '#922B21',
        'label':           'Executive',
        'min_variance':    20.0,
        'max_variance':    float('inf'),
        'owner':           'Programme Sponsor',
        'action':          (
            'Immediate executive intervention required within 48 hours. '
            'Project placed on formal review status. '
            'Delivery continuation decision to be made by Programme Sponsor. '
            'All discretionary spend frozen pending review.'
        ),
        'window':          'Within 48 hours',
        'escalation':      'Programme Sponsor must make continuation decision',
        'description':     (
            'Cost variance exceeds 20%. '
            'Executive decision on project continuation required.'
        )
    }
}

# Underrun tiers — project is coming in under budget
# Underruns also need governance attention — they may indicate
# scope reduction, deferred work, or inaccurate tracking
UNDERRUN_TIERS = {
    1: {
        'name':    'GREEN',
        'colour':  '#27AE60',
        'label':   'Monitor',
        'min_variance': -5.0,
        'max_variance':  0.0,
        'owner':   'IT Programme Manager',
        'action':  (
            'No intervention required. '
            'Monitor to confirm underrun is not due to deferred scope.'
        ),
        'window':  'Next scheduled gate review',
        'description': 'Project is within ±5% tolerance.'
    },
    5: {
        'name':    'AMBER',
        'colour':  '#E67E22',
        'label':   'Review Underrun',
        'min_variance': -10.0,
        'max_variance':  -5.0,
        'owner':   'PMO Lead',
        'action':  (
            'PMO review within 14 days to confirm underrun cause. '
            'Verify no scope has been deferred or dropped. '
            'Update forecasts if scope reduction is confirmed.'
        ),
        'window':  'Within 14 days',
        'description': (
            'Significant underrun detected. '
            'Verify this reflects genuine efficiency not deferred scope.'
        )
    },
    6: {
        'name':    'RED',
        'colour':  '#E74C3C',
        'label':   'Investigate Underrun',
        'min_variance': float('-inf'),
        'max_variance': -10.0,
        'owner':   'PMO Lead',
        'action':  (
            'Formal investigation required within 7 days. '
            'Significant underrun may indicate scope reduction, '
            'data quality issues, or inaccurate cost tracking. '
            'Report findings to programme board.'
        ),
        'window':  'Within 7 days',
        'description': (
            'Underrun exceeds 10%. '
            'Formal investigation required to confirm data integrity.'
        )
    }
}


# ── STEP 1: Classify a single variance value ──────────────────────────────────

def classify_variance(variance_pct):
    """
    Map a predicted cost variance percentage to a governance tier.

    Positive variance = over budget (cost overrun)
    Negative variance = under budget (cost underrun)

    Returns the full tier dictionary for the matched tier.
    """
    v = float(variance_pct)

    # Overrun tiers
    if v >= 20.0:
        return {**TIERS[4], 'tier_number': 4, 'variance': v}
    elif v >= 10.0:
        return {**TIERS[3], 'tier_number': 3, 'variance': v}
    elif v >= 5.0:
        return {**TIERS[2], 'tier_number': 2, 'variance': v}
    elif v >= -5.0:
        return {**TIERS[1], 'tier_number': 1, 'variance': v}
    elif v >= -10.0:
        return {**UNDERRUN_TIERS[5], 'tier_number': 5, 'variance': v}
    else:
        return {**UNDERRUN_TIERS[6], 'tier_number': 6, 'variance': v}


# ── STEP 2: Generate alert card ───────────────────────────────────────────────

def generate_alert_card(project_id, project_name, rf_variance,
                        arima_variance, explanation, bac):
    """
    Generate a complete alert card for one project combining
    RF forecast, ARIMA forecast, governance tier and prescribed action.

    The alert card is what the dashboard displays to the user.
    It contains everything a manager needs to know and do —
    no interpretation required.

    Returns a dictionary representing one complete alert.
    """
    # Classify based on RF variance (primary model)
    tier = classify_variance(rf_variance)

    # Check if RF and ARIMA agree
    model_agreement = None
    confidence      = 'single model'
    if arima_variance is not None:
        arima_tier     = classify_variance(arima_variance)
        models_agree   = tier['tier_number'] == arima_tier['tier_number']
        model_agreement = models_agree

        if models_agree:
            confidence = 'high — both models agree'
        else:
            diff = abs(rf_variance - arima_variance)
            if diff < 5:
                confidence = 'moderate — models within 5% of each other'
            else:
                confidence = (
                    f'low — models diverge by {diff:.1f}%. '
                    f'RF: {rf_variance:.1f}%  ARIMA: {arima_variance:.1f}%'
                )

    # Calculate estimated financial impact
    financial_impact = None
    if bac and bac > 0:
        financial_impact = round((rf_variance / 100) * bac, 2)

    # Lead time message
    lead_time_msg = (
        "This alert has been generated at the 20% project completion mark "
        "— providing a minimum 4-week lead time for management intervention "
        "before further budget is committed."
    )

    return {
        'project_id':         project_id,
        'project_name':       project_name,
        'tier_number':        tier['tier_number'],
        'tier_name':          tier['name'],
        'tier_colour':        tier['colour'],
        'tier_label':         tier['label'],
        'rf_variance':        round(rf_variance, 4),
        'arima_variance':     round(arima_variance, 4) if arima_variance else None,
        'model_agreement':    model_agreement,
        'confidence':         confidence,
        'prescribed_action':  tier['action'],
        'responsible_owner':  tier['owner'],
        'response_window':    tier['window'],
        'tier_description':   tier['description'],
        'financial_impact':   financial_impact,
        'explanation':        explanation,
        'lead_time_message':  lead_time_msg,
        'generated_at':       str(datetime.now()),
        'alert_date':         str(date.today()),
    }


# ── STEP 3: Load forecasts from database ─────────────────────────────────────

def load_forecasts():
    """
    Load all project forecasts from the database.
    Joins with projects and features to get everything needed
    for alert generation in one query.
    """
    query = """
        SELECT
            pr.project_id,
            pr.project_name,
            pr.budget_at_completion   AS bac,
            pr.train_test_flag,
            f.rf_variance_pct,
            f.arima_variance_pct,
            f.explanation,
            f.forecast_period
        FROM forecasts f
        JOIN projects pr ON f.project_id = pr.project_id
        WHERE pr.quality_flag = 'pass'
        ORDER BY ABS(f.rf_variance_pct) DESC
    """
    df = pd.read_sql(query, engine)
    print(f"  Loaded {len(df)} forecasts from database")
    return df


# ── STEP 4: Save alerts to database ──────────────────────────────────────────

def save_alerts(alert_cards):
    """
    Save all generated alerts to the alerts table.
    Clears existing alerts first to allow clean re-runs.
    """
    if not alert_cards:
        print("  No alerts to save")
        return

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM alerts"))
        conn.commit()

    saved = 0
    with engine.connect() as conn:
        for card in alert_cards:
            # Get forecast_id for this project
            result = conn.execute(
                text("""
                    SELECT forecast_id FROM forecasts
                    WHERE project_id = :pid
                    ORDER BY generated_at DESC
                    LIMIT 1
                """),
                {'pid': card['project_id']}
            )
            row = result.fetchone()
            if row is None:
                continue
            forecast_id = row[0]

            conn.execute(
                text("""
                    INSERT INTO alerts (
                        forecast_id,
                        project_id,
                        tier,
                        status,
                        prescribed_action,
                        responsible_role,
                        response_window,
                        acknowledged,
                        triggered_at
                    ) VALUES (
                        :forecast_id,
                        :project_id,
                        :tier,
                        :status,
                        :action,
                        :role,
                        :window,
                        FALSE,
                        NOW()
                    )
                """),
                {
                    'forecast_id': forecast_id,
                    'project_id':  card['project_id'],
                    'tier':        card['tier_number'],
                    'status':      card['tier_name'],
                    'action':      card['prescribed_action'],
                    'role':        card['responsible_owner'],
                    'window':      card['response_window'],
                }
            )
            saved += 1

        conn.commit()

    print(f"  Saved {saved} alert records to database")


# ── STEP 5: PMO Playbook entry ────────────────────────────────────────────────

def generate_playbook_entry(alert_card):
    """
    Generate a structured playbook entry for one project alert.
    The playbook is the governance document that tells an adopting
    organisation how to respond to each alert type.

    These entries are saved to governance/playbook.md
    """
    lines = [
        f"## Project: {alert_card['project_name']}",
        f"",
        f"**Alert Status:** {alert_card['tier_name']} — "
        f"{alert_card['tier_label']}",
        f"**Predicted Variance:** {alert_card['rf_variance']:.2f}%",
    ]

    if alert_card['arima_variance'] is not None:
        lines.append(
            f"**ARIMA Forecast:** {alert_card['arima_variance']:.2f}%"
        )

    lines += [
        f"**Model Confidence:** {alert_card['confidence']}",
        f"",
    ]

    if alert_card['financial_impact'] is not None:
        impact = alert_card['financial_impact']
        sign   = '+' if impact > 0 else ''
        lines.append(
            f"**Estimated Financial Impact:** "
            f"{sign}{impact:,.2f} (currency units)"
        )
        lines.append(f"")

    lines += [
        f"**Responsible Owner:** {alert_card['responsible_owner']}",
        f"**Response Window:** {alert_card['response_window']}",
        f"",
        f"**Prescribed Action:**",
        f"{alert_card['prescribed_action']}",
        f"",
        f"**Why this forecast was made:**",
        f"{alert_card['explanation'] or 'See SHAP waterfall chart.'}",
        f"",
        f"**Lead Time Note:**",
        f"{alert_card['lead_time_message']}",
        f"",
        f"---",
        f"",
    ]

    return '\n'.join(lines)


def save_playbook(alert_cards):
    """
    Write the PMO Implementation Playbook as a markdown file.
    Organised by tier — Critical first, then Red, Amber, Green.
    """
    os.makedirs('governance', exist_ok=True)
    path = 'governance/playbook.md'

    lines = [
        "# PMO Implementation Playbook",
        "",
        "Generated by the ML-Enhanced Cost Governance Framework",
        f"Date: {date.today()}",
        "",
        "This playbook translates ML forecast outputs into specific "
        "management actions aligned with MSP and PRINCE2 governance cycles.",
        "",
        "---",
        "",
    ]

    # Sort by tier — most critical first
    tier_order = {4: 0, 3: 1, 2: 2, 6: 3, 5: 4, 1: 5}
    sorted_cards = sorted(
        alert_cards,
        key=lambda x: tier_order.get(x['tier_number'], 99)
    )

    current_tier = None
    for card in sorted_cards:
        if card['tier_number'] != current_tier:
            current_tier = card['tier_number']
            tier_name    = card['tier_name']
            lines.append(f"# {tier_name} ALERTS")
            lines.append("")

        lines.append(generate_playbook_entry(card))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  PMO Playbook saved: {path}")


# ── STEP 6: Summary report ────────────────────────────────────────────────────

def print_summary(alert_cards):
    """Print a complete summary of all generated alerts."""
    print("\n" + "=" * 70)
    print("GOVERNANCE LAYER SUMMARY")
    print("=" * 70)

    # Tier distribution
    from collections import Counter
    tier_counts = Counter(c['tier_name'] for c in alert_cards)

    print(f"\n  Alert distribution:")
    tier_display_order = ['CRITICAL', 'RED', 'AMBER', 'GREEN']
    for tier_name in tier_display_order:
        count = tier_counts.get(tier_name, 0)
        if count > 0:
            tier_data = next(
                (t for t in TIERS.values() if t['name'] == tier_name),
                None
            )
            bar = '█' * count
            print(f"    {tier_name:<12} : {count:>3}  {bar}")

    # Full alert table
    print(f"\n  Full alert table:")
    print(f"  {'ID':<6} {'Project':<32} {'RF%':<10} "
          f"{'ARIMA%':<10} {'Tier':<12} {'Owner':<25} {'Window'}")
    print(f"  {'-'*6} {'-'*32} {'-'*10} "
          f"{'-'*10} {'-'*12} {'-'*25} {'-'*18}")

    for card in sorted(
        alert_cards,
        key=lambda x: abs(x['rf_variance']),
        reverse=True
    ):
        pid    = card['project_id']
        name   = card['project_name'][:30]
        rf     = f"{card['rf_variance']:.2f}%"
        arima  = (f"{card['arima_variance']:.2f}%"
                  if card['arima_variance'] is not None else 'N/A')
        tier   = card['tier_name']
        owner  = card['responsible_owner'][:23]
        window = card['response_window']
        print(f"  {pid:<6} {name:<32} {rf:<10} "
              f"{arima:<10} {tier:<12} {owner:<25} {window}")

    # Model agreement summary
    agreed = sum(
        1 for c in alert_cards
        if c['model_agreement'] is True
    )
    disagreed = sum(
        1 for c in alert_cards
        if c['model_agreement'] is False
    )
    no_arima = sum(
        1 for c in alert_cards
        if c['model_agreement'] is None
    )

    print(f"\n  Model agreement (RF vs ARIMA):")
    print(f"    Both models agree  : {agreed}")
    print(f"    Models disagree    : {disagreed}")
    print(f"    ARIMA not available: {no_arima}")

    if disagreed > 0:
        print(f"\n  Projects where models disagree "
              f"(investigate further):")
        for card in alert_cards:
            if card['model_agreement'] is False:
                print(f"    project {card['project_id']} "
                      f"— RF: {card['rf_variance']:.2f}%  "
                      f"ARIMA: {card['arima_variance']:.2f}%  "
                      f"— {card['confidence']}")

    print("=" * 70)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_decision_matrix():
    print("\nRunning governance decision matrix...\n")

    # Load all forecasts
    print("Loading forecasts:")
    forecasts_df = load_forecasts()

    if forecasts_df.empty:
        print("ERROR: No forecasts found.")
        print("Run models/train_rf.py and models/train_arima.py first.")
        return

    # Generate alert card for each project
    print("\nGenerating alert cards:")
    alert_cards = []

    for _, row in forecasts_df.iterrows():
        if row['rf_variance_pct'] is None:
            continue

        card = generate_alert_card(
            project_id    = int(row['project_id']),
            project_name  = row['project_name'],
            rf_variance   = float(row['rf_variance_pct']),
            arima_variance= float(row['arima_variance_pct'])
                            if row['arima_variance_pct'] is not None
                            else None,
            explanation   = row['explanation'],
            bac           = float(row['bac'])
                            if row['bac'] is not None else None
        )
        alert_cards.append(card)

        tier_label = card['tier_name']
        rf         = card['rf_variance']
        window     = card['response_window']
        print(f"  project {card['project_id']:<6} "
              f"({row['project_name'][:28]:<28}) — "
              f"{tier_label:<10} "
              f"RF: {rf:+.2f}%  "
              f"→ {window}")

    print(f"\n  Generated {len(alert_cards)} alert cards")

    # Save alerts to database
    print("\nSaving to database:")
    save_alerts(alert_cards)

    # Save PMO playbook
    save_playbook(alert_cards)

    # Print summary
    print_summary(alert_cards)

    print("\nGovernance layer complete. Run coordinator/ml_coordinator.py next.")


if __name__ == "__main__":
    run_decision_matrix()