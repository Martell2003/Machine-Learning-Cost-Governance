"""
evaluation/benchmark.py

Produces the technical benchmarking results for the dissertation.
Compares the Random Forest and ARIMA models against the traditional
EAC formula baseline on the held-out test set.

The EAC formula is the standard method organisations currently use
to forecast final project cost. If the ML models produce lower MAE
than EAC, the framework adds genuine predictive value.

Outputs:
    - Console report with all metrics
    - CSV saved to evaluation/results/benchmarking_results.csv
    - Summary table ready to paste into the dissertation

Run after train_rf.py and train_arima.py:
    python evaluation/benchmark.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import date
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.retrospective import MIN_LEAD_TIME
from pipeline.db import engine
from config import MODELS_SAVED

OUTPUT_DIR = 'evaluation/results'


# ── STEP 1: Load test set results ─────────────────────────────────────────────

def load_test_results():
    """
    Load the test set projects with their actual variances,
    RF forecasts and ARIMA forecasts from the database.
    """
    query = """
        SELECT
            pr.project_id,
            pr.project_name,
            pr.project_size,
            pr.budget_at_completion          AS bac,
            f.cpi_trend,
            f.sv_trajectory,
            f.burn_rate_ratio,
            f.schedule_pressure,
            f.actual_final_variance          AS actual,
            fc.rf_variance_pct               AS rf_predicted,
            fc.arima_variance_pct            AS arima_predicted,
            f.prediction_period,
            f.total_periods_used
        FROM features f
        JOIN projects  pr ON pr.project_id  = f.project_id
        LEFT JOIN forecasts fc ON fc.project_id = f.project_id
        WHERE pr.train_test_flag = 'test'
        AND   pr.quality_flag    = 'pass'
        ORDER BY f.project_id
    """
    df = pd.read_sql(query, engine)
    print(f"  Loaded {len(df)} test projects")
    return df


# ── STEP 2: EAC formula baseline ──────────────────────────────────────────────

def calc_eac_predictions(df):
    """
    Calculate the EAC formula prediction for each test project.

    Standard EAC formula:
        EAC = BAC / CPI
        Variance % = (EAC - BAC) / BAC * 100
                   = (1 / CPI - 1) * 100

    This is the method organisations currently use — the baseline
    that the ML models must beat to demonstrate added value.

    Why CPI-based EAC:
    The CPI-weighted EAC is the most widely used and cited formula
    in EVM practice (PMI, 2021) and the most accurate single-formula
    method in the literature (Batselier and Vanhoucke, 2015).
    """
    eac_predictions = []
    for _, row in df.iterrows():
        cpi = row['cpi_trend']
        if cpi and cpi > 0:
            eac_variance = (1 / cpi - 1) * 100
        else:
            eac_variance = 0.0
        eac_predictions.append(round(eac_variance, 4))
    return np.array(eac_predictions)


# ── STEP 3: Calculate metrics ─────────────────────────────────────────────────

def calc_metrics(actual, predicted, label, exclude_near_zero=True):
    """
    Calculate MAE, RMSE and MAPE for one set of predictions.

    Parameters:
        actual           — array of actual final variance values
        predicted        — array of predicted variance values
        label            — name of the model being evaluated
        exclude_near_zero — whether to exclude near-zero actuals
                           from MAPE (recommended — see note below)

    Note on MAPE:
    MAPE divides by the actual value. When actual variance is near
    zero (e.g. 0.5%), even a small prediction error produces a huge
    MAPE (e.g. predicting 3% when actual is 0.5% gives MAPE = 500%).
    MAPE is reported for completeness but MAE is the primary metric.
    """
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    # Remove any NaN predictions
    valid = ~np.isnan(predicted)
    actual_v    = actual[valid]
    predicted_v = predicted[valid]
    n_valid     = valid.sum()
    n_excluded  = (~valid).sum()

    if n_valid == 0:
        return None

    mae  = mean_absolute_error(actual_v, predicted_v)
    rmse = np.sqrt(mean_squared_error(actual_v, predicted_v))

    # MAPE on meaningful projects only
    meaningful = np.abs(actual_v) >= 1.0
    if meaningful.sum() > 0 and exclude_near_zero:
        mape     = mean_absolute_percentage_error(
            actual_v[meaningful], predicted_v[meaningful]
        ) * 100
        n_mape   = meaningful.sum()
        excluded = (~meaningful).sum()
    else:
        mape     = mean_absolute_percentage_error(
            actual_v, predicted_v
        ) * 100
        n_mape   = n_valid
        excluded = 0

    # Per-project errors
    errors    = np.abs(actual_v - predicted_v)
    within_5  = (errors <= 5).sum()
    within_10 = (errors <= 10).sum()
    within_15 = (errors <= 15).sum()

    return {
        'label':        label,
        'n_projects':   int(n_valid),
        'n_excluded':   int(n_excluded),
        'mae':          round(float(mae),  4),
        'rmse':         round(float(rmse), 4),
        'mape':         round(float(mape), 4),
        'mape_n':       int(n_mape),
        'mape_excluded':int(excluded),
        'within_5pct':  int(within_5),
        'within_10pct': int(within_10),
        'within_15pct': int(within_15),
        'errors':       errors.tolist(),
        'predictions':  predicted_v.tolist(),
        'actuals':      actual_v.tolist(),
    }


# ── STEP 4: Per-project breakdown ─────────────────────────────────────────────

def per_project_breakdown(df, eac_pred, rf_pred, arima_pred):
    """
    Build a per-project comparison table showing actual vs predicted
    for all three methods on every test project.
    """
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        actual = float(row['actual'])
        eac    = float(eac_pred[i])
        rf     = float(rf_pred[i]) if not np.isnan(rf_pred[i]) else None
        arima  = float(arima_pred[i]) \
                 if arima_pred is not None and i < len(arima_pred) \
                 and not np.isnan(arima_pred[i]) else None

        eac_err   = abs(actual - eac)
        rf_err    = abs(actual - rf)  if rf    is not None else None
        arima_err = abs(actual - arima) if arima is not None else None

        # Which model was closest
        errors = {'EAC': eac_err}
        if rf_err    is not None: errors['RF']    = rf_err
        if arima_err is not None: errors['ARIMA'] = arima_err
        winner = min(errors, key=errors.get)

        rows.append({
            'project_id':   int(row['project_id']),
            'project_name': row['project_name'],
            'project_size': row['project_size'],
            'actual_%':     round(actual, 4),
            'eac_%':        round(eac,    4),
            'rf_%':         round(rf,     4) if rf    is not None else None,
            'arima_%':      round(arima,  4) if arima is not None else None,
            'eac_error':    round(eac_err,   4),
            'rf_error':     round(rf_err,    4) if rf_err    is not None else None,
            'arima_error':  round(arima_err, 4) if arima_err is not None else None,
            'best_model':   winner,
            'prediction_period': int(row['prediction_period']),
            'total_periods':     int(row['total_periods_used']),
        })

    return pd.DataFrame(rows)


# ── STEP 5: Save results ──────────────────────────────────────────────────────

def save_results(breakdown_df, metrics_list):
    """
    Save benchmarking results to CSV files in evaluation/results/.
    These files feed directly into the dissertation Results chapter.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Per-project breakdown
    breakdown_path = os.path.join(
        OUTPUT_DIR, 'benchmarking_per_project.csv'
    )
    breakdown_df.to_csv(breakdown_path, index=False)
    print(f"  Saved: {breakdown_path}")

    # Summary metrics
    summary_rows = []
    for m in metrics_list:
        if m is None:
            continue
        summary_rows.append({
            'Model':        m['label'],
            'N Projects':   m['n_projects'],
            'MAE':          m['mae'],
            'RMSE':         m['rmse'],
            'MAPE':         m['mape'],
            'Within 5%':    m['within_5pct'],
            'Within 10%':   m['within_10pct'],
            'Within 15%':   m['within_15pct'],
        })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        OUTPUT_DIR, 'benchmarking_summary.csv'
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Save run metadata
    meta = {
        'run_date':    str(date.today()),
        'test_projects': int(len(breakdown_df)),
        'metrics':     [
            {k: v for k, v in m.items()
             if k not in ['errors', 'predictions', 'actuals']}
            for m in metrics_list if m is not None
        ]
    }
    meta_path = os.path.join(OUTPUT_DIR, 'benchmarking_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")


# ── STEP 6: Print dissertation-ready report ───────────────────────────────────

def print_report(all_findings, case_projects):
    """Print the retrospective evaluation report."""
    print("\n" + "=" * 70)
    print("RETROSPECTIVE CASE TESTING RESULTS")
    print("=" * 70)

    # Calculate counts correctly
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
        name      = str(f['project_name'])[:30]
        actual    = f"{f['actual_variance']:+.2f}%"
        tier      = f['actual_tier']
        alert_per = (f"Period {f['first_amber_period']}"
                     if f['first_amber_period'] else "None")
        lead      = (f"{f['lead_time_periods']} periods"
                     if f['lead_time_periods'] is not None else "N/A")
        adequate  = ("✓ Yes" if f['lead_time_adequate'] is True
                     else ("✗ No" if f['lead_time_adequate'] is False
                           else "N/A"))
        print(f"  {name:<32} {actual:<10} {tier:<10} "
              f"{alert_per:<12} {lead:<12} {adequate}")

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
    
    # Use the counts we calculated earlier, not the strings from the loop
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

def run_benchmarking():
    print("\nStarting technical benchmarking...\n")

    # Load test set
    print("Loading test set:")
    df = load_test_results()

    if df.empty:
        print("ERROR: No test projects found.")
        print("Make sure train_rf.py has been run and test projects "
              "exist in the database.")
        return

    # Check we have RF predictions
    n_rf = df['rf_predicted'].notna().sum()
    n_arima = df['arima_predicted'].notna().sum()
    print(f"  RF predictions available   : {n_rf}/{len(df)}")
    print(f"  ARIMA predictions available: {n_arima}/{len(df)}")

    # Extract arrays
    actual     = df['actual'].values
    rf_pred    = df['rf_predicted'].values
    arima_pred = df['arima_predicted'].values

    # EAC formula predictions
    print("\nCalculating EAC formula baseline:")
    eac_pred = calc_eac_predictions(df)
    print(f"  EAC predictions calculated for {len(eac_pred)} projects")

    # Calculate metrics for all three methods
    print("\nCalculating metrics:")
    eac_metrics   = calc_metrics(actual, eac_pred,   'EAC Formula')
    rf_metrics    = calc_metrics(actual, rf_pred,    'Random Forest')
    arima_metrics = calc_metrics(
        actual, arima_pred, 'ARIMA'
    ) if n_arima > 0 else None

    metrics_list = [eac_metrics, rf_metrics, arima_metrics]

    # Per-project breakdown
    print("\nBuilding per-project breakdown:")
    breakdown_df = per_project_breakdown(
        df, eac_pred, rf_pred, arima_pred
    )
    print(f"  Breakdown built for {len(breakdown_df)} projects")

    # Save results
    print("\nSaving results:")
    save_results(breakdown_df, metrics_list)

    # Print full report
    print_report(metrics_list, breakdown_df)

    print("\nBenchmarking complete.")
    print("Results saved to evaluation/results/")
    print("Run evaluation/retrospective.py next.")


if __name__ == "__main__":
    run_benchmarking()