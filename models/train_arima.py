"""
models/train_arima.py

Trains an ARIMA model for each project in the test set.
ARIMA works differently from Random Forest — instead of learning
patterns across many projects, it fits a time-series model to the
AC trajectory of each individual project and forecasts forward.

The two models complement each other:
    Random Forest — learns patterns ACROSS projects
    ARIMA         — models the trajectory WITHIN one project

Run after train_rf.py:
    python models/train_arima.py
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
import sys
from datetime import date
from sqlalchemy import text
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine
from config import MODELS_SAVED, MAE_TARGET


# ── STEP 1: Load data ─────────────────────────────────────────────────────────

def load_project_periods():
    """
    Load period-by-period AC time-series for all projects.
    ARIMA is fitted on the AC trajectory of each project individually.
    Also loads the prediction period cutoff from the features table
    so ARIMA uses the same 20% window as the Random Forest.
    """
    query = """
        SELECT
            pr.project_id,
            pr.project_name,
            pr.train_test_flag,
            pr.budget_at_completion  AS bac,
            p.period_number,
            p.actual_cost,
            p.earned_value,
            p.planned_value,
            f.prediction_period,
            f.actual_final_variance
        FROM projects pr
        JOIN periods p  ON p.project_id  = pr.project_id
        JOIN features f ON f.project_id  = pr.project_id
        WHERE pr.quality_flag = 'pass'
        ORDER BY pr.project_id, p.period_number
    """
    df = pd.read_sql(query, engine)

    train_count = df.groupby('project_id').first()
    train_n = (train_count['train_test_flag'] == 'train').sum()
    test_n  = (train_count['train_test_flag'] == 'test').sum()

    print(f"Loaded period data:")
    print(f"  Training projects : {train_n}")
    print(f"  Test projects     : {test_n}")
    print(f"  Total periods     : {len(df)}")

    return df


# ── STEP 2: Stationarity test ─────────────────────────────────────────────────

def check_stationarity(series):
    """
    Augmented Dickey-Fuller test to check if the time-series is stationary.
    ARIMA requires stationarity — a stable mean and variance over time.

    Returns:
        is_stationary : bool
        d             : int — number of differences needed (0 or 1)

    Why this matters:
    AC values in a project typically trend upward over time — they are
    cumulative. This means the series is NOT stationary and needs
    differencing before ARIMA can model it properly.
    """
    if len(series) < 4:
        return False, 1

    try:
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        d = 0 if is_stationary else 1
        return is_stationary, d
    except Exception:
        return False, 1


# ── STEP 3: Find best ARIMA order ─────────────────────────────────────────────

def find_best_order(series, d):
    """
    Search over p and q values to find the ARIMA order with the
    lowest AIC (Akaike Information Criterion).

    AIC balances model fit against complexity — a lower AIC means
    a better model without unnecessary parameters.

    p — AutoRegressive order: how many past values to use
    q — Moving Average order: how many past errors to use
    d — already determined by stationarity test
    """
    best_aic   = np.inf
    best_order = (1, d, 1)

    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic   = fitted.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    return best_order, best_aic

def get_arima_cutoff(group):
    """
    Get the period cutoff for ARIMA training window.
    Uses a higher percentage for short projects to ensure
    ARIMA has enough data points to fit a model.

    Short projects (< 20 periods) : use 40% cutoff
    Medium projects (20-50 periods): use 30% cutoff
    Long projects (> 50 periods)  : use 20% cutoff

    Why different thresholds:
    ARIMA needs a minimum of 4-5 data points to estimate parameters.
    The 20% rule works well for long projects but produces too few
    points for the short construction projects in DSLIB.
    """
    total = group['period_number'].max()
    if total < 20:
        pct = 0.40
    elif total < 50:
        pct = 0.30
    else:
        pct = 0.20
    return max(4, round(total * pct))


# ── STEP 4: Forecast final AC for one project ─────────────────────────────────

def forecast_project(project_id, group, bac):
    """
    Fit ARIMA on the first 20% of a project's AC time-series and
    forecast the remaining periods to estimate final AC.
    """
    group  = group.sort_values('period_number').reset_index(drop=True)
    cutoff = get_arima_cutoff(group)
    total  = len(group)
    bac    = float(bac)

    # Extract training window
    train_window = group[
        group['period_number'] <= cutoff
    ]['actual_cost'].dropna()

    full_series = group['actual_cost'].dropna()

    # ── Fix 2: raise minimum window to 4 periods ──────────────────────
    if len(train_window) < 4:
        print(f"  SKIPPED project {project_id} — "
              f"only {len(train_window)} periods in training window "
              f"(minimum 4)")
        return None

    # Stationarity check
    is_stationary, d = check_stationarity(train_window.values)

    # Find best ARIMA order
    best_order, best_aic = find_best_order(train_window.values, d)

    # Fit ARIMA
    try:
        model  = ARIMA(train_window.values, order=best_order)
        fitted = model.fit()
    except Exception as e:
        print(f"  SKIPPED project {project_id} — ARIMA fit failed: {e}")
        return None

    # Forecast remaining periods
    periods_remaining = max(1, total - cutoff)

    try:
        forecast_result = fitted.get_forecast(steps=periods_remaining)

        # Predicted mean — always a numpy array
        forecast_values   = np.array(forecast_result.predicted_mean)
        final_ac_forecast = float(forecast_values[-1])

        # ── Fix 1: handle conf_int as either DataFrame or numpy array ──
        raw_conf = forecast_result.conf_int(alpha=0.20)

        if hasattr(raw_conf, 'iloc'):
            # statsmodels older versions return a DataFrame
            lower = float(raw_conf.iloc[-1, 0])
            upper = float(raw_conf.iloc[-1, 1])
        else:
            # statsmodels newer versions return a numpy array
            raw_conf = np.array(raw_conf)
            lower    = float(raw_conf[-1, 0])
            upper    = float(raw_conf[-1, 1])

    except Exception as e:
        print(f"  SKIPPED project {project_id} — forecast failed: {e}")
        return None

    # Convert to variance percentage
    if bac <= 0:
        return None

    arima_variance  = ((final_ac_forecast - bac) / bac) * 100
    variance_lower  = ((lower - bac) / bac) * 100
    variance_upper  = ((upper - bac) / bac) * 100
    actual_variance = float(group['actual_final_variance'].iloc[0])
    actual_final_ac = float(full_series.iloc[-1])

    return {
        'project_id':        project_id,
        'arima_order':       best_order,
        'aic':               round(best_aic, 4),
        'cutoff_period':     cutoff,
        'total_periods':     total,
        'periods_forecast':  periods_remaining,
        'final_ac_forecast': round(final_ac_forecast, 2),
        'actual_final_ac':   round(actual_final_ac, 2),
        'bac':               bac,
        'arima_variance':    round(arima_variance, 4),
        'variance_lower':    round(variance_lower, 4),
        'variance_upper':    round(variance_upper, 4),
        'actual_variance':   actual_variance,
        'error':             round(abs(arima_variance - actual_variance), 4),
        'stationary':        is_stationary,
'd_used':            d,
    }

# ── STEP 5: Evaluate ARIMA across all projects ────────────────────────────────

def evaluate_arima(results_list, label=""):
    """
    Calculate MAE and RMSE across all ARIMA forecasts.
    Same metrics as Random Forest for direct comparison.
    """
    if not results_list:
        return None

    actuals    = np.array([r['actual_variance']  for r in results_list])
    predicted  = np.array([r['arima_variance']   for r in results_list])

    mae  = mean_absolute_error(actuals, predicted)
    rmse = np.sqrt(mean_squared_error(actuals, predicted))

    # MAPE on meaningful projects only
    meaningful = np.abs(actuals) >= 1.0
    if meaningful.sum() > 0:
        from sklearn.metrics import mean_absolute_percentage_error
        mape = mean_absolute_percentage_error(
            actuals[meaningful], predicted[meaningful]) * 100
    else:
        mape = None

    gate_passed = mae < MAE_TARGET

    if label:
        print(f"\n  {label} results:")
        if mape:
            print(f"    MAPE : {mape:.2f}%  "
                  f"(meaningful projects only)")
        print(f"    RMSE : {rmse:.4f}")
        print(f"    MAE  : {mae:.4f}  "
              f"← primary metric (target: below {MAE_TARGET}%)")
        if gate_passed:
            print(f"    ✓ MAE gate PASSED")
        else:
            print(f"    ✗ MAE gate not yet met")

    return {
        'mape':        round(mape, 4) if mape else None,
        'rmse':        round(rmse, 4),
        'mae':         round(mae,  4),
        'gate_passed': gate_passed
    }


# ── STEP 6: Compare RF vs ARIMA ───────────────────────────────────────────────

def compare_models(arima_results, rf_mae):
    """
    Compare ARIMA against Random Forest on the same test projects.
    This comparison is a key output for the dissertation evaluation chapter.
    """
    if not arima_results:
        return

    arima_mae = mean_absolute_error(
        [r['actual_variance'] for r in arima_results],
        [r['arima_variance']  for r in arima_results]
    )

    print(f"\n  RF vs ARIMA comparison (test set):")
    print(f"  {'Metric':<10} {'Random Forest':<18} {'ARIMA':<18} {'Winner'}")
    print(f"  {'-'*10} {'-'*18} {'-'*18} {'-'*10}")

    rf_winner   = "RF ✓"    if rf_mae   < arima_mae else "ARIMA ✓"
    print(f"  {'MAE':<10} {rf_mae:<18.4f} {arima_mae:<18.4f} {rf_winner}")

    print()
    if rf_mae < arima_mae:
        diff = round((arima_mae - rf_mae) / arima_mae * 100, 1)
        print(f"  Random Forest outperforms ARIMA by {diff}% on MAE")
        print(f"  RF is the stronger model for cross-project forecasting")
    else:
        diff = round((rf_mae - arima_mae) / rf_mae * 100, 1)
        print(f"  ARIMA outperforms Random Forest by {diff}% on MAE")
        print(f"  ARIMA captures individual project trajectories more accurately")

    print(f"\n  Combined insight:")
    print(f"  Using both models together provides two independent forecasts.")
    print(f"  When RF and ARIMA agree — confidence in the forecast is higher.")
    print(f"  When they diverge — this itself is a signal worth investigating.")


# ── STEP 7: Save ARIMA results to database ────────────────────────────────────

def save_arima_results(results_list):
    """
    Update the forecasts table with ARIMA variance predictions.
    Matches on project_id and forecast_period.
    If no existing RF forecast row exists, inserts a new one.
    """
    if not results_list:
        return

    with engine.connect() as conn:
        for r in results_list:
            # Try to update existing forecast row from RF training
            result = conn.execute(
                text("""
                    UPDATE forecasts
                    SET arima_variance_pct = :arima_variance
                    WHERE project_id     = :project_id
                    AND   forecast_period = :cutoff
                    RETURNING forecast_id
                """),
                {
                    'arima_variance': r['arima_variance'],
                    'project_id':     r['project_id'],
                    'cutoff':         r['cutoff_period']
                }
            )

            # If no RF row exists yet, insert a new row
            if result.rowcount == 0:
                conn.execute(
                    text("""
                        INSERT INTO forecasts
                            (project_id, forecast_period,
                             rf_variance_pct, arima_variance_pct,
                             generated_at)
                        VALUES
                            (:project_id, :cutoff,
                             NULL, :arima_variance,
                             NOW())
                    """),
                    {
                        'project_id':     r['project_id'],
                        'cutoff':         r['cutoff_period'],
                        'arima_variance': r['arima_variance']
                    }
                )
        conn.commit()

    print(f"  Saved {len(results_list)} ARIMA forecasts to database")


# ── STEP 8: Save ARIMA configs ────────────────────────────────────────────────

def save_arima_configs(results_list):
    """
    Save the best ARIMA order per project to a JSON file.
    Converts all numpy types to native Python types before
    serialising — JSON cannot handle numpy bool_, int64, float64.
    """
    os.makedirs(MODELS_SAVED, exist_ok=True)

    configs = {}
    for r in results_list:
        configs[str(r['project_id'])] = {
            'order':        [int(x) for x in r['arima_order']],
            'aic':          float(r['aic']),
            'd_used':       int(r['d_used']),
            'stationary':   bool(r['stationary']),   # converts numpy bool_ to Python bool
            'trained_date': str(date.today())
        }

    config_path = os.path.join(MODELS_SAVED, 'arima_configs.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"  ARIMA configs saved : {config_path}")


# ── STEP 9: Summary report ────────────────────────────────────────────────────

def print_summary(results_list, train_results, test_results):
    """Print a full summary of ARIMA results."""
    print("\n" + "=" * 70)
    print("ARIMA TRAINING SUMMARY")
    print("=" * 70)

    print(f"\n  Projects fitted   : {len(results_list)}")
    train_n = sum(1 for r in results_list
                  if r.get('split') == 'train')
    test_n  = len(results_list) - train_n

    # Most common ARIMA orders
    from collections import Counter
    orders = Counter(str(r['arima_order']) for r in results_list)
    print(f"\n  Most common ARIMA orders:")
    for order, count in orders.most_common(5):
        print(f"    {order:<15} — {count} projects")

    # Per-project results
    if test_results:
        print(f"\n  Test set predictions:")
        print(f"  {'ID':<6} {'Project':<30} {'Actual%':<12} "
              f"{'ARIMA%':<12} {'Error%':<10} {'Order'}")
        print(f"  {'-'*6} {'-'*30} {'-'*12} "
              f"{'-'*12} {'-'*10} {'-'*12}")
        for r in test_results:
            name  = str(r['project_id'])
            proj  = f"project {r['project_id']}"
            act   = r['actual_variance']
            pred  = r['arima_variance']
            err   = r['error']
            order = str(r['arima_order'])
            print(f"  {name:<6} {proj:<30} {act:<12.2f} "
                  f"{pred:<12.2f} {err:<10.2f} {order}")

    print("=" * 70)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_arima_training():
    print("\nStarting ARIMA model training...\n")

    # Load data
    df = load_project_periods()

    if df.empty:
        print("ERROR: No data loaded.")
        return

    # Load RF MAE for comparison
    # Read from saved model metadata if available
    rf_mae = None
    meta_path = os.path.join(MODELS_SAVED, 'random_forest_baseline_meta.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(MODELS_SAVED, 'random_forest_tuned_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta   = json.load(f)
            rf_mae = meta.get('mae')
            print(f"Random Forest MAE loaded from metadata: {rf_mae}")

    # Fit ARIMA for all projects
    print("\nFitting ARIMA models:")
    all_results   = []
    train_results = []
    test_results  = []
    skipped       = 0

    for project_id, group in df.groupby('project_id'):
        bac        = group['bac'].iloc[0]
        flag       = group['train_test_flag'].iloc[0]
        name       = group['project_name'].iloc[0]

        result = forecast_project(project_id, group, bac)

        if result is None:
            skipped += 1
            continue

        result['split'] = flag
        all_results.append(result)

        order = result['arima_order']
        var   = result['arima_variance']
        act   = result['actual_variance']
        err   = result['error']

        print(f"  [{flag}] project {project_id} "
              f"({name[:22]}) — "
              f"order: {order} | "
              f"forecast: {var:.2f}% | "
              f"actual: {act:.2f}% | "
              f"error: {err:.2f}%")

        if flag == 'train':
            train_results.append(result)
        else:
            test_results.append(result)

    print(f"\n  Fitted  : {len(all_results)} projects")
    print(f"  Skipped : {skipped} projects")

    # Evaluate
    print("\nEvaluating ARIMA:")
    train_metrics = evaluate_arima(train_results, "Training set") \
                    if train_results else None
    test_metrics  = evaluate_arima(test_results,  "Test set") \
                    if test_results  else None

    # Compare RF vs ARIMA
    if rf_mae and test_results:
        compare_models(test_results, rf_mae)

    # Save results
    print("\nSaving results:")
    save_arima_results(test_results)
    save_arima_configs(all_results)

    # Summary
    print_summary(all_results, train_results, test_results)

    print("\nARIMA training complete. Run explainability/shap_engine.py next.")


if __name__ == "__main__":
    run_arima_training()