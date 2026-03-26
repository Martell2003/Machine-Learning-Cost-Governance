"""
coordinator/ml_coordinator.py

The ML Coordinator is the central orchestrator of the framework.
It ties all previous components together into a single function call.

When the dashboard requests a forecast for a project, the coordinator:
    1. Fetches project data from PostgreSQL
    2. Loads the trained Random Forest model
    3. Produces an RF forecast
    4. Loads the ARIMA config and produces an ARIMA forecast
    5. Runs SHAP to explain the RF forecast
    6. Passes both forecasts to the governance layer
    7. Returns one complete package to the dashboard

No component calls another directly — everything routes through here.
This keeps each component independently testable and replaceable.

Usage:
    from coordinator.ml_coordinator import MLCoordinator
    coordinator = MLCoordinator()
    result = coordinator.run_forecast(project_id=42)
"""

import numpy as np
import pandas as pd
import joblib
import shap
import json
import os
import sys
from datetime import datetime
from sqlalchemy import text
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine
from config import MODELS_SAVED
from governance.decision_matrix import classify_variance, generate_alert_card


# ── Feature columns — must match train_rf.py exactly ─────────────────────────

FEATURE_COLS = [
    'cpi_trend',
    'sv_trajectory',
    'burn_rate_ratio',
    'schedule_pressure'
]

FEATURE_LABELS = {
    'cpi_trend':         'Cost Performance (CPI)',
    'sv_trajectory':     'Schedule Variance (SV)',
    'burn_rate_ratio':   'Budget Burn Rate',
    'schedule_pressure': 'Schedule Pressure'
}


# ══════════════════════════════════════════════════════════════════════════════
# ML COORDINATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MLCoordinator:
    """
    Central orchestrator for the ML Cost Governance Framework.

    Initialise once and reuse across multiple forecast requests.
    The model and explainer are loaded once at startup and cached
    in memory — this prevents reloading from disk on every request
    which would make the dashboard slow.

    Usage:
        coordinator = MLCoordinator()
        result = coordinator.run_forecast(project_id=42)
        result = coordinator.run_forecast(project_id=19)
    """

    def __init__(self):
        self.rf_model      = None
        self.explainer     = None
        self.base_value    = None
        self.arima_configs = None
        self.model_version = None
        self._load_models()


    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        """
        Load the Random Forest model, SHAP explainer and ARIMA configs
        at initialisation time. Raises clear errors if any are missing.
        """
        # Load Random Forest
        tuned_path    = os.path.join(
            MODELS_SAVED, 'random_forest_tuned.pkl'
        )
        baseline_path = os.path.join(
            MODELS_SAVED, 'random_forest_baseline.pkl'
        )

        if os.path.exists(tuned_path):
            self.rf_model      = joblib.load(tuned_path)
            self.model_version = 'tuned'
        elif os.path.exists(baseline_path):
            self.rf_model      = joblib.load(baseline_path)
            self.model_version = 'baseline'
        else:
            raise FileNotFoundError(
                "No trained Random Forest model found in models/saved/. "
                "Run models/train_rf.py first."
            )

        # Load SHAP Tree Explainer
        self.explainer  = shap.TreeExplainer(self.rf_model)
        self.base_value = float(
            np.atleast_1d(self.explainer.expected_value)[0]
        )

        # Load ARIMA configs
        arima_path = os.path.join(MODELS_SAVED, 'arima_configs.json')
        if os.path.exists(arima_path):
            with open(arima_path) as f:
                self.arima_configs = json.load(f)
        else:
            self.arima_configs = {}

        print(f"MLCoordinator initialised:")
        print(f"  RF model version : {self.model_version}")
        print(f"  SHAP base value  : {self.base_value:.4f}%")
        print(f"  ARIMA configs    : {len(self.arima_configs)} projects")


    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_project(self, project_id):
        """
        Fetch all data needed for one project forecast.
        Returns project metadata, engineered features and
        period-by-period AC time-series.
        """
        from sqlalchemy import text
        # Project metadata and features
        features_query = text("""
            SELECT
                pr.project_id,
                pr.project_name,
                pr.budget_at_completion  AS bac,
                pr.train_test_flag,
                pr.project_size,
                f.feature_id,
                f.prediction_period,
                f.cpi_trend,
                f.sv_trajectory,
                f.burn_rate_ratio,
                f.schedule_pressure,
                f.actual_final_variance
            FROM features f
            JOIN projects pr ON f.project_id = pr.project_id
            WHERE pr.project_id = :project_id
            AND   pr.quality_flag = 'pass'
        """)

        with engine.connect() as conn:
            result = conn.execute(features_query, {'project_id': project_id})
            rows = result.fetchall()
            if not rows:
                return None, None
            columns = result.keys()
            features_df = pd.DataFrame(rows, columns=columns)

        # Period time-series for ARIMA
        periods_query = text("""
            SELECT
                period_number,
                actual_cost,
                earned_value,
                planned_value
            FROM periods
            WHERE project_id = :project_id
            ORDER BY period_number
        """)

        with engine.connect() as conn:
            result = conn.execute(periods_query, {"project_id": project_id})
            rows = result.fetchall()
            if not rows:
                periods_df = pd.DataFrame()
            else:
                columns = result.keys()
                periods_df = pd.DataFrame(rows, columns=columns)

        return features_df.iloc[0], periods_df

    # ── Random Forest forecast ────────────────────────────────────────────────

    def _rf_forecast(self, project_row):
        """
        Produce a Random Forest cost variance forecast for one project.
        Returns the predicted variance percentage.
        """
        X = np.array([[
            float(project_row['cpi_trend']),
            float(project_row['sv_trajectory']),
            float(project_row['burn_rate_ratio']),
            float(project_row['schedule_pressure'])
        ]])

        prediction = float(self.rf_model.predict(X)[0])
        return round(prediction, 4)


    # ── SHAP explanation ──────────────────────────────────────────────────────

    def _shap_explain(self, project_row):
        """
        Generate SHAP feature attribution for one project forecast.

        Returns a list of dictionaries — one per feature — showing
        how much each feature pushed the prediction up or down.
        The sum of all SHAP values plus base_value = RF prediction.
        """
        X = np.array([[
            float(project_row['cpi_trend']),
            float(project_row['sv_trajectory']),
            float(project_row['burn_rate_ratio']),
            float(project_row['schedule_pressure'])
        ]])

        shap_vals  = self.explainer.shap_values(X)[0]
        prediction = self.base_value + float(shap_vals.sum())

        attributions = []
        for i, feature in enumerate(FEATURE_COLS):
            sv = float(shap_vals[i])
            attributions.append({
                'feature':       feature,
                'label':         FEATURE_LABELS[feature],
                'feature_value': float(project_row[feature]),
                'shap_value':    round(sv, 6),
                'direction':     'up' if sv > 0 else 'down',
                'abs_impact':    round(abs(sv), 6),
            })

        # Sort by absolute impact — biggest driver first
        attributions.sort(key=lambda x: x['abs_impact'], reverse=True)

        return {
            'attributions': attributions,
            'base_value':   round(self.base_value, 4),
            'prediction':   round(prediction, 4),
            'explanation':  self._build_explanation(
                                attributions, prediction
                            )
        }


    def _build_explanation(self, attributions, prediction):
        """
        Build a plain-language explanation sentence from SHAP values.
        This is what the dashboard shows to the programme manager.
        """
        direction   = "overrun" if prediction > 0 else "underrun"
        top_two     = attributions[:2]
        parts       = []

        for attr in top_two:
            impact = attr['abs_impact']
            label  = attr['label']
            word   = "adding" if attr['direction'] == 'up' \
                     else "reducing by"
            parts.append(
                f"{label} ({word} {impact:.1f} percentage points)"
            )

        if len(parts) == 1:
            drivers = parts[0]
        else:
            drivers = f"{parts[0]}, compounded by {parts[1]}"

        return (
            f"The forecast of {abs(prediction):.1f}% cost {direction} "
            f"is primarily driven by {drivers}. "
            f"Model baseline: {self.base_value:.1f}%."
        )


    # ── ARIMA forecast ────────────────────────────────────────────────────────

    def _arima_forecast(self, project_id, periods_df, cutoff_period):
        """
        Produce an ARIMA cost variance forecast for one project.

        Uses the pre-computed ARIMA order from arima_configs.json
        if available. Falls back to a default ARIMA(1,1,1) if not.

        Returns the predicted variance percentage or None if ARIMA
        cannot be fitted on this project's data.
        """
        if periods_df.empty:
            return None

        # Get training window up to cutoff
        train_window = periods_df[
            periods_df['period_number'] <= cutoff_period
        ]['actual_cost'].dropna()

        if len(train_window) < 4:
            return None

        # Get ARIMA order from config or use default
        config      = self.arima_configs.get(str(project_id), {})
        arima_order = tuple(config.get('order', [1, 1, 1]))
        bac         = None

        try:
            # Fit ARIMA
            model  = ARIMA(train_window.values, order=arima_order)
            fitted = model.fit()

            # Forecast remaining periods
            total_periods     = len(periods_df)
            periods_remaining = max(1, total_periods - cutoff_period)
            forecast_values   = np.array(
                fitted.get_forecast(
                    steps=periods_remaining
                ).predicted_mean
            )

            final_ac_forecast = float(forecast_values[-1])

            # Get BAC from database
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT budget_at_completion
                        FROM projects
                        WHERE project_id = :pid
                    """),
                    {'pid': project_id}
                )
                row = result.fetchone()
                if row and row[0]:
                    bac = float(row[0])

            if bac and bac > 0:
                arima_variance = ((final_ac_forecast - bac) / bac) * 100
                return round(float(arima_variance), 4)

        except Exception:
            return None

        return None


    # ── Save forecast to database ─────────────────────────────────────────────

    def _save_forecast(self, project_id, rf_variance,
                       arima_variance, cutoff_period,
                       shap_result, alert_card):
        """
        Save the complete forecast result to the database.
        Updates forecasts, shap_values and alerts tables.
        """
        with engine.connect() as conn:
            # Upsert forecast row
            existing = conn.execute(
                text("""
                    SELECT forecast_id FROM forecasts
                    WHERE project_id = :pid
                    ORDER BY generated_at DESC
                    LIMIT 1
                """),
                {'pid': project_id}
            ).fetchone()

            if existing:
                conn.execute(
                    text("""
                        UPDATE forecasts
                        SET rf_variance_pct    = :rf_var,
                            arima_variance_pct = :arima_var,
                            explanation        = :explanation,
                            generated_at       = NOW()
                        WHERE forecast_id = :fid
                    """),
                    {
                        'rf_var':      rf_variance,
                        'arima_var':   arima_variance,
                        'explanation': shap_result['explanation'],
                        'fid':         existing[0]
                    }
                )
                forecast_id = existing[0]
            else:
                result = conn.execute(
                    text("""
                        INSERT INTO forecasts (
                            project_id, forecast_period,
                            rf_variance_pct, arima_variance_pct,
                            explanation, generated_at
                        ) VALUES (
                            :pid, :period, :rf_var,
                            :arima_var, :explanation, NOW()
                        ) RETURNING forecast_id
                    """),
                    {
                        'pid':         project_id,
                        'period':      cutoff_period,
                        'rf_var':      rf_variance,
                        'arima_var':   arima_variance,
                        'explanation': shap_result['explanation']
                    }
                )
                forecast_id = result.fetchone()[0]

            # Save SHAP values
            conn.execute(
                text("DELETE FROM shap_values WHERE forecast_id = :fid"),
                {'fid': forecast_id}
            )
            for attr in shap_result['attributions']:
                conn.execute(
                    text("""
                        INSERT INTO shap_values (
                            forecast_id, feature_name,
                            shap_value, direction, base_value
                        ) VALUES (
                            :fid, :feature, :shap_val,
                            :direction, :base_val
                        )
                    """),
                    {
                        'fid':       forecast_id,
                        'feature':   attr['feature'],
                        'shap_val':  attr['shap_value'],
                        'direction': attr['direction'],
                        'base_val':  shap_result['base_value']
                    }
                )

            # Save alert
            conn.execute(
                text("DELETE FROM alerts WHERE project_id = :pid"),
                {'pid': project_id}
            )
            conn.execute(
                text("""
                    INSERT INTO alerts (
                        forecast_id, project_id, tier, status,
                        prescribed_action, responsible_role,
                        response_window, acknowledged, triggered_at
                    ) VALUES (
                        :fid, :pid, :tier, :status,
                        :action, :role, :window, FALSE, NOW()
                    )
                """),
                {
                    'fid':    forecast_id,
                    'pid':    project_id,
                    'tier':   alert_card['tier_number'],
                    'status': alert_card['tier_name'],
                    'action': alert_card['prescribed_action'],
                    'role':   alert_card['responsible_owner'],
                    'window': alert_card['response_window']
                }
            )

            conn.commit()

        return forecast_id


    # ── Main forecast function ────────────────────────────────────────────────

    def run_forecast(self, project_id, save=True, verbose=True):
        """
        Run the complete forecast pipeline for one project.

        This is the single function the dashboard calls.
        Everything else in the coordinator is private.

        Parameters:
            project_id : int  — the project to forecast
            save       : bool — whether to save results to database
            verbose    : bool — whether to print progress

        Returns a complete result dictionary containing:
            project      — project metadata
            rf_forecast  — Random Forest prediction
            arima_forecast — ARIMA prediction (or None)
            shap         — feature attributions and explanation
            alert        — governance tier and prescribed action
            summary      — plain-language summary for dashboard
        """
        if verbose:
            print(f"\nRunning forecast for project {project_id}...")

        # 1. Fetch project data
        project_row, periods_df = self._fetch_project(project_id)

        if project_row is None:
            return {
                'success': False,
                'error':   f"Project {project_id} not found or "
                           f"failed quality gate."
            }

        project_name   = project_row['project_name']
        bac            = float(project_row['bac']) \
                         if project_row['bac'] else None
        cutoff_period  = int(project_row['prediction_period'])
        actual_variance = project_row['actual_final_variance']

        if verbose:
            print(f"  Project     : {project_name}")
            print(f"  BAC         : {bac:,.2f}" if bac else
                  "  BAC         : N/A")
            print(f"  Cutoff      : period {cutoff_period}")

        # 2. Random Forest forecast
        rf_variance = self._rf_forecast(project_row)
        if verbose:
            print(f"  RF forecast : {rf_variance:.2f}%")

        # 3. SHAP explanation
        shap_result = self._shap_explain(project_row)
        if verbose:
            print(f"  SHAP        : {shap_result['explanation'][:60]}...")

        # 4. ARIMA forecast
        arima_variance = self._arima_forecast(
            project_id, periods_df, cutoff_period
        )
        if verbose:
            arima_str = (f"{arima_variance:.2f}%"
                         if arima_variance is not None else "N/A")
            print(f"  ARIMA       : {arima_str}")

        # 5. Governance layer
        alert_card = generate_alert_card(
            project_id     = project_id,
            project_name   = project_name,
            rf_variance    = rf_variance,
            arima_variance = arima_variance,
            explanation    = shap_result['explanation'],
            bac            = bac
        )
        if verbose:
            print(f"  Alert tier  : {alert_card['tier_name']} — "
                  f"{alert_card['response_window']}")

        # 6. Save to database
        forecast_id = None
        if save:
            forecast_id = self._save_forecast(
                project_id    = project_id,
                rf_variance   = rf_variance,
                arima_variance = arima_variance,
                cutoff_period  = cutoff_period,
                shap_result   = shap_result,
                alert_card    = alert_card
            )
            if verbose:
                print(f"  Saved       : forecast_id {forecast_id}")

        # 7. Build complete result package
        result = {
            'success':     True,
            'forecast_id': forecast_id,

            # Project info
            'project': {
                'id':               project_id,
                'name':             project_name,
                'bac':              bac,
                'cutoff_period':    cutoff_period,
                'actual_variance':  float(actual_variance)
                                    if actual_variance is not None
                                    else None,
                'train_test_flag':  project_row['train_test_flag'],
                'project_size':     project_row['project_size'],
            },

            # Model forecasts
            'rf_forecast': {
                'variance_pct':  rf_variance,
                'model_version': self.model_version,
            },

            'arima_forecast': {
                'variance_pct': arima_variance,
                'available':    arima_variance is not None,
            },

            # SHAP explanation
            'shap': {
                'attributions': shap_result['attributions'],
                'base_value':   shap_result['base_value'],
                'prediction':   shap_result['prediction'],
                'explanation':  shap_result['explanation'],
            },

            # Governance alert
            'alert': {
                'tier_number':       alert_card['tier_number'],
                'tier_name':         alert_card['tier_name'],
                'tier_colour':       alert_card['tier_colour'],
                'tier_label':        alert_card['tier_label'],
                'prescribed_action': alert_card['prescribed_action'],
                'responsible_owner': alert_card['responsible_owner'],
                'response_window':   alert_card['response_window'],
                'confidence':        alert_card['confidence'],
                'financial_impact':  alert_card['financial_impact'],
                'model_agreement':   alert_card['model_agreement'],
            },

            # Plain language summary for dashboard header
            'summary': (
                f"{alert_card['tier_name']} — "
                f"Predicted {rf_variance:+.1f}% cost variance. "
                f"{alert_card['response_window']}. "
                f"Owner: {alert_card['responsible_owner']}."
            ),

            'generated_at': str(datetime.now()),
        }

        return result


    # ── Batch forecast ────────────────────────────────────────────────────────

    def run_all_forecasts(self, flag=None, verbose=False):
        """
        Run forecasts for all projects in the database.

        Parameters:
            flag    : 'train', 'test', or None (all projects)
            verbose : whether to print per-project progress

        Returns a list of result dictionaries.
        Used by the portfolio overview panel in the dashboard.
        """
        where = ""
        if flag:
            where = f"AND pr.train_test_flag = '{flag}'"

        query = f"""
            SELECT f.project_id
            FROM features f
            JOIN projects pr ON f.project_id = pr.project_id
            WHERE pr.quality_flag = 'pass'
            {where}
            ORDER BY f.project_id
        """
        df = pd.read_sql(query, engine)

        print(f"\nRunning batch forecast for "
              f"{len(df)} projects "
              f"(flag={flag or 'all'})...\n")

        results  = []
        success  = 0
        failed   = 0

        for _, row in df.iterrows():
            pid = int(row['project_id'])
            try:
                result = self.run_forecast(
                    pid, save=True, verbose=verbose
                )
                results.append(result)
                if result['success']:
                    success += 1
                    tier = result['alert']['tier_name']
                    rf   = result['rf_forecast']['variance_pct']
                    name = result['project']['name'][:30]
                    print(f"  ✓ project {pid:<6} "
                          f"({name:<30}) — "
                          f"{tier:<10} RF: {rf:+.2f}%")
                else:
                    failed += 1
                    print(f"  ✗ project {pid} — "
                          f"{result.get('error', 'unknown error')}")
            except Exception as e:
                failed += 1
                print(f"  ✗ project {pid} — ERROR: {e}")

        print(f"\nBatch complete: {success} succeeded, {failed} failed")
        return results


    # ── Portfolio summary ─────────────────────────────────────────────────────

    def get_portfolio_summary(self, flag=None):
        """
        Return a summary DataFrame of all project alerts.
        Used by the portfolio overview panel in the dashboard.

        Returns columns: project_id, project_name, tier_name,
        tier_colour, rf_variance, arima_variance, response_window,
        responsible_owner, financial_impact
        """
        where = ""
        if flag:
            where = f"AND pr.train_test_flag = '{flag}'"

        query = f"""
            SELECT
                pr.project_id,
                pr.project_name,
                pr.budget_at_completion  AS bac,
                pr.train_test_flag,
                f.rf_variance_pct,
                f.arima_variance_pct,
                a.tier,
                a.status,
                a.prescribed_action,
                a.responsible_role,
                a.response_window,
                f.explanation
            FROM projects pr
            JOIN forecasts f ON f.project_id  = pr.project_id
            JOIN alerts    a ON a.project_id  = pr.project_id
            WHERE pr.quality_flag = 'pass'
            {where}
            ORDER BY a.tier DESC, ABS(f.rf_variance_pct) DESC
        """
        df = pd.read_sql(query, engine)

        # Add financial impact column
        def calc_impact(row):
            if row['bac'] and row['rf_variance_pct']:
                return round(
                    (row['rf_variance_pct'] / 100) * row['bac'], 2
                )
            return None

        df['financial_impact'] = df.apply(calc_impact, axis=1)

        # Add tier colour
        tier_colours = {
            'GREEN':    '#27AE60',
            'AMBER':    '#E67E22',
            'RED':      '#E74C3C',
            'CRITICAL': '#922B21'
        }
        df['tier_colour'] = df['status'].map(
            lambda x: tier_colours.get(x, '#AAAAAA')
        )

        return df


# ── Quick test function ───────────────────────────────────────────────────────

def test_coordinator():
    """
    Quick test — runs a forecast for the first available project
    and prints the complete result. Use this to verify the
    coordinator is wired together correctly before building
    the dashboard.
    """
    print("\nTesting ML Coordinator...\n")

    # Get first available project
    df = pd.read_sql("""
        SELECT f.project_id, pr.project_name
        FROM features f
        JOIN projects pr ON f.project_id = pr.project_id
        WHERE pr.quality_flag = 'pass'
        ORDER BY f.project_id
        LIMIT 1
    """, engine)

    if df.empty:
        print("ERROR: No projects found.")
        return

    project_id   = int(df.iloc[0]['project_id'])
    project_name = df.iloc[0]['project_name']
    print(f"Testing with: {project_name} (ID: {project_id})\n")

    # Initialise coordinator
    coordinator = MLCoordinator()

    # Run single forecast
    result = coordinator.run_forecast(project_id, verbose=True)

    if not result['success']:
        print(f"\nERROR: {result['error']}")
        return

    # Print full result
    print("\n" + "=" * 60)
    print("COMPLETE FORECAST RESULT")
    print("=" * 60)
    print(f"\n  Summary      : {result['summary']}")
    print(f"\n  RF Forecast  : {result['rf_forecast']['variance_pct']:.2f}%")
    arima = result['arima_forecast']['variance_pct']
    print(f"  ARIMA        : {arima:.2f}%" if arima else
          "  ARIMA        : N/A")
    print(f"\n  Alert Tier   : {result['alert']['tier_name']}")
    print(f"  Owner        : {result['alert']['responsible_owner']}")
    print(f"  Action       : {result['alert']['prescribed_action'][:60]}...")
    print(f"  Window       : {result['alert']['response_window']}")
    print(f"  Confidence   : {result['alert']['confidence']}")
    if result['alert']['financial_impact']:
        print(f"  Fin. Impact  : "
              f"{result['alert']['financial_impact']:+,.2f}")

    print(f"\n  SHAP explanation:")
    print(f"  {result['shap']['explanation']}")
    print(f"\n  Feature attributions:")
    for attr in result['shap']['attributions']:
        arrow = "▲" if attr['direction'] == 'up' else "▼"
        print(f"    {arrow} {attr['label']:<28} "
              f"{attr['shap_value']:+.4f}%  "
              f"(value: {attr['feature_value']:.4f})")

    print(f"\n  Base value   : {result['shap']['base_value']:.4f}%")
    print(f"  Prediction   : {result['shap']['prediction']:.4f}%")
    print("=" * 60)

    # Run portfolio summary
    print("\nTesting portfolio summary...")
    portfolio = coordinator.get_portfolio_summary()
    print(f"  Portfolio rows: {len(portfolio)}")
    print(f"  Columns: {list(portfolio.columns)}")
    print(f"\nCoordinator test complete.")
    print("Run dashboard/app.py next.")


if __name__ == "__main__":
    test_coordinator()