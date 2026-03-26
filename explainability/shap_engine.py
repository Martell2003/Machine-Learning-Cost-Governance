"""
explainability/shap_engine.py

Generates SHAP (SHapley Additive exPlanations) values for every
project forecast produced by the Random Forest model.

SHAP answers the question: why did the model predict this variance?
For each project it shows how much each of the four features pushed
the prediction up or down from the baseline average.

This makes the ML output explainable to a non-technical programme
manager — satisfying the Explainability design principle.

Run after train_rf.py:
    python explainability/shap_engine.py
"""

import shap
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine
from config import MODELS_SAVED


# ── Feature names — must match train_rf.py exactly ───────────────────────────

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


# ── STEP 1: Load the trained Random Forest model ──────────────────────────────

def load_model():
    """
    Load the best available Random Forest model from models/saved/.
    Tries tuned model first, falls back to baseline.
    """
    tuned_path    = os.path.join(MODELS_SAVED, 'random_forest_tuned.pkl')
    baseline_path = os.path.join(MODELS_SAVED, 'random_forest_baseline.pkl')

    if os.path.exists(tuned_path):
        model = joblib.load(tuned_path)
        print(f"  Loaded tuned Random Forest model")
        return model, 'tuned'
    elif os.path.exists(baseline_path):
        model = joblib.load(baseline_path)
        print(f"  Loaded baseline Random Forest model")
        return model, 'baseline'
    else:
        raise FileNotFoundError(
            "No trained model found in models/saved/. "
            "Run models/train_rf.py first."
        )


# ── STEP 2: Load features from PostgreSQL ────────────────────────────────────

def load_features():
    """
    Load all engineered features for every project.
    SHAP values are generated for both training and test projects
    so the dashboard can show explanations for any project.
    """
    query = """
        SELECT
            f.feature_id,
            f.project_id,
            pr.project_name,
            pr.train_test_flag,
            pr.budget_at_completion  AS bac,
            f.prediction_period,
            f.cpi_trend,
            f.sv_trajectory,
            f.burn_rate_ratio,
            f.schedule_pressure,
            f.actual_final_variance
        FROM features f
        JOIN projects pr ON f.project_id = pr.project_id
        WHERE pr.quality_flag = 'pass'
        ORDER BY f.project_id
    """
    df = pd.read_sql(query, engine)
    print(f"  Loaded {len(df)} projects for SHAP analysis")
    return df


# ── STEP 3: Load base value from model metadata ───────────────────────────────

def load_base_value(version):
    """
    Load the model's expected value (base value) from metadata.
    The base value is the average prediction across all training
    projects — the starting point before features push it up or down.
    """
    meta_path = os.path.join(
        MODELS_SAVED, f'random_forest_{version}_meta.json'
    )
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta
    return {}


# ── STEP 4: Run SHAP Tree Explainer ──────────────────────────────────────────

def run_shap(model, features_df):
    """
    Run the SHAP Tree Explainer on the Random Forest model.

    TreeExplainer is specifically designed for tree-based models
    like Random Forest. It calculates exact Shapley values by
    tracing the paths each data point takes through all 100 trees
    and measuring each feature's contribution at every decision node.

    Why SHAP over other explanation methods:
    SHAP values are mathematically consistent — they sum to the
    exact difference between the model prediction and the baseline.
    This makes them auditable and defensible in a governance context.
    (Lundberg and Lee, 2017)

    Returns:
        explainer   — the fitted SHAP explainer object
        shap_values — numpy array of shape (n_projects, n_features)
        base_value  — the model's average prediction (float)
    """
    X = features_df[FEATURE_COLS].values

    print(f"  Running SHAP Tree Explainer on {len(X)} projects...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    print(f"  Base value (model average prediction): {base_value:.4f}%")
    print(f"  SHAP values shape: {shap_values.shape}")

    return explainer, shap_values, base_value


# ── STEP 5: Interpret SHAP values ────────────────────────────────────────────

def interpret_shap(shap_values, features_df, base_value):
    """
    Convert raw SHAP values into human-readable interpretations.

    For each project and each feature:
        - The SHAP value shows how much that feature moved the
        prediction up (positive) or down (negative) from the
        base value
        - Direction is 'up' if positive, 'down' if negative
        - The sum of all SHAP values plus the base value equals
        the model's final prediction for that project

    Returns a list of dictionaries ready for database insertion.
    """
    records = []

    for i, (_, row) in enumerate(features_df.iterrows()):
        project_id = int(row['project_id'])
        prediction = float(base_value + shap_values[i].sum())

        for j, feature in enumerate(FEATURE_COLS):
            sv    = float(shap_values[i][j])
            fval  = float(row[feature]) if row[feature] is not None else 0.0

            records.append({
                'project_id':    project_id,
                'feature_name':  feature,
                'feature_label': FEATURE_LABELS[feature],
                'feature_value': round(fval, 6),
                'shap_value':    round(sv, 6),
                'direction':     'up' if sv > 0 else 'down',
                'abs_impact':    round(abs(sv), 6),
                'base_value':    round(base_value, 6),
                'prediction':    round(prediction, 6),
            })

    return records


# ── STEP 6: Generate plain-language explanation ───────────────────────────────

def generate_explanation(shap_records, project_id):
    """
    Generate a plain-language sentence explaining the forecast
    for a specific project.

    Example output:
    'The forecast of 14.2% cost overrun is primarily driven by
    poor Cost Performance (CPI) which adds 8.3 percentage points,
    compounded by high Budget Burn Rate adding 4.1 points.'

    This is what the dashboard displays below the SHAP waterfall
    chart — one sentence a non-technical manager can act on.
    """
    project_records = [r for r in shap_records
                    if r['project_id'] == project_id]

    if not project_records:
        return "No explanation available."

    # Sort by absolute impact — biggest driver first
    sorted_records = sorted(
        project_records,
        key=lambda x: x['abs_impact'],
        reverse=True
    )

    prediction  = sorted_records[0]['prediction']
    base        = sorted_records[0]['base_value']
    direction   = "overrun" if prediction > 0 else "underrun"
    top_drivers = sorted_records[:2]  # top two drivers

    # Build the sentence
    driver_parts = []
    for r in top_drivers:
        impact    = abs(r['shap_value'])
        label     = r['feature_label']
        direction_word = "adding" if r['direction'] == 'up' else "reducing by"
        driver_parts.append(
            f"{label} ({direction_word} {impact:.1f} percentage points)"
        )

    if len(driver_parts) == 1:
        drivers_text = driver_parts[0]
    else:
        drivers_text = f"{driver_parts[0]}, compounded by {driver_parts[1]}"

    explanation = (
        f"The forecast of {abs(prediction):.1f}% cost {direction} "
        f"is primarily driven by {drivers_text}. "
        f"The model baseline prediction is {base:.1f}%."
    )

    return explanation


# ── STEP 7: Save SHAP values to database ─────────────────────────────────────

def save_shap_values(shap_records, features_df, base_value):
    """
    Save SHAP values to the shap_values table.
    First links each project's SHAP values to the corresponding
    forecast_id in the forecasts table.
    Also saves the plain-language explanation for each project.
    """
    if not shap_records:
        print("  Nothing to save — SHAP records list is empty")
        return

    # Clear existing SHAP values to allow clean re-runs
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM shap_values"))
        conn.commit()

    saved = 0
    with engine.connect() as conn:
        for r in shap_records:
            # Find the forecast_id for this project
            result = conn.execute(
                text("""
                    SELECT forecast_id
                    FROM forecasts
                    WHERE project_id = :project_id
                    ORDER BY generated_at DESC
                    LIMIT 1
                """),
                {'project_id': r['project_id']}
            )
            row = result.fetchone()

            # If no forecast exists yet, create a placeholder
            if row is None:
                # Get the prediction for this project
                project_rows = [
                    x for x in shap_records
                    if x['project_id'] == r['project_id']
                ]
                prediction = project_rows[0]['prediction'] if project_rows else 0.0

                insert = conn.execute(
                    text("""
                        INSERT INTO forecasts
                            (project_id, forecast_period,
                            rf_variance_pct, generated_at)
                        VALUES
                            (:project_id, :period, :rf_var, NOW())
                        RETURNING forecast_id
                    """),
                    {
                        'project_id': r['project_id'],
                        'period':     int(
                            features_df[
                                features_df['project_id'] == r['project_id']
                            ]['prediction_period'].iloc[0]
                        ),
                        'rf_var': prediction
                    }
                )
                forecast_id = insert.fetchone()[0]
            else:
                forecast_id = row[0]

            # Insert the SHAP value row
            conn.execute(
                text("""
                    INSERT INTO shap_values
                        (forecast_id, feature_name, shap_value,
                        direction, base_value)
                    VALUES
                        (:forecast_id, :feature_name, :shap_value,
                        :direction, :base_value)
                """),
                {
                    'forecast_id':  forecast_id,
                    'feature_name': r['feature_name'],
                    'shap_value':   r['shap_value'],
                    'direction':    r['direction'],
                    'base_value':   r['base_value']
                }
            )
            saved += 1

        conn.commit()

    print(f"  Saved {saved} SHAP value rows to database")


# ── STEP 8: Save explanations to database ────────────────────────────────────

def save_explanations(shap_records, features_df):
    """
    Add a plain-language explanation column to the forecasts table
    and populate it for every project.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE forecasts
            ADD COLUMN IF NOT EXISTS explanation TEXT
        """))
        conn.commit()

    project_ids = features_df['project_id'].unique()
    saved = 0

    with engine.connect() as conn:
        for project_id in project_ids:
            explanation = generate_explanation(shap_records, int(project_id))
            conn.execute(
                text("""
                    UPDATE forecasts
                    SET explanation = :explanation
                    WHERE project_id = :project_id
                """),
                {
                    'explanation': explanation,
                    'project_id':  int(project_id)
                }
            )
            saved += 1
        conn.commit()

    print(f"  Saved {saved} plain-language explanations to forecasts table")


# ── STEP 9: Summary report ────────────────────────────────────────────────────

def print_summary(shap_records, features_df, base_value):
    """
    Print a summary of SHAP results including average feature
    importance across all projects and per-project top drivers.
    """
    print("\n" + "=" * 65)
    print("SHAP EXPLAINABILITY SUMMARY")
    print("=" * 65)

    print(f"\n  Base value (average prediction) : {base_value:.4f}%")
    print(f"  Projects explained              : "
        f"{len(features_df)}")
    print(f"  SHAP rows generated             : {len(shap_records)}")

    # Average absolute SHAP value per feature
    # This is the global feature importance from SHAP's perspective
    print(f"\n  Global feature importance (mean absolute SHAP value):")
    print(f"  {'Feature':<30} {'Mean |SHAP|':<14} {'Direction tendency'}")
    print(f"  {'-'*30} {'-'*14} {'-'*20}")

    for feature in FEATURE_COLS:
        feature_records = [
            r for r in shap_records if r['feature_name'] == feature
        ]
        if not feature_records:
            continue

        values     = [r['shap_value'] for r in feature_records]
        mean_abs   = round(np.mean([abs(v) for v in values]), 4)
        mean_val   = np.mean(values)
        tendency   = "tends upward" if mean_val > 0 else "tends downward"
        label      = FEATURE_LABELS[feature]
        bar        = '█' * int(mean_abs * 10)

        print(f"  {label:<30} {mean_abs:<14.4f} {tendency}  {bar}")

    # Sample explanations for test projects
    test_projects = features_df[
        features_df['train_test_flag'] == 'test'
    ]['project_id'].tolist()

    if test_projects:
        print(f"\n  Sample plain-language explanations (test projects):")
        for pid in test_projects[:3]:
            name = features_df[
                features_df['project_id'] == pid
            ]['project_name'].iloc[0]
            actual = features_df[
                features_df['project_id'] == pid
            ]['actual_final_variance'].iloc[0]
            explanation = generate_explanation(shap_records, int(pid))
            print(f"\n  Project: {name}")
            print(f"  Actual variance: {actual:.2f}%")
            print(f"  Explanation: {explanation}")

    # Verify SHAP consistency
    print(f"\n  Consistency check (SHAP values sum to prediction):")
    sample_pid    = int(features_df.iloc[0]['project_id'])
    sample_actual = float(
        features_df.iloc[0]['actual_final_variance']
    )
    sample_shaps  = [
        r for r in shap_records if r['project_id'] == sample_pid
    ]
    if sample_shaps:
        shap_sum    = sum(r['shap_value'] for r in sample_shaps)
        prediction  = base_value + shap_sum
        print(f"  Project {sample_pid}:")
        print(f"    Base value         : {base_value:.4f}%")
        print(f"    Sum of SHAP values : {shap_sum:.4f}%")
        print(f"    Final prediction   : {prediction:.4f}%")
        print(f"    Actual variance    : {sample_actual:.4f}%")
        for r in sorted(sample_shaps,
                        key=lambda x: x['abs_impact'],
                        reverse=True):
            arrow = "▲" if r['direction'] == 'up' else "▼"
            print(f"    {arrow} {r['feature_label']:<28} "
                f"{r['shap_value']:+.4f}%")

    print("=" * 65)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_shap_engine():
    print("\nStarting SHAP explainability engine...\n")

    # Load model
    print("Loading model:")
    model, version = load_model()

    # Load features
    print("\nLoading features:")
    features_df = load_features()

    if features_df.empty:
        print("ERROR: No features found.")
        print("Run features/engineer.py first.")
        return

    # Drop rows with null features
    before = len(features_df)
    features_df = features_df.dropna(subset=FEATURE_COLS)
    after = len(features_df)
    if before != after:
        print(f"  Dropped {before - after} rows with null features")

    # Run SHAP
    print("\nRunning SHAP analysis:")
    explainer, shap_values, base_value = run_shap(model, features_df)

    # Interpret values
    print("\nInterpreting SHAP values:")
    shap_records = interpret_shap(shap_values, features_df, base_value)
    print(f"  Generated {len(shap_records)} SHAP records "
        f"({len(features_df)} projects × {len(FEATURE_COLS)} features)")

    # Save to database
    print("\nSaving to database:")
    save_shap_values(shap_records, features_df, base_value)
    save_explanations(shap_records, features_df)

    # Summary
    print_summary(shap_records, features_df, base_value)

    print("\nSHAP engine complete. Run explainability/shap_plots.py next.")


if __name__ == "__main__":
    run_shap_engine()