"""
explainability/shap_plots.py

Generates SHAP waterfall charts for individual project forecasts.
These charts are embedded in the Streamlit dashboard to show a
programme manager exactly why the model predicted a particular
cost variance — which features pushed it up and which pushed it down.

Can be run standalone to generate and save PNG files, or imported
by the dashboard to render charts directly in the browser.

Run after shap_engine.py:
    python explainability/shap_plots.py
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import os
import sys
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine
from config import MODELS_SAVED


# ── Constants ─────────────────────────────────────────────────────────────────

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

# Colour scheme — red means pushes cost up, green means pushes cost down
COLOUR_UP   = '#E74C3C'   # red   — feature increases predicted overrun
COLOUR_DOWN = '#27AE60'   # green — feature decreases predicted overrun
COLOUR_BASE = '#2E74B5'   # blue  — baseline / neutral
COLOUR_PRED = '#1F4E79'   # dark  — final prediction bar

OUTPUT_DIR = 'outputs/shap_plots'


# ── STEP 1: Load model and features ──────────────────────────────────────────

def load_model():
    """Load the best available trained Random Forest model."""
    tuned_path    = os.path.join(MODELS_SAVED, 'random_forest_tuned.pkl')
    baseline_path = os.path.join(MODELS_SAVED, 'random_forest_baseline.pkl')

    if os.path.exists(tuned_path):
        return joblib.load(tuned_path), 'tuned'
    elif os.path.exists(baseline_path):
        return joblib.load(baseline_path), 'baseline'
    else:
        raise FileNotFoundError(
            "No trained model found. Run models/train_rf.py first."
        )


def load_project_features(project_id=None):
    """
    Load features for one project or all projects.
    If project_id is None, loads all projects.
    """
    if project_id:
        where = f"AND f.project_id = {int(project_id)}"
    else:
        where = ""

    query = f"""
        SELECT
            f.project_id,
            pr.project_name,
            pr.train_test_flag,
            pr.budget_at_completion  AS bac,
            f.prediction_period,
            f.cpi_trend,
            f.sv_trajectory,
            f.burn_rate_ratio,
            f.schedule_pressure,
            f.actual_final_variance,
            fc.rf_variance_pct,
            fc.explanation
        FROM features f
        JOIN projects pr ON f.project_id  = pr.project_id
        LEFT JOIN forecasts fc ON fc.project_id = f.project_id
        WHERE pr.quality_flag = 'pass'
        {where}
        ORDER BY f.project_id
    """
    return pd.read_sql(query, engine)


def load_shap_values_from_db(project_id):
    """
    Load pre-computed SHAP values for one project from the database.
    Used by the dashboard to avoid recomputing SHAP on every page load.
    """
    query = """
        SELECT
            sv.feature_name,
            sv.shap_value,
            sv.direction,
            sv.base_value
        FROM shap_values sv
        JOIN forecasts f ON sv.forecast_id = f.forecast_id
        WHERE f.project_id = :project_id
        ORDER BY ABS(sv.shap_value) DESC
    """
    return pd.read_sql(query, engine, params={'project_id': project_id})


# ── STEP 2: Compute SHAP values for one project ───────────────────────────────

def compute_shap_for_project(model, project_row):
    """
    Compute SHAP values for a single project on the fly.
    Used when pre-computed values are not available in the database.

    Returns:
        shap_vals  — array of 4 SHAP values (one per feature)
        base_value — model average prediction
        prediction — final model prediction for this project
    """
    X = np.array([[
        project_row['cpi_trend'],
        project_row['sv_trajectory'],
        project_row['burn_rate_ratio'],
        project_row['schedule_pressure']
    ]])

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)[0]
    base_value = float(np.atleast_1d(explainer.expected_value)[0])
    prediction = float(base_value + shap_vals.sum())

    return shap_vals, base_value, prediction


# ── STEP 3: Waterfall chart ───────────────────────────────────────────────────

def waterfall_chart(
    shap_vals, feature_values, base_value,
    prediction, project_name, actual_variance=None,
    save_path=None, show=False
):
    """
    Draw a SHAP waterfall chart for one project.

    A waterfall chart shows:
        - The starting point (base value = model average)
        - Each feature's contribution as a bar pushing up or down
        - The final prediction at the end

    Reading the chart:
        Red bars  — this feature is pushing the predicted cost UP
        Green bars — this feature is pushing the predicted cost DOWN
        The wider the bar, the more impact this feature had

    This is what the non-technical programme manager reads to
    understand WHY the model made its prediction.
    """
    # Sort features by absolute SHAP value — biggest impact at top
    order     = np.argsort(np.abs(shap_vals))[::-1]
    sorted_sv = shap_vals[order]
    sorted_fv = [feature_values[i] for i in order]
    sorted_fl = [FEATURE_LABELS[FEATURE_COLS[i]] for i in order]

    n = len(sorted_sv)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Calculate waterfall positions
    # Each bar starts where the previous one ended
    running = base_value
    starts  = []
    for sv in sorted_sv:
        starts.append(running)
        running += sv

    # Draw each feature bar
    bar_height = 0.5
    y_positions = list(range(n))

    for i in range(n):
        sv    = sorted_sv[i]
        start = starts[i]
        color = COLOUR_UP if sv > 0 else COLOUR_DOWN
        ax.barh(
            y_positions[i],
            sv,
            left=start,
            height=bar_height,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

        # Value label inside or outside bar
        label_x = start + sv + (0.3 if sv >= 0 else -0.3)
        label   = f"{sv:+.2f}%"
        ha      = 'left' if sv >= 0 else 'right'
        ax.text(
            label_x, y_positions[i], label,
            va='center', ha=ha,
            fontsize=9, fontweight='bold',
            color=COLOUR_UP if sv > 0 else COLOUR_DOWN
        )

        # Feature name and value on left
        fval_str = f"{sorted_fv[i]:.4f}"
        ax.text(
            ax.get_xlim()[0] - 0.5,
            y_positions[i],
            f"{sorted_fl[i]} = {fval_str}",
            va='center', ha='right',
            fontsize=9, color='#333333'
        )

    # Base value line
    ax.axvline(
        x=base_value, color=COLOUR_BASE,
        linestyle='--', linewidth=1.5, alpha=0.7,
        label=f'Base value: {base_value:.2f}%'
    )

    # Final prediction line
    ax.axvline(
        x=prediction, color=COLOUR_PRED,
        linestyle='-', linewidth=2, alpha=0.9,
        label=f'Prediction: {prediction:.2f}%'
    )

    # Actual variance line if available
    if actual_variance is not None:
        ax.axvline(
            x=actual_variance, color='#8E44AD',
            linestyle=':', linewidth=2, alpha=0.8,
            label=f'Actual: {actual_variance:.2f}%'
        )

    # Y axis labels — feature names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_fl, fontsize=9)

    # Styling
    ax.set_xlabel('Contribution to predicted cost variance (%)',
                fontsize=10)
    ax.set_title(
        f'SHAP Waterfall Chart — {project_name}\n'
        f'Predicted variance: {prediction:.2f}%'
        + (f'  |  Actual: {actual_variance:.2f}%'
        if actual_variance is not None else ''),
        fontsize=11, fontweight='bold', color='#1F4E79',
        pad=12
    )

    # Zero line
    ax.axvline(x=0, color='#AAAAAA', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOUR_UP,   label='Increases overrun'),
        mpatches.Patch(color=COLOUR_DOWN, label='Decreases overrun'),
        plt.Line2D([0], [0], color=COLOUR_BASE,
                linestyle='--', label=f'Base: {base_value:.2f}%'),
        plt.Line2D([0], [0], color=COLOUR_PRED,
                linestyle='-',  label=f'Predicted: {prediction:.2f}%'),
    ]
    if actual_variance is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color='#8E44AD',
                    linestyle=':', label=f'Actual: {actual_variance:.2f}%')
        )
    ax.legend(
        handles=legend_elements,
        loc='lower right', fontsize=8,
        framealpha=0.9
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#F8FAFC')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# ── STEP 4: Summary importance bar chart ─────────────────────────────────────

def global_importance_chart(
    all_shap_vals, save_path=None, show=False
):
    """
    Draw a global feature importance chart showing the average
    absolute SHAP value for each feature across all projects.

    This answers: which feature matters most overall?

    This chart goes in the dissertation and the dashboard overview.
    """
    mean_abs = np.mean(np.abs(all_shap_vals), axis=0)
    order    = np.argsort(mean_abs)
    labels   = [FEATURE_LABELS[FEATURE_COLS[i]] for i in order]
    values   = mean_abs[order]
    colors   = [COLOUR_DOWN if v > 0 else COLOUR_UP for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    bars = ax.barh(
        labels, values,
        color=COLOUR_BASE, alpha=0.8,
        edgecolor='white', linewidth=0.5
    )

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}',
            va='center', ha='left',
            fontsize=9, color='#333333'
        )

    ax.set_xlabel('Mean absolute SHAP value (percentage points)',
                fontsize=10)
    ax.set_title(
        'Global Feature Importance — Mean |SHAP| Across All Projects\n'
        'Higher value = stronger influence on cost variance predictions',
        fontsize=11, fontweight='bold', color='#1F4E79', pad=10
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#F8FAFC')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# ── STEP 5: Dashboard-ready chart function ────────────────────────────────────

def get_waterfall_figure(project_id, model=None):
    """
    Entry point for the Streamlit dashboard.
    Returns a matplotlib figure ready to be rendered with
    st.pyplot(fig) — no file saving needed.

    Usage in dashboard:
        from explainability.shap_plots import get_waterfall_figure
        fig = get_waterfall_figure(project_id=42, model=rf_model)
        st.pyplot(fig)
    """
    # Load project data
    df = load_project_features(project_id)
    if df.empty:
        return None

    row          = df.iloc[0]
    project_name = row['project_name']
    actual       = row['actual_final_variance']

    # Try loading pre-computed SHAP values from database first
    shap_db = load_shap_values_from_db(project_id)

    if not shap_db.empty and len(shap_db) == len(FEATURE_COLS):
        # Use pre-computed values from database
        base_value = float(shap_db['base_value'].iloc[0])
        shap_vals  = np.zeros(len(FEATURE_COLS))
        feat_vals  = []

        for i, feat in enumerate(FEATURE_COLS):
            feat_row = shap_db[shap_db['feature_name'] == feat]
            if not feat_row.empty:
                shap_vals[i] = float(feat_row['shap_value'].iloc[0])
            feat_vals.append(float(row[feat]))

        prediction = base_value + shap_vals.sum()

    elif model is not None:
        # Compute on the fly if pre-computed not available
        shap_vals, base_value, prediction = compute_shap_for_project(
            model, row
        )
        feat_vals = [float(row[f]) for f in FEATURE_COLS]

    else:
        return None

    fig = waterfall_chart(
        shap_vals    = shap_vals,
        feature_values = feat_vals,
        base_value   = base_value,
        prediction   = prediction,
        project_name = project_name,
        actual_variance = actual
    )

    plt.close()
    return fig


def get_importance_figure(model=None):
    """
    Entry point for the Streamlit dashboard — global importance chart.
    Returns a matplotlib figure ready for st.pyplot(fig).

    Usage in dashboard:
        from explainability.shap_plots import get_importance_figure
        fig = get_importance_figure(model=rf_model)
        st.pyplot(fig)
    """
    df = load_project_features()
    if df.empty or model is None:
        return None

    df    = df.dropna(subset=FEATURE_COLS)
    X     = df[FEATURE_COLS].values
    exp   = shap.TreeExplainer(model)
    sv    = exp.shap_values(X)

    fig = global_importance_chart(sv)
    plt.close()
    return fig


# ── MAIN: Generate and save all charts ────────────────────────────────────────

def run_shap_plots():
    print("\nGenerating SHAP waterfall charts...\n")

    # Load model
    model, version = load_model()
    print(f"  Model loaded: {version}\n")

    # Load all project features
    df = load_project_features()
    if df.empty:
        print("ERROR: No features found. Run features/engineer.py first.")
        return

    df = df.dropna(subset=FEATURE_COLS)

    # Compute SHAP for all projects
    X          = df[FEATURE_COLS].values
    explainer  = shap.TreeExplainer(model)
    all_shap   = explainer.shap_values(X)
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    print(f"  Generating charts for {len(df)} projects...\n")

    # Generate individual waterfall chart for each project
    for i, (_, row) in enumerate(df.iterrows()):
        project_id   = int(row['project_id'])
        project_name = row['project_name']
        actual       = row['actual_final_variance']
        shap_vals    = all_shap[i]
        feat_vals    = [float(row[f]) for f in FEATURE_COLS]
        prediction   = float(base_value + shap_vals.sum())

        save_path = os.path.join(
            OUTPUT_DIR,
            f'waterfall_project_{project_id}.png'
        )

        waterfall_chart(
            shap_vals      = shap_vals,
            feature_values = feat_vals,
            base_value     = base_value,
            prediction     = prediction,
            project_name   = project_name,
            actual_variance = actual,
            save_path      = save_path
        )

        print(f"  [{row['train_test_flag']}] project {project_id} "
            f"({project_name[:30]}) — "
            f"predicted: {prediction:.2f}% | "
            f"actual: {actual:.2f}%")

    # Generate global importance chart
    print(f"\n  Generating global importance chart...")
    importance_path = os.path.join(OUTPUT_DIR, 'global_importance.png')
    global_importance_chart(
        all_shap_vals = all_shap,
        save_path     = importance_path
    )

    # Summary
    print(f"\n" + "=" * 60)
    print(f"SHAP PLOTS SUMMARY")
    print(f"=" * 60)
    print(f"  Charts generated : {len(df)} waterfall + 1 importance")
    print(f"  Saved to         : {OUTPUT_DIR}/")
    print(f"  Base value       : {base_value:.4f}%")
    print(f"\n  Global feature importance:")
    mean_abs = np.mean(np.abs(all_shap), axis=0)
    for feat, imp in sorted(
        zip(FEATURE_COLS, mean_abs),
        key=lambda x: x[1], reverse=True
    ):
        bar = '█' * int(imp * 15)
        print(f"    {FEATURE_LABELS[feat]:<30} {imp:.4f}  {bar}")
    print(f"=" * 60)
    print(f"\nSHAP plots complete. Run governance/decision_matrix.py next.")


if __name__ == "__main__":
    run_shap_plots()