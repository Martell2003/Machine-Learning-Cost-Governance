"""
dashboard/app.py

The Streamlit dashboard for the ML-Enhanced Cost Governance Framework.
This is the user-facing interface — what the programme manager,
PMO Lead and expert validator interact with directly.

Four panels:
    1. Alert Panel        — governance tier, prescribed action, owner
    2. Forecast Panel     — RF and ARIMA predictions side by side
    3. SHAP Panel         — waterfall chart explaining the forecast
    4. Portfolio Panel    — all projects overview with tier status

No ML knowledge required to use this dashboard.
Every output is connected to a specific management action.

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine
from coordinator.ml_coordinator import MLCoordinator


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "ML Cost Governance Framework",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8FAFC; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }

    /* Alert banner */
    .alert-green   { background:#EAFAEA; border-left:6px solid #27AE60;
                     padding:16px; border-radius:6px; margin:8px 0; }
    .alert-amber   { background:#FFF8E1; border-left:6px solid #E67E22;
                     padding:16px; border-radius:6px; margin:8px 0; }
    .alert-red     { background:#FDEDEC; border-left:6px solid #E74C3C;
                     padding:16px; border-radius:6px; margin:8px 0; }
    .alert-critical{ background:#F9EBEA; border-left:6px solid #922B21;
                     padding:16px; border-radius:6px; margin:8px 0; }

    /* Section headers */
    .section-header {
        color: #1F4E79;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 2px solid #D0E8F5;
    }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models...")
def get_coordinator():
    """
    Initialise the ML Coordinator once and cache it.
    This prevents reloading the Random Forest and SHAP explainer
    on every user interaction — keeps the dashboard fast.
    """
    return MLCoordinator()


@st.cache_data(ttl=300, show_spinner=False)
def get_all_projects():
    """
    Load the list of all available projects from the database.
    Cached for 5 minutes — refreshes automatically.
    """
    query = """
        SELECT
            pr.project_id,
            pr.project_name,
            pr.train_test_flag,
            pr.project_size,
            pr.budget_at_completion
        FROM projects pr
        JOIN features f ON f.project_id = pr.project_id
        WHERE pr.quality_flag = 'pass'
        ORDER BY pr.project_name
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=60, show_spinner=False)
def get_portfolio(_coordinator):
    """
    Load portfolio summary for all projects.
    Cached for 60 seconds.
    """
    return _coordinator.get_portfolio_summary()


# ── Helper: tier colour mapping ───────────────────────────────────────────────

TIER_COLOURS = {
    'GREEN':    '#27AE60',
    'AMBER':    '#E67E22',
    'RED':      '#E74C3C',
    'CRITICAL': '#922B21'
}

TIER_CSS = {
    'GREEN':    'alert-green',
    'AMBER':    'alert-amber',
    'RED':      'alert-red',
    'CRITICAL': 'alert-critical'
}


# ── Helper: format currency ───────────────────────────────────────────────────

def fmt_currency(value):
    if value is None:
        return 'N/A'
    sign = '+' if value > 0 else ''
    if abs(value) >= 1_000_000:
        return f"{sign}{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{sign}{value/1_000:.1f}K"
    else:
        return f"{sign}{value:.2f}"


# ── Panel 1: Alert panel ──────────────────────────────────────────────────────

def render_alert_panel(result):
    """
    Render the governance alert card.
    This is the most prominent element on the screen — the first
    thing a manager sees when they open a project forecast.
    """
    alert    = result['alert']
    tier     = alert['tier_name']
    colour   = TIER_COLOURS.get(tier, '#AAAAAA')
    css_cls  = TIER_CSS.get(tier, 'alert-green')

    st.markdown(
        f'<div class="section-header">🚨 Governance Alert</div>',
        unsafe_allow_html=True
    )

    # Main alert banner
    st.markdown(f"""
    <div class="{css_cls}">
        <h2 style="color:{colour}; margin:0 0 8px 0;">
            {tier} — {alert['tier_label']}
        </h2>
        <p style="font-size:1.05rem; margin:0 0 8px 0;">
            <strong>Predicted variance:</strong>
            {result['rf_forecast']['variance_pct']:+.2f}%
        </p>
        <p style="margin:0 0 8px 0;">
            <strong>Prescribed action:</strong><br>
            {alert['prescribed_action']}
        </p>
        <p style="margin:0 0 4px 0;">
            <strong>Responsible:</strong> {alert['responsible_owner']}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Response window:</strong> {alert['response_window']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence and financial impact row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label = "Model Confidence",
            value = alert['confidence'].split('—')[0].strip()
                    if '—' in alert['confidence']
                    else alert['confidence'].split(' ')[0]
        )
        st.caption(alert['confidence'])

    with col2:
        impact = alert['financial_impact']
        if impact is not None:
            label = "Est. Cost Overrun" if impact > 0 \
                    else "Est. Cost Saving"
            st.metric(
                label = label,
                value = fmt_currency(impact),
                delta = fmt_currency(impact),
                delta_color = "inverse"
            )
        else:
            st.metric(label="Financial Impact", value="N/A")

    with col3:
        agreement = alert.get('model_agreement')
        if agreement is True:
            st.metric(label="RF vs ARIMA", value="✓ Agree")
            st.caption("Both models predict same tier")
        elif agreement is False:
            st.metric(label="RF vs ARIMA", value="⚠ Diverge")
            arima = result['arima_forecast']['variance_pct']
            rf    = result['rf_forecast']['variance_pct']
            st.caption(
                f"RF: {rf:+.1f}%  ARIMA: {arima:+.1f}%  "
                f"— investigate divergence"
            )
        else:
            st.metric(label="RF vs ARIMA", value="RF only")
            st.caption("ARIMA not available for this project")

    # Lead time note
    st.info(
        "⏱ This alert is generated at the **20% project completion "
        "mark** — providing a minimum 4-week lead time for management "
        "intervention before further budget is committed.",
        icon=None
    )


# ── Panel 2: Forecast panel ───────────────────────────────────────────────────

def render_forecast_panel(result):
    """
    Render the RF and ARIMA forecast side by side with a
    simple bar chart comparison.
    """
    st.markdown(
        '<div class="section-header">📈 Cost Variance Forecasts</div>',
        unsafe_allow_html=True
    )

    rf_var    = result['rf_forecast']['variance_pct']
    arima_var = result['arima_forecast']['variance_pct']
    actual    = result['project']['actual_variance']

    col1, col2, col3 = st.columns(3)

    with col1:
        delta_color = "inverse" if rf_var > 0 else "normal"
        st.metric(
            label       = "Random Forest Forecast",
            value       = f"{rf_var:+.2f}%",
            delta       = f"{rf_var:+.2f}% vs BAC",
            delta_color = delta_color,
            help        = "Cross-project pattern model. "
                          "Trained on 40+ completed IT projects."
        )

    with col2:
        if arima_var is not None:
            delta_color = "inverse" if arima_var > 0 else "normal"
            st.metric(
                label       = "ARIMA Forecast",
                value       = f"{arima_var:+.2f}%",
                delta       = f"{arima_var:+.2f}% vs BAC",
                delta_color = delta_color,
                help        = "Time-series trajectory model. "
                              "Forecasts based on this project's "
                              "own cost trajectory."
            )
        else:
            st.metric(
                label = "ARIMA Forecast",
                value = "N/A",
                help  = "Insufficient period data for ARIMA."
            )

    with col3:
        if actual is not None:
            delta_color = "inverse" if actual > 0 else "normal"
            st.metric(
                label       = "Actual Outcome",
                value       = f"{actual:+.2f}%",
                delta       = f"{actual:+.2f}% vs BAC",
                delta_color = delta_color,
                help        = "Known final outcome from DSLIB dataset. "
                              "Used for evaluation only."
            )
        else:
            st.metric(
                label = "Actual Outcome",
                value = "Unknown",
                help  = "Final outcome not yet available."
            )

    # Forecast comparison bar chart
    st.markdown("####")
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    labels = ['Random Forest']
    values = [rf_var]
    colors = ['#E74C3C' if rf_var > 0 else '#27AE60']

    if arima_var is not None:
        labels.append('ARIMA')
        values.append(arima_var)
        colors.append('#E74C3C' if arima_var > 0 else '#27AE60')

    if actual is not None:
        labels.append('Actual')
        values.append(actual)
        colors.append('#8E44AD')

    bars = ax.barh(labels, values, color=colors, alpha=0.8,
                   edgecolor='white', height=0.5)

    # Value labels
    for bar, val in zip(bars, values):
        x = bar.get_width()
        ax.text(
            x + (0.2 if x >= 0 else -0.2),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}%",
            va='center',
            ha='left' if x >= 0 else 'right',
            fontsize=10, fontweight='bold'
        )

    ax.axvline(x=0, color='#AAAAAA', linewidth=1)
    ax.set_xlabel('Predicted cost variance (%)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ── Panel 3: SHAP waterfall panel ─────────────────────────────────────────────

def render_shap_panel(result):
    """
    Render the SHAP waterfall chart and plain-language explanation.
    Shows which features drove the prediction and by how much.
    """
    st.markdown(
        '<div class="section-header">'
        '🔍 Why This Forecast Was Made'
        '</div>',
        unsafe_allow_html=True
    )

    shap_data  = result['shap']
    attrs      = shap_data['attributions']
    base_value = shap_data['base_value']
    prediction = shap_data['prediction']

    # Plain language explanation box
    st.success(f"💬 {shap_data['explanation']}")

    # Waterfall chart
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Sort by absolute impact
    sorted_attrs = sorted(attrs, key=lambda x: x['abs_impact'],
                          reverse=True)
    n = len(sorted_attrs)

    running     = base_value
    starts      = []
    shap_values = []
    labels      = []
    feat_vals   = []

    for attr in sorted_attrs:
        starts.append(running)
        shap_values.append(attr['shap_value'])
        labels.append(attr['label'])
        feat_vals.append(attr['feature_value'])
        running += attr['shap_value']

    y_pos  = list(range(n))
    colors = ['#E74C3C' if sv > 0 else '#27AE60'
              for sv in shap_values]

    bars = ax.barh(
        y_pos, shap_values, left=starts,
        color=colors, alpha=0.85,
        edgecolor='white', linewidth=0.5,
        height=0.5
    )

    # Value labels on bars
    for i, (sv, start) in enumerate(zip(shap_values, starts)):
        x    = start + sv
        ha   = 'left' if sv >= 0 else 'right'
        xoff = 0.15 if sv >= 0 else -0.15
        ax.text(
            x + xoff, y_pos[i],
            f"{sv:+.2f}%",
            va='center', ha=ha,
            fontsize=9, fontweight='bold',
            color='#E74C3C' if sv > 0 else '#27AE60'
        )

    # Feature labels on left with value
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{l}  =  {v:.4f}" for l, v in zip(labels, feat_vals)],
        fontsize=9
    )

    # Reference lines
    ax.axvline(x=base_value, color='#2E74B5', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Base: {base_value:.2f}%')
    ax.axvline(x=prediction, color='#1F4E79', linestyle='-',
               linewidth=2,
               label=f'Prediction: {prediction:.2f}%')
    ax.axvline(x=0, color='#AAAAAA', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='Increases cost overrun'),
        mpatches.Patch(color='#27AE60', label='Decreases cost overrun'),
        plt.Line2D([0], [0], color='#2E74B5', linestyle='--',
                   label=f'Base value: {base_value:.2f}%'),
        plt.Line2D([0], [0], color='#1F4E79', linestyle='-',
                   label=f'Prediction: {prediction:.2f}%'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=8, framealpha=0.9)

    ax.set_xlabel('Contribution to predicted variance (%)', fontsize=9)
    ax.set_title(
        'SHAP Feature Attribution — How each factor influenced '
        'this forecast',
        fontsize=10, fontweight='bold', color='#1F4E79'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Feature breakdown table
    with st.expander("📋 Full feature breakdown"):
        rows = []
        for attr in sorted_attrs:
            arrow = "▲ Up" if attr['direction'] == 'up' else "▼ Down"
            rows.append({
                'Feature':       attr['label'],
                'Value':         f"{attr['feature_value']:.4f}",
                'SHAP Impact':   f"{attr['shap_value']:+.4f}%",
                'Direction':     arrow,
                'Abs. Impact':   f"{attr['abs_impact']:.4f}%"
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        f"Base value: {base_value:.4f}%  |  "
        f"Sum of SHAP: {sum(a['shap_value'] for a in attrs):+.4f}%  |  "
        f"Final prediction: {prediction:.4f}%  |  "
        f"SHAP values sum to the exact difference between prediction "
        f"and base value (Lundberg and Lee, 2017)."
    )


# ── Panel 4: Portfolio overview panel ────────────────────────────────────────

def render_portfolio_panel(coordinator):
    """
    Render the portfolio overview showing all projects and their
    current alert tier. Used by the PMO Lead to identify
    Red and Critical projects without opening each one individually.
    """
    st.markdown(
        '<div class="section-header">📁 Portfolio Overview</div>',
        unsafe_allow_html=True
    )

    portfolio = get_portfolio(coordinator)

    if portfolio.empty:
        st.info("No forecasts available. Run the coordinator first.")
        return

    # Tier filter
    col1, col2 = st.columns([3, 1])
    with col2:
        tier_filter = st.selectbox(
            "Filter by tier",
            options=["All", "CRITICAL", "RED", "AMBER", "GREEN"],
            index=0
        )

    filtered = portfolio if tier_filter == "All" \
               else portfolio[portfolio['status'] == tier_filter]

    with col1:
        st.caption(
            f"Showing {len(filtered)} of {len(portfolio)} projects"
        )

    # Summary metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    counts = portfolio['status'].value_counts()

    with m1:
        st.metric("Total Projects", len(portfolio))
    with m2:
        n = counts.get('CRITICAL', 0)
        st.metric("🔴 Critical", n,
                  delta="Action required" if n > 0 else None,
                  delta_color="inverse" if n > 0 else "off")
    with m3:
        n = counts.get('RED', 0)
        st.metric("🔶 Red", n,
                  delta="Escalate" if n > 0 else None,
                  delta_color="inverse" if n > 0 else "off")
    with m4:
        n = counts.get('AMBER', 0)
        st.metric("🟡 Amber", n)
    with m5:
        n = counts.get('GREEN', 0)
        st.metric("🟢 Green", n)

    st.markdown("---")

    # Portfolio table
    if not filtered.empty:
        display_cols = {
            'project_id':       'ID',
            'project_name':     'Project',
            'status':           'Tier',
            'rf_variance_pct':  'RF Variance %',
            'arima_variance_pct': 'ARIMA %',
            'responsible_role': 'Owner',
            'response_window':  'Response Window',
            'financial_impact': 'Est. Impact'
        }

        display_df = filtered[
            [c for c in display_cols.keys() if c in filtered.columns]
        ].copy()
        display_df.columns = [
            display_cols[c] for c in display_df.columns
        ]

        # Format numeric columns
        if 'RF Variance %' in display_df.columns:
            display_df['RF Variance %'] = display_df[
                'RF Variance %'
            ].apply(lambda x: f"{x:+.2f}%" if x is not None else 'N/A')

        if 'ARIMA %' in display_df.columns:
            display_df['ARIMA %'] = display_df['ARIMA %'].apply(
                lambda x: f"{x:+.2f}%" if x is not None else 'N/A'
            )

        if 'Est. Impact' in display_df.columns:
            display_df['Est. Impact'] = display_df[
                'Est. Impact'
            ].apply(lambda x: fmt_currency(x) if x is not None else 'N/A')

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    # Tier distribution chart
    with st.expander("📊 Tier distribution chart"):
        tier_order  = ['CRITICAL', 'RED', 'AMBER', 'GREEN']
        tier_counts = [counts.get(t, 0) for t in tier_order]
        tier_colors = ['#922B21', '#E74C3C', '#E67E22', '#27AE60']

        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#F8FAFC')
        ax.set_facecolor('#F8FAFC')

        bars = ax.barh(
            tier_order, tier_counts,
            color=tier_colors, alpha=0.85,
            edgecolor='white', height=0.5
        )
        for bar, val in zip(bars, tier_counts):
            if val > 0:
                ax.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    va='center', fontsize=10, fontweight='bold'
                )

        ax.set_xlabel('Number of projects', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(projects_df):
    """Render the sidebar with project selection and framework info."""
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/280x60/1F4E79/FFFFFF"
            "?text=ML+Cost+Governance",
            use_column_width=True
        )

        st.markdown("### Select Project")

        # Build project options
        options = {
            f"{row['project_name']} [{row['train_test_flag'].upper()}]":
            int(row['project_id'])
            for _, row in projects_df.iterrows()
        }

        selected_label = st.selectbox(
            "Project",
            options=list(options.keys()),
            label_visibility="collapsed"
        )
        selected_id = options[selected_label]

        st.markdown("---")

        # Project metadata
        project_row = projects_df[
            projects_df['project_id'] == selected_id
        ].iloc[0]

        st.markdown("**Project details**")
        st.markdown(f"- **ID:** {selected_id}")
        st.markdown(f"- **Size:** {project_row['project_size']}")
        bac = project_row['budget_at_completion']
        if bac:
            st.markdown(f"- **BAC:** {fmt_currency(bac)}")
        st.markdown(
            f"- **Set:** {project_row['train_test_flag'].upper()}"
        )

        st.markdown("---")
        st.markdown("**Framework info**")
        st.markdown(
            "- Prediction point: **20% completion**\n"
            "- Min. lead time: **4 weeks**\n"
            "- Primary metric: **MAE**\n"
            "- Models: **RF + ARIMA**\n"
            "- Explainability: **SHAP**"
        )

        st.markdown("---")
        st.caption(
            "ML-Enhanced Cost Governance Framework  \n"
            "University of the West of Scotland  \n"
            "MSc Masters Project"
        )

    return selected_id


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        "<h1 style='color:#1F4E79; margin-bottom:4px;'>"
        "📊 ML-Enhanced Cost Governance Framework"
        "</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#555; margin-top:0;'>"
        "Predictive analytics for IT programme budget control — "
        "early warning, explained forecasts, actionable governance."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Load coordinator and project list
    try:
        coordinator = get_coordinator()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info(
            "Run the full pipeline first:\n\n"
            "```\n"
            "python pipeline/ingest.py\n"
            "python pipeline/clean.py\n"
            "python pipeline/split.py\n"
            "python features/engineer.py\n"
            "python models/train_rf.py\n"
            "python models/train_arima.py\n"
            "python explainability/shap_engine.py\n"
            "python governance/decision_matrix.py\n"
            "```"
        )
        return

    projects_df = get_all_projects()

    if projects_df.empty:
        st.warning(
            "No projects found in the database. "
            "Run the data pipeline first."
        )
        return

    # Sidebar — project selection
    selected_id = render_sidebar(projects_df)

    # Tab layout
    tab1, tab2, tab3 = st.tabs([
        "🎯  Project Forecast",
        "📁  Portfolio Overview",
        "ℹ️  About"
    ])

    # ── Tab 1: Project Forecast ───────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1, 3])

        with col_left:
            run_btn = st.button(
                "▶ Run Forecast",
                type="primary",
                use_container_width=True
            )
            st.caption(
                "Runs RF + ARIMA + SHAP + Governance "
                "in one click"
            )

        with col_right:
            project_name = projects_df[
                projects_df['project_id'] == selected_id
            ]['project_name'].iloc[0]
            st.markdown(
                f"**Selected:** {project_name}"
            )

        if run_btn or 'last_result' in st.session_state:
            if run_btn:
                with st.spinner(
                    f"Running forecast for {project_name}..."
                ):
                    result = coordinator.run_forecast(
                        selected_id,
                        verbose=False
                    )
                st.session_state['last_result']   = result
                st.session_state['last_project']  = selected_id

            result = st.session_state.get('last_result')

            # Show stale data warning if project changed
            if st.session_state.get('last_project') != selected_id:
                st.warning(
                    "⚠ Showing forecast for a different project. "
                    "Click Run Forecast to update."
                )

            if result and result.get('success'):
                st.markdown("---")

                # Panel 1 — Alert
                render_alert_panel(result)
                st.markdown("---")

                # Panels 2 and 3 side by side
                col_forecast, col_shap = st.columns([1, 1])
                with col_forecast:
                    render_forecast_panel(result)
                with col_shap:
                    render_shap_panel(result)

            elif result:
                st.error(
                    f"Forecast failed: {result.get('error')}"
                )
        else:
            st.info(
                "👆 Select a project from the sidebar and click "
                "**Run Forecast** to generate a prediction."
            )

    # ── Tab 2: Portfolio Overview ─────────────────────────────────────────────
    with tab2:
        render_portfolio_panel(coordinator)

    # ── Tab 3: About ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
## About This Framework

This dashboard is the presentation layer of the
**ML-Enhanced Cost Governance Framework** — an MSc Masters Project
at the University of the West of Scotland.

### What It Does
The framework predicts IT programme cost variance at the **20%
completion mark** — before the money is spent — and translates
every prediction into a specific management action.

### How It Works
| Component | Purpose |
|---|---|
| **Random Forest** | Learns cost variance patterns from 40+ completed projects |
| **ARIMA** | Models the time-series cost trajectory of each individual project |
| **SHAP** | Explains which factors drove each prediction |
| **Governance Layer** | Maps predictions to Green / Amber / Red / Critical actions |

### Design Principles
- **Actionability** — every forecast maps to a named action and owner
- **Explainability** — every forecast is explained in plain language
- **Integrability** — aligns with MSP and PRINCE2 governance cycles

### Data Source
DSLIB dataset (Vanhoucke, 2023; Batselier and Vanhoucke, 2015)
— 40+ completed real projects with known outcomes.

### Key Finding
The Random Forest model achieves a **21.5% improvement in MAE**
over the traditional EAC formula baseline, demonstrating that
ML-based forecasting provides meaningful value over existing methods.

---
*University of the West of Scotland — School of Computing,
Engineering and Physical Sciences — MSc Masters Project*
        """)


if __name__ == "__main__":
    main()