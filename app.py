"""
app.py
------
Customer Churn Analysis & Prediction — Interactive Streamlit Dashboard
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

Author : Umar Rahi
Dataset: customer_churn_data_raw.xlsx (21 customers, 3 worksheets)

Deployment: Render Web Service
  Build:  pip install -r requirements.txt
  Start:  streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
"""

import sys
import os
import logging
from pathlib import Path

# Make src importable when running from project root or Render working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import load_and_prepare, DEFAULT_DATA_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("churn_dashboard")

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analysis | Umar Rahi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Navy / Blue enterprise palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Layout ── */
    .stApp { background-color: #f4f6fb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e38 0%, #1a2b4a 60%, #1e3461 100%);
    }
    [data-testid="stSidebar"] * { color: #dce6f5 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    [data-testid="stSidebar"] hr { border-color: #2a4070 !important; }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px 14px 20px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 4px;
        height: 100%;
    }
    .kpi-card.danger  { border-left-color: #dc2626; }
    .kpi-card.success { border-left-color: #16a34a; }
    .kpi-card.neutral { border-left-color: #6b7280; }
    .kpi-label {
        font-size: 11px;
        color: #6b7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #0f1e38;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 5px;
        line-height: 1.4;
    }

    /* ── Section Headers ── */
    .section-header {
        background: linear-gradient(90deg, #0f1e38 0%, #2563eb 100%);
        color: #ffffff !important;
        padding: 9px 18px;
        border-radius: 8px;
        margin: 20px 0 14px 0;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* ── Alert Boxes ── */
    .limitation-box {
        background: #fef9ec;
        border: 1px solid #f59e0b;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #78350f;
        margin: 10px 0;
        line-height: 1.6;
    }
    .demo-banner {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #1e3a8a;
        margin: 10px 0;
        line-height: 1.6;
    }
    .info-box {
        background: #f0fdf4;
        border: 1px solid #22c55e;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 13px;
        color: #14532d;
        margin: 8px 0;
    }

    /* ── Tables ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Page title area ── */
    .dashboard-title {
        font-size: 28px;
        font-weight: 800;
        color: #0f1e38;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    .dashboard-subtitle {
        font-size: 14px;
        color: #4b5563;
        margin-bottom: 4px;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 11px;
        padding: 20px 0 6px 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 40px;
        line-height: 1.8;
    }
    .footer a { color: #2563eb; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------
def _make_demo_data() -> dict:
    """Generate clearly-labelled synthetic data for demo mode.

    All values are fabricated and do NOT represent real customers.
    Demo mode activates automatically when the real dataset is unavailable.
    """
    import random
    rng = np.random.default_rng(42)

    n = 21
    cids = [f"DEMO-{i+1:03d}" for i in range(n)]
    churned = np.array([0]*15 + [1]*6)
    rng.shuffle(churned)

    plan_types = rng.choice(["Basic", "Standard", "Premium"], n,
                             p=[0.3, 0.4, 0.3])
    contract_types = np.where(churned == 1,
                               rng.choice(["Monthly", "Annual"], n, p=[0.9, 0.1]),
                               rng.choice(["Monthly", "Annual"], n, p=[0.4, 0.6]))
    sub_types = rng.choice(["Organic", "Referral", "Paid"], n, p=[0.4, 0.35, 0.25])
    states = rng.choice(["Maharashtra", "Delhi", "Karnataka", "Rajasthan",
                          "Telangana", "Uttar Pradesh", "Nepal"], n)
    genders = rng.choice(["Male", "Female"], n)
    monthly_charges = np.where(
        plan_types == "Premium", rng.uniform(45, 95, n),
        np.where(plan_types == "Standard", rng.uniform(18, 45, n),
                 rng.uniform(7, 18, n))
    ).round(2)
    tenure_months = np.where(
        churned == 1, rng.uniform(2, 20, n), rng.uniform(6, 72, n)
    ).round(1)
    cltv = (monthly_charges * tenure_months * rng.uniform(0.8, 1.2, n)).round(0).astype(int)
    churn_score = np.where(churned == 1, rng.integers(55, 96, n),
                            rng.integers(10, 60, n))
    complaint_count = np.where(churned == 1,
                                rng.integers(1, 4, n), rng.integers(0, 2, n))
    has_escalation = (churned == 1) & (rng.random(n) > 0.35)
    avg_csat_score = np.where(
        complaint_count > 0,
        np.where(churned == 1, rng.uniform(15, 55, n), rng.uniform(55, 95, n)),
        np.nan
    )
    cancellation_reason = np.where(
        churned == 1,
        rng.choice(["Switched to competitor", "Too expensive",
                    "Not enough content", "Poor streaming quality",
                    "Forgot to cancel trial"], n),
        None
    )

    csat_score_raw = np.where(complaint_count > 0, avg_csat_score, np.nan)

    df_master = pd.DataFrame({
        "customerid": cids,
        "churned": churned,
        "plan_type": plan_types,
        "contract_type": contract_types,
        "subscription_type": sub_types,
        "state": states,
        "gender": genders,
        "monthly_charges": monthly_charges,
        "cltv": cltv,
        "churn_score": churn_score,
        "tenure_months": tenure_months,
        "complaint_count": complaint_count,
        "has_escalation": has_escalation,
        "avg_csat_score": avg_csat_score.round(1),
        "min_csat_score": (avg_csat_score * rng.uniform(0.7, 1.0, n)).round(1),
        "cancellation_reason": cancellation_reason,
        "age_group": rng.choice(["<25", "25–34", "35–44", "45–54", "55+"], n),
    })

    # Raw support tickets (some customers have none)
    ticket_cids = [cids[i] for i in range(n) if complaint_count[i] > 0]
    rows = []
    for cid_idx, cid in enumerate(cids):
        for _ in range(int(complaint_count[cid_idx])):
            rows.append({
                "customerid": cid,
                "csat_score": float(csat_score_raw[cid_idx]) if not np.isnan(csat_score_raw[cid_idx]) else np.nan,
                "escalations": has_escalation[cid_idx],
            })
    df_sup_raw = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["customerid", "csat_score", "escalations"])

    return {"master": df_master, "support_raw": df_sup_raw}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing dataset…")
def get_real_data():
    """Load the actual private dataset."""
    data = load_and_prepare(DEFAULT_DATA_PATH)
    from src.model import load_model_results
    model_results = load_model_results(data["master"])
    return data, model_results


@st.cache_data(show_spinner="Generating demo dataset…")
def get_demo_data():
    """Load demo (synthetic) data and run model workflow."""
    data = _make_demo_data()
    from src.model import load_model_results
    model_results = load_model_results(data["master"])
    return data, model_results


# ---------------------------------------------------------------------------
# Load data — real first, demo fallback
# ---------------------------------------------------------------------------
DEMO_MODE = False

try:
    data, model_results = get_real_data()
    df_master = data["master"]
    df_sup_raw = data["support_raw"]
    log.info("Real dataset loaded successfully (%d rows).", len(df_master))
except FileNotFoundError as fnf:
    log.warning("Dataset not found (%s) — switching to demo mode.", fnf)
    DEMO_MODE = True
    data, model_results = get_demo_data()
    df_master = data["master"]
    df_sup_raw = data["support_raw"]
except Exception as exc:
    log.error("Unexpected error loading dataset: %s", exc)
    st.error(f"⚠️ Unexpected error loading the dataset: {exc}")
    st.info(
        "If the problem persists, check that `customer_churn_data_raw.xlsx` is a valid "
        "Excel workbook with sheets `db_customer`, `db_subscription`, and `db_support`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Demo mode banner (prominent, stays visible throughout)
# ---------------------------------------------------------------------------
if DEMO_MODE:
    st.markdown("""
    <div class="demo-banner">
    🔵 <strong>DEMO MODE</strong> — The real dataset
    (<code>data/customer_churn_data_raw.xlsx</code>) was not found in this deployment.<br>
    All data shown below is <strong>clearly synthetic and randomly generated</strong>
    — it does NOT represent real customers or real business metrics.<br>
    To use real data: place <code>customer_churn_data_raw.xlsx</code> in the
    <code>data/</code> folder and restart the application.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
def _safe_unique(series: pd.Series) -> list:
    """Return sorted unique non-null values as plain Python strings."""
    return sorted([str(v) for v in series.dropna().unique()])


with st.sidebar:
    st.markdown("## 📊 Churn Dashboard")
    if DEMO_MODE:
        st.markdown(
            "<div style='background:#1e3a8a;border-radius:6px;padding:6px 10px;"
            "font-size:12px;color:#93c5fd;margin-bottom:8px;'>"
            "⚡ DEMO MODE — synthetic data</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#14532d;border-radius:6px;padding:6px 10px;"
            "font-size:12px;color:#86efac;margin-bottom:8px;'>"
            "✅ Real dataset loaded</div>",
            unsafe_allow_html=True,
        )
    st.markdown("**AICTE | IBM SkillsBuild 2026**")
    st.markdown("---")

    st.markdown("### 🔍 Filters")

    contract_opts = ["All"] + _safe_unique(df_master["contract_type"])
    sel_contract = st.selectbox("Contract Type", contract_opts)

    plan_opts = ["All"] + _safe_unique(df_master["plan_type"])
    sel_plan = st.selectbox("Plan Type", plan_opts)

    sub_opts = ["All"] + _safe_unique(df_master["subscription_type"])
    sel_sub = st.selectbox("Subscription Type", sub_opts)

    churn_opts = ["All", "Churned", "Retained"]
    sel_churn = st.selectbox("Churn Status", churn_opts)

    state_opts = ["All"] + _safe_unique(df_master["state"])
    sel_state = st.selectbox("State / Region", state_opts)

    mc_min = float(df_master["monthly_charges"].min())
    mc_max = float(df_master["monthly_charges"].max())
    if mc_min == mc_max:
        mc_max = mc_min + 1.0   # guard against degenerate slider
    sel_mc = st.slider(
        "Monthly Charges ($)",
        min_value=mc_min,
        max_value=mc_max,
        value=(mc_min, mc_max),
        step=0.5,
    )

    st.markdown("---")
    dataset_label = "Synthetic (demo)" if DEMO_MODE else "21 customers · 3 worksheets"
    st.markdown(
        f"<small>Dataset: {dataset_label}<br>Reference date: 2025-01-01</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#93c5fd;'>Author: Umar Rahi<br>"
        "AICTE IBM SkillsBuild 2026</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
df = df_master.copy()
if sel_contract != "All":
    df = df[df["contract_type"] == sel_contract]
if sel_plan != "All":
    df = df[df["plan_type"] == sel_plan]
if sel_sub != "All":
    df = df[df["subscription_type"] == sel_sub]
if sel_churn == "Churned":
    df = df[df["churned"] == 1]
elif sel_churn == "Retained":
    df = df[df["churned"] == 0]
if sel_state != "All":
    df = df[df["state"] == sel_state]
df = df[
    (df["monthly_charges"] >= sel_mc[0]) &
    (df["monthly_charges"] <= sel_mc[1])
]

if df.empty:
    st.warning(
        "⚠️ No customers match the selected filters. "
        "Adjust the sidebar filters to see data."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Import analysis helpers (after data load to avoid unnecessary imports on error)
# ---------------------------------------------------------------------------
from src.analysis import (
    compute_kpis,
    churn_by_column,
    cancellation_reasons,
    churn_timing_by_tenure,
)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
col_hdr, col_mode = st.columns([3, 1])
with col_hdr:
    st.markdown(
        '<p class="dashboard-title">📊 Customer Churn Analysis & Prediction</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="dashboard-subtitle">'
        'AI-Powered Customer Retention Dashboard · '
        'AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026'
        '</p>',
        unsafe_allow_html=True,
    )
with col_mode:
    if DEMO_MODE:
        st.info("⚡ **Demo Mode**\nSynthetic data only")
    else:
        st.success("✅ **Real Dataset**\n21 customers")

st.caption(
    f"Showing **{len(df)}** of {len(df_master)} customers after filters · "
    + ("Synthetic demo data" if DEMO_MODE
       else "Dataset: 21 customers · 3 worksheets (db_customer, db_subscription, db_support)")
)

if len(df) < len(df_master):
    st.info(
        f"ℹ️ Filters active — {len(df)} customers displayed. "
        "KPIs and charts reflect the filtered subset only."
    )


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Executive Overview",
    "📉 Churn Analysis",
    "👥 Customer Insights",
    "🤖 Prediction & Risk",
    "💡 Recommendations",
])

_CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f9fafb",
    font=dict(family="'Segoe UI', system-ui, sans-serif", size=12),
    margin=dict(t=50, b=30, l=40, r=20),
)


# ============================================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================================
with tab1:
    kpis = compute_kpis(df)

    st.markdown(
        '<div class="section-header">📌 Key Performance Indicators</div>',
        unsafe_allow_html=True,
    )

    # Row 1 — core churn metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{kpis['total_customers']:,}</div>
            <div class="kpi-sub">Unique IDs in filtered view</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card danger">
            <div class="kpi-label">Churned Customers</div>
            <div class="kpi-value">{kpis['churned_customers']:,}</div>
            <div class="kpi-sub">Confirmed cancellation on record</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card success">
            <div class="kpi-label">Retained Customers</div>
            <div class="kpi-value">{kpis['retained_customers']:,}</div>
            <div class="kpi-sub">No cancellation date recorded</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card danger">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value">{float(kpis['churn_rate_pct']):.1f}%</div>
            <div class="kpi-sub">Churned ÷ Total customers</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — financial metrics
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Monthly Revenue (Active)</div>
            <div class="kpi-value">${float(kpis['total_monthly_charges']):,.2f}</div>
            <div class="kpi-sub">Sum of retained customers' charges</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="kpi-card danger">
            <div class="kpi-label">CLTV at Risk</div>
            <div class="kpi-value">${kpis['churned_cltv']:,}</div>
            <div class="kpi-sub">Lifetime value of churned customers</div>
        </div>""", unsafe_allow_html=True)
    with c7:
        st.markdown(f"""<div class="kpi-card neutral">
            <div class="kpi-label">Avg Monthly Charge</div>
            <div class="kpi-value">${float(kpis['avg_monthly_charge']):.2f}</div>
            <div class="kpi-sub">All customers in filtered set</div>
        </div>""", unsafe_allow_html=True)
    with c8:
        avg_c = float(kpis['avg_cltv_churned']) if kpis['avg_cltv_churned'] == kpis['avg_cltv_churned'] else 0.0
        avg_r = float(kpis['avg_cltv_retained']) if kpis['avg_cltv_retained'] == kpis['avg_cltv_retained'] else 0.0
        st.markdown(f"""<div class="kpi-card neutral">
            <div class="kpi-label">Avg CLTV</div>
            <div class="kpi-value">${float(kpis['avg_cltv']):,.0f}</div>
            <div class="kpi-sub">Churned ${avg_c:,.0f} · Retained ${avg_r:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">📊 Churn vs Retention at a Glance</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig_donut = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[kpis["retained_customers"], kpis["churned_customers"]],
            hole=0.58,
            marker_colors=["#2563eb", "#dc2626"],
            textinfo="label+percent",
            textfont_size=12,
            pull=[0, 0.04],
        ))
        fig_donut.update_layout(
            title="Churn Distribution",
            height=300,
            showlegend=False,
            **{k: v for k, v in _CHART_LAYOUT.items() if k != "plot_bgcolor"},
        )
        fig_donut.add_annotation(
            text=f"<b>{float(kpis['churn_rate_pct']):.1f}%</b><br>Churn",
            x=0.5, y=0.5,
            font_size=15,
            showarrow=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        cltv_bar = go.Figure()
        cltv_bar.add_trace(go.Bar(
            x=["Retained", "Churned"],
            y=[avg_r, avg_c],
            marker_color=["#2563eb", "#dc2626"],
            text=[f"${avg_r:,.0f}", f"${avg_c:,.0f}"],
            textposition="auto",
            width=0.45,
        ))
        cltv_bar.update_layout(
            title="Average CLTV: Retained vs Churned",
            yaxis_title="Avg CLTV ($)",
            height=300,
            **_CHART_LAYOUT,
        )
        st.plotly_chart(cltv_bar, use_container_width=True)

    st.markdown(
        '<div class="limitation-box">'
        '⚠️ <strong>Dataset Limitation:</strong> '
        + ("This is <strong>synthetic demo data</strong> — numbers are randomly generated and carry no analytical meaning."
           if DEMO_MODE else
           "This analysis is based on <strong>21 customers</strong>. All percentages and averages are "
           "descriptive statistics of this small cohort and should not be generalised to a larger population "
           "without additional data collection.")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Customer data table (no names column)
    st.markdown(
        '<div class="section-header">📋 Customer Overview Table</div>',
        unsafe_allow_html=True,
    )
    display_cols = [
        "customerid", "plan_type", "contract_type", "subscription_type",
        "monthly_charges", "cltv", "tenure_months", "churned",
        "complaint_count", "has_escalation", "state",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    tbl = df[display_cols].copy()
    tbl["churned"] = tbl["churned"].map({1: "✅ Churned", 0: "Retained"})
    if "has_escalation" in tbl.columns:
        tbl["has_escalation"] = tbl["has_escalation"].map({True: "Yes", False: "No"})
    st.dataframe(tbl, use_container_width=True, height=270)
    st.caption(
        "ℹ️ Customer names from the raw dataset are intentionally excluded from all "
        "dashboard outputs to protect privacy."
    )


# ============================================================================
# TAB 2 — CHURN ANALYSIS
# ============================================================================
with tab2:
    st.markdown(
        '<div class="section-header">📉 Churn Rate Breakdown by Segment</div>',
        unsafe_allow_html=True,
    )

    def _churn_bar(col_name: str, title: str, x_label: str) -> go.Figure:
        grp = churn_by_column(df, col_name)
        if grp.empty:
            return None
        fig = px.bar(
            grp, x=col_name, y="churn_rate_pct",
            color="churn_rate_pct",
            color_continuous_scale=["#bfdbfe", "#1d4ed8", "#7f1d1d"],
            text="churn_rate_pct",
            labels={col_name: x_label, "churn_rate_pct": "Churn Rate (%)"},
            title=title,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            height=330,
            **_CHART_LAYOUT,
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        fig = _churn_bar("contract_type", "Churn Rate by Contract Type", "Contract Type")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for this segment with current filters.")
    with col2:
        fig2 = _churn_bar("plan_type", "Churn Rate by Plan Type", "Plan Type")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data available for this segment with current filters.")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = _churn_bar("subscription_type", "Churn Rate by Acquisition Channel",
                           "Subscription Type")
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data available for this segment with current filters.")
    with col4:
        cb_state = churn_by_column(df, "state")
        if not cb_state.empty:
            fig4 = px.bar(
                cb_state, x="state", y="churned",
                text="churn_rate_pct",
                labels={"state": "State / Region", "churned": "Churned Count"},
                title="Churned Customers by State / Region",
                color="churned",
                color_continuous_scale=["#bfdbfe", "#1d4ed8"],
            )
            fig4.update_traces(texttemplate="%{text}%", textposition="outside")
            fig4.update_layout(
                coloraxis_showscale=False,
                height=330,
                xaxis_tickangle=-30,
                **_CHART_LAYOUT,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No state data available with current filters.")

    # Cancellation reasons
    st.markdown(
        '<div class="section-header">❌ Cancellation Reasons</div>',
        unsafe_allow_html=True,
    )
    churned_df = df[df["churned"] == 1]
    if len(churned_df) > 0:
        reasons = cancellation_reasons(df)
        if not reasons.empty:
            col_ra, col_rb = st.columns([1, 1])
            with col_ra:
                fig_reasons = px.pie(
                    reasons, names="reason", values="count",
                    title="Cancellation Reasons",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    hole=0.3,
                )
                fig_reasons.update_traces(textinfo="label+percent")
                fig_reasons.update_layout(
                    height=330,
                    showlegend=True,
                    **{k: v for k, v in _CHART_LAYOUT.items() if k != "plot_bgcolor"},
                )
                st.plotly_chart(fig_reasons, use_container_width=True)
            with col_rb:
                st.markdown("**Reason Details**")
                st.dataframe(reasons, use_container_width=True, height=220)
                st.markdown(
                    f"**{len(churned_df)} churned customer(s)** in current view. "
                    "Post-cancellation reasons are shown for reference — they are "
                    "excluded from predictive features."
                )
        else:
            st.info("No cancellation reasons recorded for churned customers in this view.")
    else:
        st.info("No churned customers in the current filter selection.")

    # Tenure distribution
    st.markdown(
        '<div class="section-header">⏱ Tenure at Subscription: Churned vs Retained</div>',
        unsafe_allow_html=True,
    )
    if "tenure_months" in df.columns and not df["tenure_months"].isna().all():
        fig_ten = px.histogram(
            df,
            x="tenure_months",
            color=df["churned"].map({1: "Churned", 0: "Retained"}),
            barmode="overlay",
            opacity=0.75,
            nbins=10,
            labels={"tenure_months": "Tenure (months)", "color": "Status"},
            title="Customer Tenure Distribution",
            color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
        )
        fig_ten.update_layout(height=310, **_CHART_LAYOUT)
        st.plotly_chart(fig_ten, use_container_width=True)

        # Tenure-bin churn rate table
        tenure_tbl = churn_timing_by_tenure(df)
        if not tenure_tbl.empty:
            st.markdown("**Churn Rate by Tenure Bucket**")
            tenure_display = tenure_tbl.copy()
            tenure_display["churn_rate_pct"] = tenure_display["churn_rate_pct"].astype(str) + "%"
            st.dataframe(tenure_display, use_container_width=True, height=220)
    else:
        st.info("Tenure data not available for current filter selection.")


# ============================================================================
# TAB 3 — CUSTOMER INSIGHTS
# ============================================================================
with tab3:
    st.markdown(
        '<div class="section-header">💰 Monthly Charges & CLTV Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if "monthly_charges" in df.columns:
            status_col = df["churned"].map({1: "Churned", 0: "Retained"})
            fig_mc = px.box(
                df,
                x=status_col,
                y="monthly_charges",
                color=status_col,
                color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
                title="Monthly Charges by Churn Status",
                labels={"x": "Status", "monthly_charges": "Monthly Charges ($)"},
                points="all",
            )
            fig_mc.update_layout(height=330, showlegend=False, **_CHART_LAYOUT)
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("Monthly charges data not available.")

    with col2:
        if "cltv" in df.columns:
            status_col = df["churned"].map({1: "Churned", 0: "Retained"})
            fig_cltv = px.box(
                df,
                x=status_col,
                y="cltv",
                color=status_col,
                color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
                title="Customer Lifetime Value (CLTV) by Churn Status",
                labels={"x": "Status", "cltv": "CLTV ($)"},
                points="all",
            )
            fig_cltv.update_layout(height=330, showlegend=False, **_CHART_LAYOUT)
            st.plotly_chart(fig_cltv, use_container_width=True)
        else:
            st.info("CLTV data not available.")

    # Monthly charges vs CLTV scatter
    st.markdown(
        '<div class="section-header">🔍 Monthly Charges vs CLTV</div>',
        unsafe_allow_html=True,
    )
    if {"monthly_charges", "cltv", "tenure_months"}.issubset(df.columns):
        hover_cols = [c for c in ["customerid", "plan_type", "contract_type"]
                      if c in df.columns]
        fig_scatter = px.scatter(
            df,
            x="monthly_charges", y="cltv",
            color=df["churned"].map({1: "Churned", 0: "Retained"}),
            color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
            size="tenure_months",
            hover_data=hover_cols,
            title="Monthly Charges vs CLTV (bubble size = tenure)",
            labels={"monthly_charges": "Monthly Charges ($)", "cltv": "CLTV ($)"},
        )
        fig_scatter.update_layout(height=370, **_CHART_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Support analysis
    st.markdown(
        '<div class="section-header">🎧 Support Ticket Analysis</div>',
        unsafe_allow_html=True,
    )
    col3, col4 = st.columns(2)
    with col3:
        if "complaint_count" in df.columns:
            supp_grp = df[["customerid", "churned", "complaint_count"]].copy()
            supp_grp["status"] = supp_grp["churned"].map({1: "Churned", 0: "Retained"})
            fig_supp = px.bar(
                supp_grp.sort_values("complaint_count", ascending=False),
                x="customerid",
                y="complaint_count",
                color="status",
                color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
                title="Support Ticket Count per Customer",
                labels={"customerid": "Customer ID", "complaint_count": "Ticket Count"},
            )
            fig_supp.update_layout(
                height=330,
                xaxis_tickangle=-40,
                **_CHART_LAYOUT,
            )
            st.plotly_chart(fig_supp, use_container_width=True)
        else:
            st.info("Support ticket data not available.")

    with col4:
        if not df_sup_raw.empty and "csat_score" in df_sup_raw.columns:
            filtered_sup = df_sup_raw[df_sup_raw["customerid"].isin(df["customerid"])]
            if not filtered_sup.empty and not filtered_sup["csat_score"].isna().all():
                fig_csat = px.histogram(
                    filtered_sup.dropna(subset=["csat_score"]),
                    x="csat_score",
                    nbins=10,
                    color_discrete_sequence=["#2563eb"],
                    title="CSAT Score Distribution (Support Tickets)",
                    labels={"csat_score": "CSAT Score (0–100)"},
                )
                fig_csat.update_layout(height=330, **_CHART_LAYOUT)
                st.plotly_chart(fig_csat, use_container_width=True)
            else:
                st.info("No CSAT scores available for customers in this filter selection.")
        else:
            st.info("Support ticket data not available.")

    # Escalation analysis
    st.markdown(
        '<div class="section-header">🚨 Escalation Analysis</div>',
        unsafe_allow_html=True,
    )
    if "has_escalation" in df.columns:
        esc_df = df.copy()
        esc_df["status"] = esc_df["churned"].map({1: "Churned", 0: "Retained"})
        esc_df["escalation_label"] = esc_df["has_escalation"].map(
            {True: "Escalated", False: "No Escalation"}
        )
        esc_summary = (
            esc_df.groupby(["status", "escalation_label"])
            .size()
            .reset_index(name="count")
        )
        if not esc_summary.empty:
            fig_esc = px.bar(
                esc_summary,
                x="status", y="count",
                color="escalation_label",
                barmode="group",
                color_discrete_map={
                    "Escalated": "#f59e0b",
                    "No Escalation": "#2563eb",
                },
                title="Escalations: Churned vs Retained",
                labels={"status": "Customer Status", "count": "Customer Count"},
            )
            fig_esc.update_layout(height=310, **_CHART_LAYOUT)
            st.plotly_chart(fig_esc, use_container_width=True)
    else:
        st.info("Escalation data not available.")


# ============================================================================
# TAB 4 — PREDICTION & RISK ANALYSIS
# ============================================================================
with tab4:
    st.markdown(
        '<div class="section-header">🤖 Predictive Modelling — Exploratory Workflow</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="limitation-box">
    ⚠️ <strong>Important Modelling Limitation</strong><br>
    This dataset contains only <strong>21 customers</strong> (6 churned, 15 retained).
    Reliable ML evaluation requires substantially larger samples.
    A standard 80/20 stratified split would yield only ~4 test observations, making
    precision, recall, and AUC estimates statistically meaningless.<br><br>
    The models below were trained on <em>all 21 customers</em> and their metrics
    reflect <strong>in-sample fit only — NOT generalisation performance</strong>.
    These results demonstrate the modelling workflow and pipeline, and should
    <strong>NOT</strong> be used to make real business decisions.
    </div>
    """, unsafe_allow_html=True)

    if DEMO_MODE:
        st.markdown("""
        <div class="demo-banner">
        🔵 <strong>Demo Mode Active:</strong> Model results below are trained on
        <em>synthetic data</em>. They carry no analytical meaning.
        </div>
        """, unsafe_allow_html=True)

    if model_results and model_results.get("models"):
        models_dict = model_results["models"]

        # Model comparison table
        st.markdown("#### Model Comparison (In-Sample — Exploratory Only)")
        comparison_rows = []
        for model_name, res in models_dict.items():
            rep = res.get("classification_report", {})
            comparison_rows.append({
                "Model": model_name,
                "In-Sample Accuracy": f"{res.get('train_accuracy', 0):.3f}",
                "Precision (Churn class)": f"{rep.get('1', {}).get('precision', 0):.3f}",
                "Recall (Churn class)": f"{rep.get('1', {}).get('recall', 0):.3f}",
                "F1 (Churn class)": f"{rep.get('1', {}).get('f1-score', 0):.3f}",
                "ROC-AUC": (
                    f"{res['roc_auc']:.3f}" if res.get("roc_auc") else "N/A"
                ),
            })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
        st.caption(
            "⚠️ All metrics above are in-sample (train set = test set = entire dataset). "
            "They reflect how well models memorise the training data, NOT how well they "
            "generalise. Interpret with caution."
        )

        # Feature importance
        fi = model_results.get("feature_importance")
        if fi is not None and not fi.empty:
            st.markdown("#### Feature Importance — Random Forest (In-Sample)")
            fi_top = fi.head(12)
            fig_fi = px.bar(
                fi_top,
                x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#bfdbfe", "#1d4ed8"],
                title="Top Feature Importances (Random Forest, in-sample exploratory)",
                labels={"importance": "Importance Score", "feature": "Feature"},
            )
            fig_fi.update_layout(
                height=380,
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
                **_CHART_LAYOUT,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # Risk score table — uses df_master so all customers visible even if filtered
        risk_scores = model_results.get("risk_scores")
        if risk_scores is not None:
            merge_cols = [c for c in
                          ["customerid", "plan_type", "contract_type",
                           "monthly_charges", "churned"]
                          if c in df_master.columns]
            risk_with_info = risk_scores.merge(
                df_master[merge_cols], on="customerid", how="inner"
            )
            risk_with_info = risk_with_info[
                risk_with_info["customerid"].isin(df["customerid"])
            ]
            if not risk_with_info.empty:
                risk_with_info["actual_status"] = risk_with_info["churned"].map(
                    {1: "Churned", 0: "Retained"}
                )
                risk_with_info = risk_with_info.drop(columns=["churned"], errors="ignore")
                risk_with_info["churn_probability"] = (
                    risk_with_info["churn_probability"] * 100
                ).round(1)

                st.markdown("#### Customer Risk Scores (Exploratory, In-Sample)")
                display_risk_cols = [c for c in
                                     ["customerid", "plan_type", "contract_type",
                                      "monthly_charges", "risk_group",
                                      "churn_probability", "actual_status"]
                                     if c in risk_with_info.columns]
                st.dataframe(
                    risk_with_info[display_risk_cols],
                    use_container_width=True,
                    height=280,
                )

                # Risk group bar
                rg_count = risk_with_info["risk_group"].value_counts().reset_index()
                rg_count.columns = ["risk_group", "count"]
                order_map = {"Low": 0, "Medium": 1, "High": 2}
                rg_count["_order"] = rg_count["risk_group"].map(order_map)
                rg_count = rg_count.sort_values("_order").drop(columns="_order")
                fig_rg = px.bar(
                    rg_count,
                    x="risk_group", y="count",
                    color="risk_group",
                    color_discrete_map={
                        "Low": "#22c55e",
                        "Medium": "#f59e0b",
                        "High": "#dc2626",
                    },
                    title="Customers by Risk Group (Exploratory In-Sample)",
                    labels={"risk_group": "Risk Group", "count": "Customer Count"},
                )
                fig_rg.update_layout(height=310, showlegend=False, **_CHART_LAYOUT)
                st.plotly_chart(fig_rg, use_container_width=True)
            else:
                st.info("No risk score data available for the current filter selection.")
    else:
        st.info(
            "ℹ️ Predictive modelling workflow could not be completed. "
            "Check that `scikit-learn` is installed correctly."
        )

    # Existing churn score reference
    st.markdown(
        '<div class="section-header">📊 Pre-Existing Churn Score (Dataset Reference)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="limitation-box">
    ℹ️ The dataset includes a <strong>churn_score</strong> column in db_subscription.
    Its provenance is unknown — we cannot confirm whether it was calculated before or
    after the cancellation event. It has been <strong>excluded from all predictive
    features</strong> to prevent data leakage. It is displayed here for reference only.
    </div>
    """, unsafe_allow_html=True)

    if "churn_score" in df.columns and "churned" in df.columns:
        score_df = df[["customerid", "churn_score", "churned"]].copy()
        score_df["status"] = score_df["churned"].map({1: "Churned", 0: "Retained"})
        if not score_df["churn_score"].isna().all():
            fig_cs = px.scatter(
                score_df,
                x="customerid", y="churn_score",
                color="status",
                color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
                title="Existing Churn Score by Customer (Reference Only — Not Model Output)",
                labels={
                    "customerid": "Customer ID",
                    "churn_score": "Churn Score (0–100)",
                },
            )
            fig_cs.add_hline(
                y=70, line_dash="dash", line_color="orange",
                annotation_text="Score = 70 (informal threshold)"
            )
            fig_cs.update_layout(
                height=350,
                xaxis_tickangle=-35,
                **_CHART_LAYOUT,
            )
            st.plotly_chart(fig_cs, use_container_width=True)
        else:
            st.info("Churn score data not available for the current filter selection.")
    else:
        st.info("Churn score column not present in the current dataset.")


# ============================================================================
# TAB 5 — RECOMMENDATIONS
# ============================================================================
with tab5:
    st.markdown(
        '<div class="section-header">💡 Business Recommendations & Retention Strategies</div>',
        unsafe_allow_html=True,
    )

    if DEMO_MODE:
        st.markdown("""
        <div class="demo-banner">
        🔵 <strong>Demo Mode:</strong> The recommendations below are based on patterns
        observed in the <strong>real 21-customer dataset</strong>, not on the synthetic
        demo data currently displayed in other tabs. They are directional observations only.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    > The following insights are **observations from the real dataset** (21 customers).
    > Associations do not imply causation. Recommendations are directional only
    > and should be validated with a larger dataset before implementation.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔍 Observed Churn Drivers")
        st.markdown("""
**1. Monthly Contract = Higher Churn Risk**
- All 6 churned customers were on **Monthly** contracts.
- Annual-contract customers showed 0% churn in this dataset.
- *Strategy:* Offer incentives to migrate monthly subscribers to annual plans.

**2. Referral-Acquired Customers at Risk**
- 3 of 6 churned customers were acquired via Referral.
- *Strategy:* Build loyalty programs and post-referral onboarding flows.

**3. Cost Sensitivity: "Too Expensive"**
- Cited as a cancellation reason — affects Standard-plan monthly customers.
- *Strategy:* Consider flexible pricing, loyalty discounts, or pause options.

**4. Content Gap: "Not Enough Content"**
- One customer left due to content dissatisfaction.
- *Strategy:* Personalise content recommendations; survey at-risk users.

**5. Low CLTV Churners**
- Churned avg CLTV ≈ $341 vs retained avg ≈ $1,016.
- Early-stage customers appear most vulnerable.
- *Strategy:* Focus engagement programs on the first 3–6 months.
        """)

    with col2:
        st.markdown("### 🛡️ Retention Strategies by Segment")
        st.markdown("""
**High-Value Annual Customers (Retain)**
- Premium + Annual contracts show 0% churn here.
- Send renewal reminders 60 days before the renewal date.
- Offer early-renewal discounts to lock in another year.

**Monthly Contract Customers (Priority Risk)**
- Proactively offer annual plan migration at months 3–4.
- Provide a "loyalty rate" after 6+ months on the monthly plan.

**Escalation Incidents (Urgent)**
- 5 of 9 support tickets involved escalations.
- 4 escalated customers eventually churned.
- Implement a 24-hour callback SLA for escalated tickets.
- Personal outreach from account management post-escalation.

**Referral Customers (Engagement)**
- Send a welcome-journey series at days 7, 30, and 90.
- Evaluate incentive structures to avoid low-quality referrals.

**Content-Driven Churn (Product)**
- Survey new subscribers at day 30 about content satisfaction.
- Use churn reasons to prioritise content acquisition roadmap.
        """)

    # Insights chart — churn score vs monthly charges
    st.markdown(
        '<div class="section-header">📊 Churn Score vs Key Risk Indicators</div>',
        unsafe_allow_html=True,
    )
    plot_src = df_master  # always use full dataset for this insights chart
    if {"monthly_charges", "churn_score", "churned", "contract_type", "cltv"}.issubset(
        plot_src.columns
    ):
        hover_fields = [c for c in ["customerid", "plan_type", "cancellation_reason"]
                        if c in plot_src.columns]
        fig_insight = px.scatter(
            plot_src,
            x="monthly_charges",
            y="churn_score",
            color=plot_src["churned"].map({1: "Churned", 0: "Retained"}),
            color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
            symbol="contract_type",
            hover_data=hover_fields,
            size="cltv",
            size_max=22,
            title="Monthly Charges vs Churn Score — coloured by status, symbol = contract type",
            labels={
                "monthly_charges": "Monthly Charges ($)",
                "churn_score": "Existing Churn Score (Reference)",
            },
        )
        fig_insight.update_layout(height=390, **_CHART_LAYOUT)
        st.plotly_chart(fig_insight, use_container_width=True)
        if DEMO_MODE:
            st.caption(
                "⚡ This chart uses synthetic demo data — the churn score values "
                "are randomly generated and carry no analytical meaning."
            )
    else:
        st.info("Insufficient columns available for this chart.")

    # Limitations table
    st.markdown(
        '<div class="section-header">⚠️ Limitations & Privacy Considerations</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
| Topic | Note |
|---|---|
| **Sample size** | 21 customers is too small for statistically robust conclusions |
| **Churn target** | Defined by presence of `cancellation_date` — not all at-risk customers may have cancelled yet |
| **Churn score** | Origin unknown; excluded from ML features to prevent leakage |
| **Privacy** | Customer names are in the raw dataset but are **not** displayed in any analysis or dashboard output |
| **Causation** | All findings are correlational within this cohort only |
| **ML validity** | In-sample metrics only; no generalisation guarantee |
| **Geography** | Mix of Indian states and Nepal — should be segmented separately at scale |
| **Future work** | Collect more data, add product usage metrics, time-series survival analysis |
| **Demo mode** | Synthetic data is generated with fixed seed 42 and has no analytical value |
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Customer Churn Analysis &amp; Prediction ·
    AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026<br>
    Author: <strong>Umar Rahi</strong> ·
    Dataset: 21 customers · 3 worksheets ·
    <a href="https://github.com/Umar-Rahi/Customer-Churn-Analysis" target="_blank">GitHub Repo</a><br>
    Built with Streamlit · Python · Plotly · scikit-learn · pandas
</div>
""", unsafe_allow_html=True)
