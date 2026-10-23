"""
app.py
------
Customer Churn Analysis & Prediction — Interactive Streamlit Dashboard
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

Author : Umar Rahi
Dataset: customer_churn_data_raw.xlsx (21 customers, 3 worksheets)
"""

import sys
import os
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_processing import load_and_prepare, DEFAULT_DATA_PATH
from src.analysis import (
    compute_kpis,
    churn_by_column,
    cancellation_reasons,
    monthly_charges_summary,
    cltv_summary,
    support_churn_summary,
    churn_timing_by_tenure,
)
from src.model import load_model_results

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analysis | Umar Rahi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — navy/blue palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f4f6fb; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a2b4a; color: #ffffff; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #c9d6e8 !important; }
    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 4px;
    }
    .kpi-card.warning { border-left-color: #dc2626; }
    .kpi-card.neutral { border-left-color: #6b7280; }
    .kpi-label { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #1a2b4a; line-height: 1.2; }
    .kpi-sub   { font-size: 12px; color: #9ca3af; margin-top: 4px; }
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a2b4a 0%, #2563eb 100%);
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        margin: 16px 0 12px 0;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    /* Limitation badge */
    .limitation-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #92400e;
        margin: 8px 0;
    }
    /* Chart container */
    .chart-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 20px 0 8px 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 32px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing dataset…")
def get_data():
    data = load_and_prepare(DEFAULT_DATA_PATH)
    model_results = load_model_results(data["master"])
    return data, model_results


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    data, model_results = get_data()
    df_master = data["master"]
    df_sup_raw = data["support_raw"]
except FileNotFoundError as e:
    st.error(f"⚠️ Dataset not found: {e}")
    st.info("Place `customer_churn_data_raw.xlsx` inside the `data/` folder and restart the app.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Customer Churn Dashboard")
    st.markdown("**AICTE | IBM SkillsBuild 2026**")
    st.markdown("---")

    st.markdown("### 🔍 Filters")

    # Contract type
    contract_opts = ["All"] + sorted(df_master["contract_type"].dropna().unique().tolist())
    sel_contract = st.selectbox("Contract Type", contract_opts)

    # Plan type
    plan_opts = ["All"] + sorted(df_master["plan_type"].dropna().unique().tolist())
    sel_plan = st.selectbox("Plan Type", plan_opts)

    # Subscription type
    sub_opts = ["All"] + sorted(df_master["subscription_type"].dropna().unique().tolist())
    sel_sub = st.selectbox("Subscription Type", sub_opts)

    # Churn status
    churn_opts = ["All", "Churned", "Retained"]
    sel_churn = st.selectbox("Churn Status", churn_opts)

    # State
    state_opts = ["All"] + sorted(df_master["state"].dropna().unique().tolist())
    sel_state = st.selectbox("State", state_opts)

    # Monthly charges range
    mc_min = float(df_master["monthly_charges"].min())
    mc_max = float(df_master["monthly_charges"].max())
    sel_mc = st.slider(
        "Monthly Charges (₹/USD)",
        min_value=mc_min,
        max_value=mc_max,
        value=(mc_min, mc_max),
        step=0.5,
    )

    st.markdown("---")
    st.markdown(
        "<small>Dataset: 21 customers · 3 worksheets<br>"
        "Reference date: 2025-01-01</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#c9d6e8;'>Author: Umar Rahi<br>"
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

# Guard against empty filtered data
if df.empty:
    st.warning("⚠️ No customers match the selected filters. Adjust the sidebar filters.")
    st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "# 📊 Customer Churn Analysis & Prediction",
)
st.markdown(
    "**AI-Powered Customer Retention Dashboard** · AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026"
)
st.caption(
    f"Showing **{len(df)}** of {len(df_master)} customers after filters · "
    "Dataset: 21 customers across 3 worksheets (db_customer, db_subscription, db_support)"
)

if len(df) < len(df_master):
    st.info(
        f"ℹ️ Filters active: {len(df)} customers displayed. "
        "KPIs and charts reflect the filtered subset only."
    )

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📉 Churn Analysis",
    "👥 Customer Insights",
    "🤖 Prediction / Risk",
    "💡 Recommendations",
])


# ============================================================================
# TAB 1 — OVERVIEW / EXECUTIVE SUMMARY
# ============================================================================
with tab1:
    kpis = compute_kpis(df)

    st.markdown('<div class="section-header">📌 Executive Summary — Key Performance Indicators</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{kpis['total_customers']}</div>
            <div class="kpi-sub">Unique customer IDs in filtered set</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card warning">
            <div class="kpi-label">Churned Customers</div>
            <div class="kpi-value">{kpis['churned_customers']}</div>
            <div class="kpi-sub">Confirmed cancellation on record</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Retained Customers</div>
            <div class="kpi-value">{kpis['retained_customers']}</div>
            <div class="kpi-sub">No cancellation date recorded</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card warning">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value">{kpis['churn_rate_pct']}%</div>
            <div class="kpi-sub">Churned ÷ Total customers</div>
        </div>""", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Monthly Revenue (Active)</div>
            <div class="kpi-value">${kpis['total_monthly_charges']:,.2f}</div>
            <div class="kpi-sub">Sum of retained customers' monthly charges</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="kpi-card warning">
            <div class="kpi-label">CLTV at Risk</div>
            <div class="kpi-value">${kpis['churned_cltv']:,}</div>
            <div class="kpi-sub">Lifetime value of churned customers</div>
        </div>""", unsafe_allow_html=True)
    with c7:
        st.markdown(f"""<div class="kpi-card neutral">
            <div class="kpi-label">Avg Monthly Charge</div>
            <div class="kpi-value">${kpis['avg_monthly_charge']}</div>
            <div class="kpi-sub">All customers in filtered set</div>
        </div>""", unsafe_allow_html=True)
    with c8:
        st.markdown(f"""<div class="kpi-card neutral">
            <div class="kpi-label">Avg CLTV</div>
            <div class="kpi-value">${kpis['avg_cltv']:,.0f}</div>
            <div class="kpi-sub">Avg: Churned ${kpis['avg_cltv_churned']:,.0f} · Retained ${kpis['avg_cltv_retained']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Churn vs Retention at a Glance</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        # Donut chart
        fig_donut = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[kpis["retained_customers"], kpis["churned_customers"]],
            hole=0.55,
            marker_colors=["#2563eb", "#dc2626"],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig_donut.update_layout(
            title="Churn Distribution",
            height=300,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
            paper_bgcolor="white",
        )
        fig_donut.add_annotation(
            text=f"<b>{kpis['churn_rate_pct']}%</b><br>Churn",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        # CLTV comparison bar
        cltv_bar = go.Figure()
        cltv_bar.add_trace(go.Bar(
            x=["Retained", "Churned"],
            y=[kpis["avg_cltv_retained"], kpis["avg_cltv_churned"]],
            marker_color=["#2563eb", "#dc2626"],
            text=[f"${kpis['avg_cltv_retained']:,.0f}", f"${kpis['avg_cltv_churned']:,.0f}"],
            textposition="auto",
        ))
        cltv_bar.update_layout(
            title="Average CLTV: Retained vs Churned",
            yaxis_title="Avg CLTV ($)",
            height=300,
            margin=dict(t=40, b=30, l=40, r=10),
            paper_bgcolor="white",
            plot_bgcolor="#f9fafb",
        )
        st.plotly_chart(cltv_bar, use_container_width=True)

    st.markdown('<div class="limitation-box">⚠️ <strong>Dataset Limitation:</strong> This analysis is based on <strong>21 customers</strong>. All percentages and averages are descriptive statistics of this small cohort and should not be generalised to a larger population without additional data collection.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Customer Data Table</div>', unsafe_allow_html=True)
    display_cols = [
        "customerid", "plan_type", "contract_type", "subscription_type",
        "monthly_charges", "cltv", "tenure_months", "churned",
        "complaint_count", "has_escalation", "state"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    tbl = df[display_cols].copy()
    tbl["churned"] = tbl["churned"].map({1: "✅ Churned", 0: "Retained"})
    tbl["has_escalation"] = tbl["has_escalation"].map({True: "Yes", False: "No"})
    st.dataframe(tbl, use_container_width=True, height=260)


# ============================================================================
# TAB 2 — CHURN ANALYSIS
# ============================================================================
with tab2:
    st.markdown('<div class="section-header">📉 Churn Breakdown by Segment</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Churn by contract type
        cb_contract = churn_by_column(df, "contract_type")
        fig = px.bar(
            cb_contract, x="contract_type", y="churn_rate_pct",
            color="churn_rate_pct",
            color_continuous_scale=["#bfdbfe", "#1d4ed8", "#7f1d1d"],
            text="churn_rate_pct",
            labels={"contract_type": "Contract Type", "churn_rate_pct": "Churn Rate (%)"},
            title="Churn Rate by Contract Type",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f9fafb",
            coloraxis_showscale=False, height=320,
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by plan type
        cb_plan = churn_by_column(df, "plan_type")
        fig2 = px.bar(
            cb_plan, x="plan_type", y="churn_rate_pct",
            color="churn_rate_pct",
            color_continuous_scale=["#bfdbfe", "#1d4ed8", "#7f1d1d"],
            text="churn_rate_pct",
            labels={"plan_type": "Plan Type", "churn_rate_pct": "Churn Rate (%)"},
            title="Churn Rate by Plan Type",
        )
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f9fafb",
            coloraxis_showscale=False, height=320,
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Churn by subscription type
        cb_sub = churn_by_column(df, "subscription_type")
        fig3 = px.bar(
            cb_sub, x="subscription_type", y="churn_rate_pct",
            color="churn_rate_pct",
            color_continuous_scale=["#bfdbfe", "#1d4ed8", "#7f1d1d"],
            text="churn_rate_pct",
            labels={"subscription_type": "Subscription Type", "churn_rate_pct": "Churn Rate (%)"},
            title="Churn Rate by Acquisition Channel",
        )
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        fig3.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f9fafb",
            coloraxis_showscale=False, height=320,
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Churn by state
        cb_state = churn_by_column(df, "state")
        fig4 = px.bar(
            cb_state, x="state", y="churned",
            text="churn_rate_pct",
            labels={"state": "State", "churned": "Churned Count"},
            title="Churned Customers by State",
            color="churned",
            color_continuous_scale=["#bfdbfe", "#1d4ed8"],
        )
        fig4.update_traces(texttemplate="%{text}%", textposition="outside")
        fig4.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f9fafb",
            coloraxis_showscale=False, height=320,
            margin=dict(t=50, b=30),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">❌ Cancellation Reasons</div>', unsafe_allow_html=True)

    churned_df = df[df["churned"] == 1]
    if len(churned_df) > 0:
        reasons = cancellation_reasons(df)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            fig_reasons = px.pie(
                reasons, names="reason", values="count",
                title="Cancellation Reasons Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.3,
            )
            fig_reasons.update_traces(textinfo="label+percent")
            fig_reasons.update_layout(
                paper_bgcolor="white", height=320,
                margin=dict(t=50, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig_reasons, use_container_width=True)
        with col_b:
            st.markdown("**Cancellation Reason Details**")
            st.dataframe(reasons, use_container_width=True, height=200)
            st.markdown(
                f"**{len(churned_df)} churned customers** in the filtered view. "
                "All had a documented cancellation reason."
            )
    else:
        st.info("No churned customers in the current filter selection.")

    # Tenure at churn
    st.markdown('<div class="section-header">⏱ Tenure Distribution: Churned vs Retained</div>', unsafe_allow_html=True)
    fig_ten = px.histogram(
        df, x="tenure_months", color=df["churned"].map({1: "Churned", 0: "Retained"}),
        barmode="overlay",
        opacity=0.75,
        nbins=10,
        labels={"tenure_months": "Tenure (months)", "color": "Status"},
        title="Customer Tenure Distribution",
        color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
    )
    fig_ten.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f9fafb", height=300,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_ten, use_container_width=True)


# ============================================================================
# TAB 3 — CUSTOMER INSIGHTS
# ============================================================================
with tab3:
    st.markdown('<div class="section-header">💰 Monthly Charges Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_mc = px.box(
            df,
            x=df["churned"].map({1: "Churned", 0: "Retained"}),
            y="monthly_charges",
            color=df["churned"].map({1: "Churned", 0: "Retained"}),
            color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
            title="Monthly Charges by Churn Status",
            labels={"x": "Status", "monthly_charges": "Monthly Charges ($)"},
            points="all",
        )
        fig_mc.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f9fafb", height=320,
            margin=dict(t=50, b=30), showlegend=False,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col2:
        fig_cltv = px.box(
            df,
            x=df["churned"].map({1: "Churned", 0: "Retained"}),
            y="cltv",
            color=df["churned"].map({1: "Churned", 0: "Retained"}),
            color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
            title="Customer Lifetime Value (CLTV) by Churn Status",
            labels={"x": "Status", "cltv": "CLTV ($)"},
            points="all",
        )
        fig_cltv.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f9fafb", height=320,
            margin=dict(t=50, b=30), showlegend=False,
        )
        st.plotly_chart(fig_cltv, use_container_width=True)

    st.markdown('<div class="section-header">🎧 Support Analysis</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        # Complaint count per customer
        supp_grp = df[["customerid", "churned", "complaint_count", "has_escalation"]].copy()
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
            paper_bgcolor="white", plot_bgcolor="#f9fafb", height=320,
            margin=dict(t=50, b=60), xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_supp, use_container_width=True)

    with col4:
        # CSAT scores from raw support data
        if not df_sup_raw.empty:
            # Filter raw support to customers in filtered df only
            filtered_sup = df_sup_raw[df_sup_raw["customerid"].isin(df["customerid"])]
            if len(filtered_sup) > 0:
                fig_csat = px.histogram(
                    filtered_sup, x="csat_score", nbins=10,
                    color_discrete_sequence=["#2563eb"],
                    title="CSAT Score Distribution (Support Tickets)",
                    labels={"csat_score": "CSAT Score (0–100)"},
                )
                fig_csat.update_layout(
                    paper_bgcolor="white", plot_bgcolor="#f9fafb", height=320,
                    margin=dict(t=50, b=30),
                )
                st.plotly_chart(fig_csat, use_container_width=True)
            else:
                st.info("No support tickets for customers in this filter selection.")
        else:
            st.info("Support data not available.")

    # Escalation summary
    st.markdown('<div class="section-header">🚨 Escalation Analysis</div>', unsafe_allow_html=True)
    esc_df = df.copy()
    esc_df["status"] = esc_df["churned"].map({1: "Churned", 0: "Retained"})
    esc_summary = esc_df.groupby(["status", "has_escalation"]).size().reset_index(name="count")
    esc_summary["escalation_label"] = esc_summary["has_escalation"].map({True: "Escalated", False: "No Escalation"})

    fig_esc = px.bar(
        esc_summary,
        x="status", y="count",
        color="escalation_label",
        barmode="group",
        color_discrete_map={"Escalated": "#f59e0b", "No Escalation": "#2563eb"},
        title="Escalations: Churned vs Retained",
        labels={"status": "Customer Status", "count": "Customer Count"},
    )
    fig_esc.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f9fafb", height=300,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_esc, use_container_width=True)

    # Monthly charges scatter
    st.markdown('<div class="section-header">🔍 Monthly Charges vs CLTV</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        df,
        x="monthly_charges", y="cltv",
        color=df["churned"].map({1: "Churned", 0: "Retained"}),
        color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
        size="tenure_months",
        hover_data=["customerid", "plan_type", "contract_type"],
        title="Monthly Charges vs CLTV (bubble size = tenure)",
        labels={"monthly_charges": "Monthly Charges ($)", "cltv": "CLTV ($)"},
    )
    fig_scatter.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f9fafb", height=360,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================================
# TAB 4 — PREDICTION / RISK ANALYSIS
# ============================================================================
with tab4:
    st.markdown('<div class="section-header">🤖 Predictive Modelling Workflow</div>', unsafe_allow_html=True)

    # Always show the limitation notice
    st.markdown("""
    <div class="limitation-box">
    ⚠️ <strong>Important Modelling Limitation</strong><br>
    This dataset contains only <strong>21 customers</strong> (6 churned, 15 retained).
    Reliable machine-learning evaluation requires substantially larger samples.
    A standard 80/20 stratified split would yield only ~4 test observations, making
    precision, recall, and AUC estimates statistically meaningless.<br><br>
    The models below were trained on <em>all 21 customers</em> and their metrics reflect
    <strong>in-sample fit only — NOT generalisation performance</strong>.
    These results demonstrate the modelling workflow and should NOT be used to make
    real business decisions.
    </div>
    """, unsafe_allow_html=True)

    if model_results and model_results.get("models"):
        models = model_results["models"]

        # Model comparison table
        st.markdown("#### Model Comparison (In-Sample — Exploratory Only)")
        comparison_rows = []
        for model_name, res in models.items():
            rep = res["classification_report"]
            comparison_rows.append({
                "Model": model_name,
                "Accuracy": f"{res['train_accuracy']:.3f}",
                "Precision (Churn)": f"{rep.get('1', {}).get('precision', 0):.3f}",
                "Recall (Churn)": f"{rep.get('1', {}).get('recall', 0):.3f}",
                "F1 (Churn)": f"{rep.get('1', {}).get('f1-score', 0):.3f}",
                "ROC-AUC": f"{res['roc_auc']:.3f}" if res["roc_auc"] else "N/A",
            })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
        st.caption("⚠️ All metrics above are in-sample (train = test = all 21 customers). Interpret with caution.")

        # Feature importance
        fi = model_results.get("feature_importance")
        if fi is not None and not fi.empty:
            st.markdown("#### Feature Importance (Random Forest, In-Sample)")
            fig_fi = px.bar(
                fi.head(10),
                x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#bfdbfe", "#1d4ed8"],
                title="Top Feature Importances",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig_fi.update_layout(
                paper_bgcolor="white", plot_bgcolor="#f9fafb", height=350,
                margin=dict(t=50, b=30), coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # Risk scores table
        risk_scores = model_results.get("risk_scores")
        if risk_scores is not None:
            risk_with_info = risk_scores.merge(
                df_master[["customerid", "plan_type", "contract_type", "monthly_charges", "churned"]],
                on="customerid", how="inner"
            )
            risk_with_info = risk_with_info[
                risk_with_info["customerid"].isin(df["customerid"])
            ]
            risk_with_info["actual_status"] = risk_with_info["churned"].map({1: "Churned", 0: "Retained"})
            risk_with_info = risk_with_info.drop(columns=["churned"])
            risk_with_info["churn_probability"] = (risk_with_info["churn_probability"] * 100).round(1)

            st.markdown("#### Customer Risk Scores (Exploratory, In-Sample)")
            st.dataframe(
                risk_with_info[["customerid", "plan_type", "contract_type",
                                "monthly_charges", "risk_group", "churn_probability", "actual_status"]],
                use_container_width=True,
                height=280,
            )

            # Risk group bar
            rg_count = risk_with_info["risk_group"].value_counts().reset_index()
            rg_count.columns = ["risk_group", "count"]
            fig_rg = px.bar(
                rg_count,
                x="risk_group", y="count",
                color="risk_group",
                color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#dc2626"},
                title="Customers by Risk Group (Exploratory)",
                labels={"risk_group": "Risk Group", "count": "Customer Count"},
            )
            fig_rg.update_layout(
                paper_bgcolor="white", plot_bgcolor="#f9fafb", height=300,
                margin=dict(t=50, b=30), showlegend=False,
            )
            st.plotly_chart(fig_rg, use_container_width=True)
    else:
        st.info("Predictive modelling workflow could not be completed. Check that scikit-learn is installed.")

    # Churn score note
    st.markdown('<div class="section-header">📊 Pre-Existing Churn Score (from Dataset)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="limitation-box">
    ℹ️ The dataset includes a <strong>churn_score</strong> column in db_subscription.
    Its provenance is unknown — we cannot confirm whether it was calculated before or after
    cancellation events. It has been <strong>excluded from all predictive features</strong>
    to prevent data leakage. It is displayed here for reference only.
    </div>
    """, unsafe_allow_html=True)

    score_df = df[["customerid", "churn_score", "churned"]].copy()
    score_df["status"] = score_df["churned"].map({1: "Churned", 0: "Retained"})
    fig_cs = px.scatter(
        score_df, x="customerid", y="churn_score",
        color="status",
        color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
        title="Existing Churn Score by Customer (Reference Only — Not Model Output)",
        labels={"customerid": "Customer ID", "churn_score": "Churn Score"},
    )
    fig_cs.add_hline(y=70, line_dash="dash", line_color="orange",
                     annotation_text="Score = 70 (informal threshold)")
    fig_cs.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f9fafb", height=340,
        margin=dict(t=50, b=60), xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_cs, use_container_width=True)


# ============================================================================
# TAB 5 — RECOMMENDATIONS
# ============================================================================
with tab5:
    st.markdown('<div class="section-header">💡 Business Recommendations & Retention Strategies</div>', unsafe_allow_html=True)

    st.markdown("""
    > The following insights are **observations from the dataset** (21 customers).
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
        - Cited as a cancellation reason — affects Standard plan monthly customers.
        - *Strategy:* Consider flexible pricing, loyalty discounts, or pause options.

        **4. Content Gap: "Not Enough Content"**
        - One customer left due to content dissatisfaction.
        - *Strategy:* Personalise content recommendations; survey at-risk users.

        **5. Low CLTV Churners**
        - Churned avg CLTV ≈ $162 vs retained avg ≈ $965.
        - Early-stage customers appear most vulnerable.
        - *Strategy:* Focus engagement programs on the first 3–6 months.
        """)

    with col2:
        st.markdown("### 🛡️ Retention Strategies by Segment")
        st.markdown("""
        **High-Value Annual Customers (Retain)**
        - Premium + Annual contracts show 0% churn here.
        - Send renewal reminders 60 days before renewal date.
        - Offer early-renewal discounts to lock in another year.

        **Monthly Contract Customers (Priority Risk)**
        - Proactively offer annual plan migration at month 3–4.
        - Provide a "loyalty rate" after 6+ months on monthly plan.

        **Escalation Incidents (Urgent)**
        - 5 of 9 support tickets involved escalations.
        - 4 escalated customers eventually churned.
        - Implement a 24-hour callback SLA for escalated tickets.
        - Personal outreach from account management post-escalation.

        **Referral Customers (Engagement)**
        - Send welcome journey emails in weeks 1, 4, 12.
        - Referral customers may have lower intrinsic motivation.
        - Evaluate incentive structures: avoid low-quality referrals.

        **Content-Driven Churn (Product)**
        - Survey new subscribers at day 30 about content satisfaction.
        - Use churn reasons to prioritise content acquisition roadmap.
        """)

    st.markdown('<div class="section-header">📊 Churn Score vs Key Risk Indicators</div>', unsafe_allow_html=True)

    # Churn score vs monthly charges scatter with cancellation reason tooltip
    fig_insight = px.scatter(
        df_master,
        x="monthly_charges",
        y="churn_score",
        color=df_master["churned"].map({1: "Churned", 0: "Retained"}),
        color_discrete_map={"Churned": "#dc2626", "Retained": "#2563eb"},
        symbol="contract_type",
        hover_data=["customerid", "plan_type", "cancellation_reason"],
        size="cltv",
        title="Monthly Charges vs Churn Score — coloured by churn, symbol = contract type",
        labels={
            "monthly_charges": "Monthly Charges ($)",
            "churn_score": "Existing Churn Score (Reference)",
        },
    )
    fig_insight.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f9fafb", height=380,
        margin=dict(t=60, b=30),
    )
    st.plotly_chart(fig_insight, use_container_width=True)

    st.markdown('<div class="section-header">⚠️ Limitations & Privacy Considerations</div>', unsafe_allow_html=True)
    st.markdown("""
    | Topic | Note |
    |---|---|
    | **Sample size** | 21 customers is too small for statistically robust conclusions |
    | **Churn target** | Defined by presence of `cancellation_date` — not all at-risk customers may have cancelled yet |
    | **Churn score** | Origin unknown; excluded from ML features to prevent leakage |
    | **Privacy** | Customer names are in the raw dataset but not displayed in analysis or dashboard |
    | **Causation** | All findings are correlational within this cohort only |
    | **ML validity** | In-sample metrics only; no generalisation guarantee |
    | **Geography** | Mix of Indian states and Nepal — should be segmented separately at scale |
    | **Future work** | Collect more data, add product usage metrics, time-series survival analysis |
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Customer Churn Analysis & Prediction — AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026<br>
    Author: <strong>Umar Rahi</strong> · Dataset: 21 customers · 3 worksheets ·
    <a href="https://github.com/Umar-Rahi/Customer-Churn-Analysis" target="_blank">GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
