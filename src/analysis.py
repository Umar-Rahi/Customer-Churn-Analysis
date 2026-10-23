"""
src/analysis.py
---------------
Reusable analytical calculation functions for the Customer Churn Analysis project.
All functions accept a cleaned master DataFrame and return analysis-ready structures.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# KPI / Summary metrics
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Return top-level KPI metrics from the master dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Merged master dataframe (one row per customer).

    Returns
    -------
    dict with keys:
        total_customers, churned_customers, retained_customers,
        churn_rate_pct, total_monthly_charges, avg_monthly_charge,
        total_cltv, churned_cltv, avg_cltv, avg_cltv_churned,
        avg_cltv_retained, complaint_rate_pct, escalation_rate_pct
    """
    n = len(df)
    n_churned = int(df["churned"].sum())
    n_retained = n - n_churned

    churn_rate = round((n_churned / n * 100), 1) if n > 0 else 0.0

    # Monthly charges: sum of all active customers' monthly charges
    # (churned customers are no longer billed; we report the current cohort total)
    total_monthly = round(df.loc[df["churned"] == 0, "monthly_charges"].sum(), 2)
    avg_monthly = round(df["monthly_charges"].mean(), 2)

    total_cltv = int(df["cltv"].sum())
    churned_cltv = int(df.loc[df["churned"] == 1, "cltv"].sum())
    avg_cltv = round(df["cltv"].mean(), 0)
    avg_cltv_churned = round(df.loc[df["churned"] == 1, "cltv"].mean(), 0)
    avg_cltv_retained = round(df.loc[df["churned"] == 0, "cltv"].mean(), 0)

    complaint_rate = round(
        (df["complaint_count"].gt(0).sum() / n * 100), 1
    ) if n > 0 else 0.0
    escalation_rate = round(
        (df["has_escalation"].sum() / n * 100), 1
    ) if n > 0 else 0.0

    return {
        "total_customers": n,
        "churned_customers": n_churned,
        "retained_customers": n_retained,
        "churn_rate_pct": churn_rate,
        "total_monthly_charges": total_monthly,
        "avg_monthly_charge": avg_monthly,
        "total_cltv": total_cltv,
        "churned_cltv": churned_cltv,
        "avg_cltv": avg_cltv,
        "avg_cltv_churned": avg_cltv_churned,
        "avg_cltv_retained": avg_cltv_retained,
        "complaint_rate_pct": complaint_rate,
        "escalation_rate_pct": escalation_rate,
    }


# ---------------------------------------------------------------------------
# Churn breakdown helpers
# ---------------------------------------------------------------------------

def churn_by_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return churn count and rate grouped by a categorical column.

    Returns
    -------
    DataFrame with columns: [col, 'total', 'churned', 'retained', 'churn_rate_pct']
    """
    grp = (
        df.groupby(col)["churned"]
        .agg(total="count", churned="sum")
        .reset_index()
    )
    grp["retained"] = grp["total"] - grp["churned"]
    grp["churn_rate_pct"] = (grp["churned"] / grp["total"] * 100).round(1)
    return grp.sort_values("churn_rate_pct", ascending=False).reset_index(drop=True)


def cancellation_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Count cancellation reasons for churned customers."""
    churned = df[df["churned"] == 1].copy()
    reasons = (
        churned["cancellation_reason"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "reason", "cancellation_reason": "count"})
    )
    reasons.columns = ["reason", "count"]
    reasons["pct"] = (reasons["count"] / reasons["count"].sum() * 100).round(1)
    return reasons


def monthly_charges_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for monthly_charges by churn status."""
    return (
        df.groupby("churned")["monthly_charges"]
        .describe()
        .round(2)
        .reset_index()
        .replace({0: "Retained", 1: "Churned"})
    )


def cltv_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for CLTV by churn status."""
    return (
        df.groupby("churned")["cltv"]
        .describe()
        .round(0)
        .reset_index()
        .replace({0: "Retained", 1: "Churned"})
    )


def support_churn_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare complaint count and CSAT between churned and retained."""
    return (
        df.groupby("churned")
        .agg(
            avg_complaints=("complaint_count", "mean"),
            pct_with_escalation=("has_escalation", "mean"),
            avg_csat=("avg_csat_score", "mean"),
        )
        .round(2)
        .reset_index()
        .replace({0: "Retained", 1: "Churned"})
    )


def churn_timing_by_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Bin customers by tenure and return churn rate per bin.

    Only includes churned customers who have a cancellation date, to derive
    how early in the subscription lifecycle churn tends to occur.
    """
    bins = [0, 6, 12, 24, 48, 999]
    labels = ["0–6 mo", "7–12 mo", "13–24 mo", "25–48 mo", "48+ mo"]
    df = df.copy()
    df["tenure_bin"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, right=True)
    grp = (
        df.groupby("tenure_bin")["churned"]
        .agg(total="count", churned="sum")
        .reset_index()
    )
    grp["churn_rate_pct"] = (grp["churned"] / grp["total"] * 100).round(1)
    return grp


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def churn_rate_matrix(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    """Pivot churn rate by two categorical variables (row × column).

    Useful for contract_type × plan_type, etc.
    Cells with n < 3 are replaced with NaN to avoid misleading rates.
    """
    ct = df.groupby([row_col, col_col])["churned"].agg(["sum", "count"]).reset_index()
    ct["rate"] = ct.apply(
        lambda r: round(r["sum"] / r["count"] * 100, 1) if r["count"] >= 3 else np.nan,
        axis=1,
    )
    pivot = ct.pivot(index=row_col, columns=col_col, values="rate")
    return pivot


# ---------------------------------------------------------------------------
# Support analysis helpers
# ---------------------------------------------------------------------------

def support_ticket_counts(df_support_raw: pd.DataFrame) -> pd.DataFrame:
    """Count tickets per customer with escalation flag."""
    return (
        df_support_raw.groupby("customerid")
        .agg(tickets=("csat_score", "count"), escalated=("escalations", "any"))
        .reset_index()
    )


def csat_distribution(df_support_raw: pd.DataFrame) -> pd.Series:
    """Return the CSAT score distribution."""
    return df_support_raw["csat_score"].describe()
