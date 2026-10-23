"""
tests/test_analysis.py
----------------------
Automated tests for KPI and analysis calculation functions.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import (
    compute_kpis,
    churn_by_column,
    cancellation_reasons,
)


@pytest.fixture
def sample_master():
    """Minimal master dataframe matching expected structure."""
    return pd.DataFrame({
        "customerid": ["C001", "C002", "C003", "C004", "C005"],
        "churned": [0, 1, 0, 1, 0],
        "monthly_charges": [22.99, 13.99, 6.99, 13.99, 22.99],
        "cltv": [1800, 300, 250, 400, 1200],
        "tenure_months": [48.0, 12.0, 24.0, 8.0, 60.0],
        "contract_type": ["Annual", "Monthly", "Annual", "Monthly", "Annual"],
        "plan_type": ["Premium", "Standard", "Basic", "Standard", "Premium"],
        "subscription_type": ["Organic", "Referral", "Organic", "Referral", "Paid"],
        "complaint_count": [0, 2, 0, 1, 0],
        "has_escalation": [False, True, False, True, False],
        "avg_csat_score": [np.nan, 25.0, np.nan, 30.0, np.nan],
        "cancellation_reason": [None, "Too expensive", None, "Competitor", None],
        "state": ["Maharashtra", "Delhi", "Karnataka", "Delhi", "Rajasthan"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
    })


class TestComputeKpis:

    def test_total_customers(self, sample_master):
        kpis = compute_kpis(sample_master)
        assert kpis["total_customers"] == 5

    def test_churned_count(self, sample_master):
        kpis = compute_kpis(sample_master)
        assert kpis["churned_customers"] == 2

    def test_retained_count(self, sample_master):
        kpis = compute_kpis(sample_master)
        assert kpis["retained_customers"] == 3

    def test_churn_rate(self, sample_master):
        kpis = compute_kpis(sample_master)
        assert kpis["churn_rate_pct"] == 40.0

    def test_total_monthly_charges_excludes_churned(self, sample_master):
        """Total monthly charges should sum only retained customers."""
        kpis = compute_kpis(sample_master)
        expected = sample_master.loc[sample_master["churned"] == 0, "monthly_charges"].sum()
        assert abs(kpis["total_monthly_charges"] - round(expected, 2)) < 0.01

    def test_churned_cltv(self, sample_master):
        kpis = compute_kpis(sample_master)
        expected = int(sample_master.loc[sample_master["churned"] == 1, "cltv"].sum())
        assert kpis["churned_cltv"] == expected

    def test_churn_rate_zero(self):
        df_no_churn = pd.DataFrame({
            "customerid": ["C1", "C2"],
            "churned": [0, 0],
            "monthly_charges": [10.0, 20.0],
            "cltv": [100, 200],
            "complaint_count": [0, 0],
            "has_escalation": [False, False],
            "avg_csat_score": [np.nan, np.nan],
        })
        kpis = compute_kpis(df_no_churn)
        assert kpis["churn_rate_pct"] == 0.0
        assert kpis["churned_customers"] == 0

    def test_churn_rate_hundred_percent(self):
        df_all_churn = pd.DataFrame({
            "customerid": ["C1", "C2"],
            "churned": [1, 1],
            "monthly_charges": [10.0, 20.0],
            "cltv": [100, 200],
            "complaint_count": [0, 0],
            "has_escalation": [False, False],
            "avg_csat_score": [np.nan, np.nan],
        })
        kpis = compute_kpis(df_all_churn)
        assert kpis["churn_rate_pct"] == 100.0


class TestChurnByColumn:

    def test_returns_dataframe(self, sample_master):
        result = churn_by_column(sample_master, "contract_type")
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, sample_master):
        result = churn_by_column(sample_master, "contract_type")
        expected_cols = {"contract_type", "total", "churned", "retained", "churn_rate_pct"}
        assert expected_cols.issubset(set(result.columns))

    def test_annual_churn_rate_zero(self, sample_master):
        result = churn_by_column(sample_master, "contract_type")
        annual = result[result["contract_type"] == "Annual"]
        assert annual["churn_rate_pct"].iloc[0] == 0.0

    def test_monthly_churn_rate_positive(self, sample_master):
        result = churn_by_column(sample_master, "contract_type")
        monthly = result[result["contract_type"] == "Monthly"]
        assert monthly["churn_rate_pct"].iloc[0] > 0

    def test_total_adds_up(self, sample_master):
        result = churn_by_column(sample_master, "contract_type")
        assert result["total"].sum() == len(sample_master)

    def test_churned_plus_retained_equals_total(self, sample_master):
        result = churn_by_column(sample_master, "contract_type")
        assert (result["churned"] + result["retained"] == result["total"]).all()


class TestCancellationReasons:

    def test_returns_only_churned(self, sample_master):
        result = cancellation_reasons(sample_master)
        # Total count should equal number of churned customers with a reason
        churned_with_reason = sample_master[
            (sample_master["churned"] == 1) &
            (sample_master["cancellation_reason"].notna())
        ]
        assert result["count"].sum() == len(churned_with_reason)

    def test_percentage_sums_to_100(self, sample_master):
        result = cancellation_reasons(sample_master)
        if not result.empty:
            assert abs(result["pct"].sum() - 100.0) < 0.2
