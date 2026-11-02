"""
tests/test_data_processing.py
------------------------------
Automated tests for critical data-processing and metric calculations.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (
    clean_customer,
    clean_subscription,
    clean_support,
    aggregate_support,
    merge_all,
    validate_join_coverage,
    REFERENCE_DATE,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal in-memory data matching actual workbook structure
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_customer():
    return pd.DataFrame({
        "customerid": ["CUST-001", "CUST-002", "CUST-003", None],
        "name": ["Alice", "Bob", "Carol", None],
        "country": ["India", None, "Nepal", None],
        "state": ["Maharashtra", "Delhi", "Kathmandu", None],
        "gender": ["Female", "Men", "Women", None],
        "dob": ["1990-01-15", "1985-06-20", "1978-11-05", None],
        "interests": ["Travel", None, None, None],
        "pincode": [None, None, None, None],
    })


@pytest.fixture
def raw_subscription():
    return pd.DataFrame({
        "customerid": ["CUST-001", "CUST-002", "CUST-003"],
        "subscription_start_date": ["2021-01-01", "2020-06-01", "2022-03-15"],
        "subscription_type": ["Organic", "Refferal", "Paid"],
        "renewal_date": ["2025-01-01", "2025-06-01", "2025-03-15"],
        "plan_type": ["Premium", "Standard", "Basic"],
        "contract_type": ["Annual", "Monthly", "Monthly"],
        "cancellation_date": [None, "2024-08-01", None],
        "cancellation_reason": [None, "Too expensive", None],
        "monthly_charges": [22.99, 13.99, 6.99],
        "cltv": [1800, 300, 250],
        "churn_score": [5, 85, 40],
    })


@pytest.fixture
def raw_support():
    return pd.DataFrame({
        "customerid": ["CUST-002", "CUST-002", "CUST-003"],
        "complaint_date": ["2024-07-10", "2024-07-15", "2024-05-01"],
        "escalations": ["Y", "N", "Y"],
        "csat_score": [20, 60, 30],
        "col_1": [None, None, None],
        "comment": ["angry", None, None],
    })


# ---------------------------------------------------------------------------
# Tests: clean_customer
# ---------------------------------------------------------------------------

class TestCleanCustomer:

    def test_drops_blank_row(self, raw_customer):
        df = clean_customer(raw_customer)
        assert df["customerid"].isnull().sum() == 0
        assert len(df) == 3

    def test_gender_normalisation(self, raw_customer):
        df = clean_customer(raw_customer)
        genders = set(df["gender"].dropna().unique())
        assert genders.issubset({"Male", "Female"}), f"Unexpected gender values: {genders}"

    def test_country_inference_from_state(self, raw_customer):
        df = clean_customer(raw_customer)
        # CUST-001 has India explicitly; CUST-002 has state Delhi → should infer India
        cust_02 = df[df["customerid"] == "CUST-002"]
        assert cust_02["country"].iloc[0] == "India"

    def test_age_computed(self, raw_customer):
        df = clean_customer(raw_customer)
        assert "age" in df.columns
        ages = df["age"].dropna()
        assert all(ages > 0)
        assert all(ages < 120)

    def test_age_group_present(self, raw_customer):
        df = clean_customer(raw_customer)
        assert "age_group" in df.columns

    def test_column_names_lowercase(self, raw_customer):
        df = clean_customer(raw_customer)
        for col in df.columns:
            assert col == col.lower(), f"Column not lowercase: {col}"


# ---------------------------------------------------------------------------
# Tests: clean_subscription
# ---------------------------------------------------------------------------

class TestCleanSubscription:

    def test_churn_target_definition(self, raw_subscription):
        df = clean_subscription(raw_subscription)
        assert "churned" in df.columns
        # CUST-002 has cancellation_date → churned = 1
        assert df.loc[df["customerid"] == "CUST-002", "churned"].iloc[0] == 1
        # CUST-001 no cancellation → churned = 0
        assert df.loc[df["customerid"] == "CUST-001", "churned"].iloc[0] == 0

    def test_referral_typo_fixed(self, raw_subscription):
        df = clean_subscription(raw_subscription)
        assert "Refferal" not in df["subscription_type"].values
        assert "Referral" in df["subscription_type"].values

    def test_tenure_months_positive(self, raw_subscription):
        df = clean_subscription(raw_subscription)
        assert "tenure_months" in df.columns
        assert (df["tenure_months"] >= 0).all()

    def test_dates_are_datetime(self, raw_subscription):
        df = clean_subscription(raw_subscription)
        assert pd.api.types.is_datetime64_any_dtype(df["subscription_start_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["cancellation_date"])

    def test_churn_rate_correct(self, raw_subscription):
        df = clean_subscription(raw_subscription)
        assert df["churned"].sum() == 1
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Tests: clean_support
# ---------------------------------------------------------------------------

class TestCleanSupport:

    def test_col_1_dropped(self, raw_support):
        df = clean_support(raw_support)
        assert "col_1" not in df.columns

    def test_escalations_boolean(self, raw_support):
        df = clean_support(raw_support)
        assert df["escalations"].dtype == bool or df["escalations"].dtype == object
        # Check mapped values
        vals = df["escalations"].dropna().unique()
        assert set(vals).issubset({True, False})

    def test_csat_numeric(self, raw_support):
        df = clean_support(raw_support)
        assert pd.api.types.is_numeric_dtype(df["csat_score"])


# ---------------------------------------------------------------------------
# Tests: aggregate_support
# ---------------------------------------------------------------------------

class TestAggregateSupport:

    def test_one_row_per_customer(self, raw_support):
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        assert agg["customerid"].is_unique

    def test_complaint_count_correct(self, raw_support):
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        cust_02 = agg[agg["customerid"] == "CUST-002"]
        assert cust_02["complaint_count"].iloc[0] == 2

    def test_escalation_any(self, raw_support):
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        # CUST-002 has one Y escalation → has_escalation = True
        cust_02 = agg[agg["customerid"] == "CUST-002"]
        assert cust_02["has_escalation"].iloc[0] == True

    def test_avg_csat(self, raw_support):
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        cust_02 = agg[agg["customerid"] == "CUST-002"]
        assert abs(cust_02["avg_csat_score"].iloc[0] - 40.0) < 0.1


# ---------------------------------------------------------------------------
# Tests: merge_all
# ---------------------------------------------------------------------------

class TestMergeAll:

    def test_no_duplicate_customers(self, raw_customer, raw_subscription, raw_support):
        df_c = clean_customer(raw_customer)
        df_s = clean_subscription(raw_subscription)
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        master = merge_all(df_c, df_s, agg)
        assert master["customerid"].is_unique

    def test_master_rows_equals_subscription_rows(self, raw_customer, raw_subscription, raw_support):
        df_c = clean_customer(raw_customer)
        df_s = clean_subscription(raw_subscription)
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        master = merge_all(df_c, df_s, agg)
        assert len(master) == len(df_s)

    def test_no_support_customer_gets_zero_complaints(self, raw_customer, raw_subscription, raw_support):
        df_c = clean_customer(raw_customer)
        df_s = clean_subscription(raw_subscription)
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        master = merge_all(df_c, df_s, agg)
        # CUST-001 has no support tickets
        cust_01 = master[master["customerid"] == "CUST-001"]
        assert cust_01["complaint_count"].iloc[0] == 0
        assert cust_01["has_escalation"].iloc[0] == False


# ---------------------------------------------------------------------------
# Tests: validate_join_coverage
# ---------------------------------------------------------------------------

class TestValidateJoinCoverage:

    def test_no_duplicates_in_master(self, raw_customer, raw_subscription, raw_support):
        df_c = clean_customer(raw_customer)
        df_s = clean_subscription(raw_subscription)
        df_sup = clean_support(raw_support)
        agg = aggregate_support(df_sup)
        master = merge_all(df_c, df_s, agg)
        report = validate_join_coverage(master, df_c)
        assert report["duplicate_rows_in_master"] == 0
