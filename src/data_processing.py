"""
src/data_processing.py
----------------------
Reusable data loading, validation, cleaning, and merging logic
for the Customer Churn Analysis project.

Dataset: customer_churn_data_raw.xlsx
Worksheets: db_customer, db_subscription, db_support
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_PATH = DATA_DIR / "customer_churn_data_raw.xlsx"

# Reference date used to compute tenure (day after the latest subscription date
# observed in the dataset, so it is reproducible without relying on today's date).
REFERENCE_DATE = pd.Timestamp("2025-01-01")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_sheets(filepath: str | Path = DEFAULT_DATA_PATH) -> dict[str, pd.DataFrame]:
    """Load all three worksheets from the Excel workbook.

    Returns a dict with keys 'db_customer', 'db_subscription', 'db_support'.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Place 'customer_churn_data_raw.xlsx' in the data/ folder. "
            "See data/README.md for instructions."
        )
    sheets = pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
    expected = {"db_customer", "db_subscription", "db_support"}
    missing = expected - set(sheets.keys())
    if missing:
        raise ValueError(f"Workbook is missing expected sheets: {missing}")
    return {k: sheets[k].copy() for k in expected}


# ---------------------------------------------------------------------------
# Per-sheet cleaning
# ---------------------------------------------------------------------------

def clean_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean db_customer worksheet."""
    df = df.copy()

    # Standardise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Drop completely blank rows
    df = df.dropna(subset=["customerid"])

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Normalise gender: consolidate "Women"→"Female", "Men"→"Male"
    gender_map = {"women": "Female", "men": "Male", "male": "Male", "female": "Female"}
    df["gender"] = df["gender"].str.lower().map(gender_map)

    # Normalise country: fill missing where state clearly indicates India
    india_states = {
        "maharashtra", "karnataka", "delhi", "nagaland", "meghalaya",
        "rajasthan", "uttar pradesh", "telangana",
    }
    mask_no_country = df["country"].isna()
    df.loc[mask_no_country, "country"] = df.loc[mask_no_country, "state"].apply(
        lambda s: "India" if str(s).lower() in india_states else np.nan
    )

    # Normalise interests to lowercase
    df["interests"] = df["interests"].str.lower()

    # Convert dob to datetime
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Derive age at reference date
    df["age"] = (
        (REFERENCE_DATE - df["dob"]).dt.days / 365.25
    ).round(1)

    # Derive age group
    bins = [0, 25, 35, 45, 55, 120]
    labels = ["<25", "25–34", "35–44", "45–54", "55+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    return df.reset_index(drop=True)


def clean_subscription(df: pd.DataFrame) -> pd.DataFrame:
    """Clean db_subscription worksheet."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["customerid"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Date columns
    for col in ["subscription_start_date", "renewal_date", "cancellation_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Fix known typo: "Refferal" → "Referral"
    df["subscription_type"] = df["subscription_type"].str.replace(
        "Refferal", "Referral", regex=False
    )

    # Numeric columns
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["cltv"] = pd.to_numeric(df["cltv"], errors="coerce")
    df["churn_score"] = pd.to_numeric(df["churn_score"], errors="coerce")

    # ----- Churn target definition -----
    # A customer is labelled as churned (1) if cancellation_date is not null.
    # cancellation_date and cancellation_reason are POST-CANCELLATION facts and
    # must NOT be used as predictive features.
    df["churned"] = df["cancellation_date"].notna().astype(int)

    # Tenure in months from subscription start to reference date (capped at 0)
    df["tenure_months"] = (
        (REFERENCE_DATE - df["subscription_start_date"]).dt.days / 30.44
    ).clip(lower=0).round(1)

    return df.reset_index(drop=True)


def clean_support(df: pd.DataFrame) -> pd.DataFrame:
    """Clean db_support worksheet."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["customerid"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Drop the empty col_1 column if it exists
    if "col_1" in df.columns:
        df = df.drop(columns=["col_1"])

    df["complaint_date"] = pd.to_datetime(df["complaint_date"], errors="coerce")
    df["escalations"] = df["escalations"].str.upper().map({"Y": True, "N": False})
    df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation & merging
# ---------------------------------------------------------------------------

def aggregate_support(df_support: pd.DataFrame) -> pd.DataFrame:
    """Aggregate support records to one row per customer.

    Multiple tickets per customer are collapsed to:
    - complaint_count: number of tickets
    - has_escalation: any ticket was escalated
    - avg_csat_score: mean CSAT
    - min_csat_score: worst CSAT (proxy for worst experience)
    """
    agg = df_support.groupby("customerid").agg(
        complaint_count=("csat_score", "count"),
        has_escalation=("escalations", "any"),
        avg_csat_score=("csat_score", "mean"),
        min_csat_score=("csat_score", "min"),
    ).reset_index()
    agg["avg_csat_score"] = agg["avg_csat_score"].round(1)
    return agg


def merge_all(
    df_customer: pd.DataFrame,
    df_subscription: pd.DataFrame,
    df_support_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Join the three cleaned tables on customerid.

    Cardinality checks:
    - db_customer  : 1 row per customerid (enforced by assertion)
    - db_subscription : 1 row per customerid (enforced by assertion)
    - db_support_agg : 0–1 rows per customerid after aggregation

    A LEFT join ensures all subscription customers are kept even if they have
    no support record.
    """
    assert df_customer["customerid"].is_unique, "db_customer has duplicate customerids"
    assert df_subscription["customerid"].is_unique, (
        "db_subscription has duplicate customerids"
    )

    df = df_subscription.merge(df_customer, on="customerid", how="left")
    df = df.merge(df_support_agg, on="customerid", how="left")

    # Fill support columns with sensible defaults for customers with no ticket
    df["complaint_count"] = df["complaint_count"].fillna(0).astype(int)
    df["has_escalation"] = df["has_escalation"].fillna(False)
    df["avg_csat_score"] = df["avg_csat_score"]  # NaN = no support contact
    df["min_csat_score"] = df["min_csat_score"]  # NaN = no support contact

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def load_and_prepare(filepath: str | Path = DEFAULT_DATA_PATH) -> dict[str, pd.DataFrame]:
    """Full pipeline: load → clean → aggregate → merge.

    Returns a dict with:
    - 'master'       : merged dataframe, one row per customer
    - 'customer'     : cleaned db_customer
    - 'subscription' : cleaned db_subscription
    - 'support_raw'  : cleaned db_support (one row per ticket)
    - 'support_agg'  : aggregated support (one row per customer)
    """
    sheets = load_raw_sheets(filepath)
    df_cust = clean_customer(sheets["db_customer"])
    df_sub = clean_subscription(sheets["db_subscription"])
    df_sup_raw = clean_support(sheets["db_support"])
    df_sup_agg = aggregate_support(df_sup_raw)
    df_master = merge_all(df_cust, df_sub, df_sup_agg)

    return {
        "master": df_master,
        "customer": df_cust,
        "subscription": df_sub,
        "support_raw": df_sup_raw,
        "support_agg": df_sup_agg,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_join_coverage(df_master: pd.DataFrame, df_customer: pd.DataFrame) -> dict:
    """Check that join did not lose or duplicate rows."""
    sub_ids = set(df_master["customerid"])
    cust_ids = set(df_customer["customerid"])
    return {
        "subscription_rows": len(df_master),
        "customer_rows": len(df_customer),
        "ids_in_sub_not_customer": sub_ids - cust_ids,
        "ids_in_customer_not_sub": cust_ids - sub_ids,
        "duplicate_rows_in_master": int(df_master["customerid"].duplicated().sum()),
    }


def data_quality_report(data: dict[str, pd.DataFrame]) -> dict:
    """Return a summary dict of data-quality metrics for all tables."""
    report = {}
    for name, df in data.items():
        if df is None or df.empty:
            continue
        report[name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }
    return report
