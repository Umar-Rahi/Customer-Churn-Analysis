"""
src/model.py
------------
Machine-learning utilities for the Customer Churn Analysis project.

IMPORTANT TRANSPARENCY NOTE
----------------------------
The dataset contains only 21 customers, of which 6 churned (28.6% churn rate).
Reliable ML model evaluation requires substantially more independent labelled
observations. With 21 samples a standard 80/20 stratified split yields only
~4–5 test samples, making precision, recall, and AUC estimates statistically
meaningless.

This module therefore:
1. Documents and enforces a minimum-sample guard (MIN_SAMPLES_FOR_MODELLING).
2. Provides a clearly-labelled "exploratory workflow only" mode that fits models
   and reports metrics but attaches explicit warnings.
3. Returns a structured result dict that the notebook and dashboard use to
   render appropriate caveats.

The `churn_score` column is treated as a potential leakage / proxy variable and
is excluded from predictive features.
"""

import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Guard constants
# ---------------------------------------------------------------------------
MIN_SAMPLES_FOR_MODELLING = 50   # minimum rows before modelling is attempted
MIN_CLASS_SAMPLES = 10           # minimum samples per class

# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

# Features that must NOT be used (post-cancellation / leakage):
LEAKAGE_COLS = [
    "cancellation_date",
    "cancellation_reason",
    "churned",          # the target itself
    "churn_score",      # provenance unknown — treat as leakage
    "customerid",
    "name",
    "dob",
    "pincode",
    "interests",
    "renewal_date",     # too close to churn event
    "subscription_start_date",
]

NUMERIC_FEATURES = ["monthly_charges", "cltv", "tenure_months", "complaint_count"]
CATEGORICAL_FEATURES = [
    "subscription_type", "plan_type", "contract_type", "gender", "age_group"
]

TARGET = "churned"


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y from master dataframe."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Keep only columns that actually exist in the dataframe
    existing_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    existing_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[existing_num + existing_cat].copy()
    y = df[TARGET].copy()
    return X, y


def can_train_model(df: pd.DataFrame) -> tuple[bool, str]:
    """Check whether the dataset meets minimum requirements for modelling.

    Returns (True, "") if modelling can proceed,
    or (False, reason_string) if it cannot.
    """
    n = len(df)
    if n < MIN_SAMPLES_FOR_MODELLING:
        return False, (
            f"Dataset has only {n} customers (minimum required: {MIN_SAMPLES_FOR_MODELLING}). "
            "With this sample size, train/test split produces too few test observations "
            "for reliable evaluation. Any metrics would be dominated by random chance."
        )
    n_pos = int(df[TARGET].sum())
    n_neg = n - n_pos
    if n_pos < MIN_CLASS_SAMPLES or n_neg < MIN_CLASS_SAMPLES:
        return False, (
            f"Insufficient class balance: {n_pos} churned, {n_neg} retained "
            f"(minimum {MIN_CLASS_SAMPLES} per class required)."
        )
    return True, ""


def run_exploratory_workflow(df: pd.DataFrame) -> dict:
    """Run a clearly-labelled exploratory ML workflow.

    Because the dataset is too small for valid holdout evaluation, this
    function trains on ALL available data and reports training-set metrics.
    These metrics are NOT generalisation estimates — they are reported only
    to demonstrate the workflow and are labelled as such.

    Returns
    -------
    dict with keys:
        can_model (bool), limitation_reason (str), is_valid_evaluation (bool),
        models (dict of model_name → result dict), feature_importance (DataFrame)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            classification_report, confusion_matrix, roc_auc_score
        )
    except ImportError:
        return {
            "can_model": False,
            "limitation_reason": "scikit-learn is not installed.",
            "is_valid_evaluation": False,
        }

    can_model, reason = can_train_model(df)

    X, y = get_feature_matrix(df)

    existing_num = [c for c in NUMERIC_FEATURES if c in X.columns]
    existing_cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, existing_num),
        ("cat", categorical_transformer, existing_cat),
    ])

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    for name, clf in classifiers.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
        y_pred = pipe.predict(X)
        y_prob = pipe.predict_proba(X)[:, 1]

        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred).tolist()
        try:
            auc = round(roc_auc_score(y, y_prob), 3)
        except Exception:
            auc = None

        results[name] = {
            "pipeline": pipe,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc,
            "train_accuracy": round(report["accuracy"], 3),
        }

    # Feature importance from Random Forest (on training data)
    rf_pipe = results["Random Forest"]["pipeline"]
    try:
        ohe_cats = (
            rf_pipe.named_steps["pre"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(existing_cat)
        )
        feat_names = existing_num + list(ohe_cats)
        importances = rf_pipe.named_steps["clf"].feature_importances_
        fi_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        fi_df = pd.DataFrame()

    # Risk scores on the full dataset (for dashboard display)
    rf_probs = results["Random Forest"]["pipeline"].predict_proba(X)[:, 1]
    risk_df = df[["customerid"]].copy()
    risk_df["churn_probability"] = rf_probs.round(3)
    risk_df["risk_group"] = pd.cut(
        rf_probs,
        bins=[0, 0.33, 0.67, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    return {
        "can_model": can_model,
        "limitation_reason": reason if not can_model else (
            "EXPLORATORY ONLY: Model was trained and evaluated on the same 21 "
            "customers due to insufficient sample size. Reported metrics reflect "
            "in-sample fit, NOT generalisation performance. Do not use these results "
            "to make business decisions."
        ),
        "is_valid_evaluation": False,
        "models": results,
        "feature_importance": fi_df,
        "risk_scores": risk_df,
        "feature_names": existing_num + existing_cat,
    }


def load_model_results(df: pd.DataFrame) -> dict:
    """Public entry point: always returns a result dict safe for the dashboard."""
    return run_exploratory_workflow(df)
