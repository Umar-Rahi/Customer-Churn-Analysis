# Customer Churn Analysis & Prediction
## AI-Powered Customer Retention Analytics Dashboard

**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**  
**Author:** Umar Rahi  
**GitHub:** [https://github.com/Umar-Rahi/Customer-Churn-Analysis](https://github.com/Umar-Rahi/Customer-Churn-Analysis)

> 🚀 **Live Demo:** [Open Customer Churn Analysis Dashboard](https://customer-churn-analysis-nmrb.onrender.com)

---

## Overview

This project delivers a professional, interactive **Customer Churn Analytics Dashboard** built with Python and Streamlit. It analyses a real subscription-service dataset (21 customers, 3 worksheets) to:

1. Identify which customers churned and surface documented cancellation reasons
2. Break down churn patterns by contract type, plan type, acquisition channel, and geography
3. Compare customer support metrics (CSAT, escalation rate) between churned and retained customers
4. Demonstrate a complete machine-learning predictive modelling pipeline (with transparent limitations)
5. Deliver actionable, data-backed retention strategy recommendations

The dashboard is designed for a professional Data Analyst portfolio and is ready for one-click deployment on [Render](https://render.com).

---

## Features

| Feature | Detail |
|---|---|
| **Executive Overview** | KPI cards (total / churned / retained customers, churn rate, monthly revenue, CLTV at risk), donut chart, bar comparisons |
| **Churn Analysis** | Churn rate by contract type, plan type, subscription channel, state; cancellation reason pie chart; tenure histogram; tenure-bucket table |
| **Customer Insights** | Monthly charges boxplot, CLTV boxplot, scatter (charges × CLTV × tenure), support ticket count per customer, CSAT distribution, escalation analysis |
| **Prediction & Risk** | Three-model comparison (Logistic Regression, Decision Tree, Random Forest), feature importance chart, per-customer risk scores (Low / Medium / High), pre-existing churn score reference |
| **Recommendations** | Observed churn drivers, retention strategies by segment, business impact discussion, limitations table |
| **Sidebar Filters** | Contract type, plan type, subscription type, churn status, state, monthly-charges range — all charts and KPIs respond dynamically |
| **Demo Mode** | Automatic fallback to clearly-labelled synthetic data when the private workbook is unavailable (e.g. on Render) |
| **Privacy** | Customer names excluded from all dashboard outputs; only anonymised IDs displayed |
| **Honest ML Evaluation** | In-sample metrics clearly labelled; explicit warnings against treating them as generalisation performance |

---

## Technology Stack

| Category | Library | Notes |
|---|---|---|
| Data manipulation | `pandas` ≥ 2.0 | Supports pandas 3.x |
| Numerical | `numpy` ≥ 1.24 | Supports numpy 2.x |
| Excel reading | `openpyxl` ≥ 3.1 | Required for `.xlsx` workbooks |
| Visualisation | `matplotlib`, `seaborn` | Used in Jupyter notebook |
| Dashboard charts | `plotly` ≥ 5.18 | Supports plotly 7.x |
| Dashboard framework | `streamlit` ≥ 1.30 | Tested on 1.66.x |
| Machine learning | `scikit-learn` ≥ 1.3 | Tested on 1.8.x |
| Notebooks | `jupyter`, `ipykernel` | Local development only |

Tested on **Python 3.14.2**. Python 3.10+ required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         app.py                               │
│           Streamlit Dashboard (5 tabbed sections)            │
│                                                              │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ src/         │  │ src/       │  │ src/                 │  │
│  │ data_        │  │ analysis   │  │ model.py             │  │
│  │ processing   │  │ .py        │  │ (ML pipeline)        │  │
│  │ .py          │  │ (KPIs)     │  │                      │  │
│  └──────┬───────┘  └─────┬──────┘  └────────────┬─────────┘  │
│         │                │                      │            │
│         └────────────────┴──────────────────────┘            │
│                          │                                   │
│              data/customer_churn_data_raw.xlsx               │
│              (or auto-generated synthetic demo data)         │
└──────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. `data_processing.py` loads the Excel workbook → cleans three sheets → merges to one master DataFrame
2. `analysis.py` computes KPIs, churn breakdowns, and support summaries from the master DataFrame
3. `model.py` runs an exploratory ML workflow (preprocessing → training → in-sample evaluation → risk scoring)
4. `app.py` renders all results interactively via Streamlit

---

## Project Structure

```
Customer-Churn-Analysis-main/
├── app.py                                      # Streamlit dashboard (entry point)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── .gitignore                                  # Excludes private data and caches
├── .streamlit/
│   └── config.toml                             # Streamlit theme & server config
├── src/
│   ├── __init__.py
│   ├── data_processing.py                      # Load, clean, merge pipeline
│   ├── analysis.py                             # KPI and EDA calculation functions
│   └── model.py                                # ML workflow utilities
├── data/
│   ├── README.md                               # Dataset placement instructions
│   └── customer_churn_data_raw.xlsx            # ← Place private workbook here
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py                 # 22 unit tests
│   └── test_analysis.py                        # 16 unit tests
├── outputs/
│   └── figures/                                # Saved chart PNGs from notebook
├── UmarRahi_CustomerChurnAnalysis.ipynb        # Full analysis notebook
└── UmarRahi_CustomerChurnAnalysis_ProjectReport.docx  # Written report
```

---

## Local Installation

### Prerequisites

- Python 3.10 or higher
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/Umar-Rahi/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis-main
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset (optional — demo mode available without it)

Copy the private Excel workbook to:
```
data/customer_churn_data_raw.xlsx
```

See [`data/README.md`](data/README.md) for the full data dictionary and column descriptions.

> If the file is absent, the dashboard automatically switches to **Demo Mode** — a clearly-labelled synthetic dataset is generated so the application remains fully functional.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

### 6. Run the tests

```bash
python -m pytest tests/ -v
```

Expected output: **38 passed** (no failures, no warnings suppressed).

### 7. Run the notebook (optional)

```bash
jupyter notebook UmarRahi_CustomerChurnAnalysis.ipynb
```

Kernel → Restart & Run All.

---

## Dataset

| Property | Value |
|---|---|
| File | `customer_churn_data_raw.xlsx` |
| Sheets | `db_customer`, `db_subscription`, `db_support` |
| Total customers | 21 |
| Churned customers | 6 |
| Churn rate | 28.6% |
| Support tickets | 9 (some customers have multiple) |

**The dataset is private** — it contains customer personal information and is not included in the repository. The `.gitignore` file excludes all `data/*.xlsx` files.

To obtain the dataset, contact the project author.

### Churn Target Definition

A customer is labelled **Churned (1)** if `cancellation_date IS NOT NULL` in `db_subscription`.

### Leakage Prevention

The following columns are excluded from all predictive features:

| Column | Reason |
|---|---|
| `cancellation_date` | Post-cancellation fact |
| `cancellation_reason` | Post-cancellation fact |
| `churn_score` | Unknown provenance — may be post-cancellation |
| `renewal_date` | Tied to subscription lifecycle event |

---

## Analysis Methodology

### Data Pipeline (`src/data_processing.py`)

1. **Load:** Read all three worksheets from `.xlsx` via `openpyxl`
2. **Clean `db_customer`:** Normalise column names, drop blank rows, fix gender labels (`Men`→`Male`, `Women`→`Female`), infer missing country from state, derive age and age group
3. **Clean `db_subscription`:** Parse date columns, fix known typo `Refferal`→`Referral`, define churn target, compute tenure in months from subscription start to `2025-01-01`
4. **Clean `db_support`:** Drop empty `col_1` column, map escalation flag `Y/N`→`True/False`, parse CSAT as numeric
5. **Aggregate support:** Collapse multiple tickets per customer to complaint count, escalation flag, avg CSAT, min CSAT
6. **Merge:** LEFT join `db_subscription` ← `db_customer` ← `support_agg` on `customerid`; customers with no tickets get `complaint_count=0`, `has_escalation=False`

### Machine Learning (`src/model.py`)

**Preprocessing pipeline (scikit-learn):**
- Numeric features (`monthly_charges`, `cltv`, `tenure_months`, `complaint_count`): median imputation → StandardScaler
- Categorical features (`subscription_type`, `plan_type`, `contract_type`, `gender`, `age_group`): most-frequent imputation → OneHotEncoder

**Models trained:** Logistic Regression, Decision Tree (max_depth=3), Random Forest (50 estimators, max_depth=3), all with `class_weight="balanced"`.

**Reference date:** `2025-01-01` (fixed, reproducible)

---

## Key Findings (Real Dataset)

| Finding | Value |
|---|---|
| Total customers | 21 |
| Churned customers | 6 |
| Overall churn rate | 28.6% |
| Monthly contract churn rate | 54.5% |
| Annual contract churn rate | 0.0% |
| Avg CLTV — Churned | ~$341 |
| Avg CLTV — Retained | ~$1,016 |
| Most common cancellation reason | Switched to competitor (2 of 6) |
| Escalation→churn link | 4 of 6 churned customers had an escalated ticket |

---

## Machine Learning Limitations

⚠️ **The 21-customer sample is far too small for valid holdout ML evaluation.**

- A standard 80/20 stratified split produces only ~4 test observations
- Any precision, recall, or AUC from such a split has extreme statistical variance and is not meaningful
- All ML results in this project are **in-sample (exploratory only)** — trained and evaluated on the same 21 records
- Reported metrics demonstrate the workflow, not real predictive power
- Minimum recommended sample for reliable evaluation: **500–1,000+ customers**

**Do not use these models to make real business decisions without a substantially larger dataset.**

---

## Privacy & Responsible Use

- Customer **names** appear in the raw dataset but are **never displayed** in any analysis, chart, or dashboard output
- Only anonymised **Customer IDs** are shown in outputs
- The workbook `customer_churn_data_raw.xlsx` must **not** be committed to a public repository — it is excluded by `.gitignore`
- All findings are correlational observations within this small cohort; no causal claims are made
- Do not share the workbook without explicit permission from the data owner

---

## Render Deployment

This application is designed to be deployed as a **Streamlit Web Service on [Render](https://render.com)**.

### Configuration

| Setting | Value |
|---|---|
| **Environment** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true` |
| **Root Directory** | _(project root — where `app.py` is located)_ |

### Important Notes

1. **The private dataset** (`customer_churn_data_raw.xlsx`) is **not in the repository**. When deployed on Render without the file, the app automatically activates **Demo Mode** — a clearly-labelled synthetic dataset is generated and the application remains fully functional.

2. **No `PORT` env var configuration needed** — Render injects `$PORT` automatically.

3. **No Flask / Gunicorn needed** — Streamlit has its own server.

4. **No separate frontend** — the Streamlit app is the complete UI.

### Steps

1. Push this repository to GitHub (without `data/*.xlsx`).
2. Create a new **Web Service** on Render.
3. Connect your GitHub repository.
4. Set the Build Command and Start Command as above.
5. Deploy. The app starts in Demo Mode until you provide the dataset.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: customer_churn_data_raw.xlsx` | Place the workbook at `data/customer_churn_data_raw.xlsx` and restart. Demo mode activates automatically if the file is absent. |
| `Missing sheet: db_customer` / `db_subscription` / `db_support` | Ensure the workbook has exactly these three sheet names (case-sensitive). |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `ModuleNotFoundError: No module named 'openpyxl'` | Run `pip install openpyxl` |
| Streamlit sidebar shows numpy scalar TypeError | Upgrade to numpy ≥ 2.0 or ensure `float()` wrapping is applied — already handled in `app.py` |
| Tests fail with `ValueError: cancellation_reasons columns` | Update to the fixed `analysis.py` (pandas 3.x compatible `value_counts` renaming) |
| Charts don't render | Ensure `plotly` ≥ 5.18 is installed — `pip install plotly` |
| Streamlit doesn't start on Render | Verify the Start Command includes `--server.address 0.0.0.0 --server.port $PORT` |

---

## Screenshots

_Screenshots will be added after deployment._

---

## Author

**Umar Rahi**  
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 — BharatCares  
GitHub: [https://github.com/Umar-Rahi](https://github.com/Umar-Rahi)

---

## License

- Source code by **Umar Rahi** — available for educational and portfolio use.
- The dataset (`customer_churn_data_raw.xlsx`) is **not licensed for public redistribution** and is not included in the repository.
- Third-party libraries are subject to their respective open-source licenses.

---

*AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 — BharatCares*
