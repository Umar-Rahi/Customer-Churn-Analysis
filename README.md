# Customer Churn Analysis and Prediction
## An AI-Powered Customer Retention Dashboard

**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**  
**Author:** Umar Rahi | **BharatCares**  
**GitHub:** [https://github.com/Umar-Rahi/Customer-Churn-Analysis](https://github.com/Umar-Rahi/Customer-Churn-Analysis)
**Dataset Link:** [https://docs.google.com/spreadsheets/d/1POBGa8tgfELvQIrz-ca_tOEhQPI_8BbT/edit?usp=sharing&ouid=105897510512459950654&rtpof=true&sd=true](Access_the_Customer_Churn_Dataset)

---

## Business Problem & Objectives

Customer churn — the loss of subscribers who cancel their service — directly reduces revenue and increases customer acquisition costs. This project analyzes a subscription service dataset to:

1. Identify which customers churned and why
2. Discover patterns and risk factors associated with churn
3. Demonstrate a full predictive modelling workflow (with honest limitations)
4. Deliver actionable retention strategy recommendations
5. Present findings in an interactive, professional dashboard

---

## Dataset Description

| File | Location |
|---|---|
| `customer_churn_data_raw.xlsx` | `data/customer_churn_data_raw.xlsx` |

The dataset is provided locally and is **not publicly available**. To reproduce this project, obtain the workbook from the project owner and place it at `data/customer_churn_data_raw.xlsx`.

### Worksheets

| Sheet | Rows | Columns | Description |
|---|---|---|---|
| `db_customer` | 21 (22 incl. blank) | 8 | Customer demographics: ID, name, country, state, gender, DOB, interests, pincode |
| `db_subscription` | 21 | 11 | Subscription details: start/renewal/cancellation dates, type, plan, contract, charges, CLTV, churn score |
| `db_support` | 9 | 6 | Support tickets: complaint date, escalation flag, CSAT score, comment |

### Key Dataset Facts
- **21 unique customers** across all tables
- **6 churned customers** (cancellation_date present) → **28.6% churn rate**
- **15 retained customers**
- `churn_score` column present but provenance unknown — **excluded from ML features** to prevent leakage

---

## Technologies & Libraries

| Category | Library | Version |
|---|---|---|
| Data manipulation | pandas | ≥2.0 |
| Numerical | numpy | ≥1.24 |
| Excel reading | openpyxl | ≥3.1 |
| Visualization | matplotlib, seaborn | ≥3.7, ≥0.12 |
| Dashboard charts | plotly | ≥5.18 |
| Dashboard framework | streamlit | ≥1.30 |
| Machine learning | scikit-learn | ≥1.3 |
| Notebooks | jupyter, ipykernel | ≥1.0, ≥6.0 |

---

## Project Folder Structure

```
Customer-Churn-Analysis/
├── UmarRahi_CustomerChurnAnalysis.ipynb        # Full analysis notebook
├── UmarRahi_CustomerChurnAnalysis_ProjectReport.docx  # Word report
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── app.py                                      # Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data_processing.py                      # Data loading, cleaning, merging
│   ├── analysis.py                             # KPI and EDA calculations
│   └── model.py                                # ML workflow utilities
├── data/
│   ├── README.md                               # Dataset placement instructions
│   └── customer_churn_data_raw.xlsx            # ← Place dataset here (not in repo)
├── tests/
│   ├── test_data_processing.py
│   └── test_analysis.py
├── outputs/
│   └── figures/                                # Saved chart PNGs from notebook
├── .gitignore
└── .streamlit/
    └── config.toml                             # Dashboard theme config
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Umar-Rahi/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

### 2. Set up a virtual environment

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

### 4. Place the dataset

Copy the Excel workbook to:
```
data/customer_churn_data_raw.xlsx
```

See `data/README.md` for details.

---

## Running the Project

### Launch Jupyter Notebook

```bash
jupyter notebook UmarRahi_CustomerChurnAnalysis.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`) in a clean kernel.

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

---

## Dashboard Features

| Tab | Contents |
|---|---|
| 🏠 Overview | KPI cards, churn donut chart, CLTV comparison, customer data table |
| 📉 Churn Analysis | Churn by contract/plan/subscription type, state breakdown, cancellation reasons, tenure histogram |
| 👥 Customer Insights | Monthly charges & CLTV boxplots, support ticket analysis, escalation chart, scatter plot |
| 🤖 Prediction / Risk | Exploratory ML model comparison, feature importance, risk score table, existing churn score reference |
| 💡 Recommendations | Observed churn drivers, retention strategies by segment, limitations table |

**Sidebar filters:** Contract type, plan type, subscription type, churn status, state, monthly charges range. All charts and KPIs respond to filters.

---

## Methodology & Churn Target Definition

**Churn is defined as:** `cancellation_date IS NOT NULL` in `db_subscription`.

**Leakage prevention:**
- `cancellation_date`, `cancellation_reason` — post-cancellation facts, not used as features
- `churn_score` — unknown provenance, excluded from ML features
- `renewal_date` — tied to subscription lifecycle

**Preprocessing pipeline (scikit-learn):**
- Numeric: median imputation → standard scaling
- Categorical: most-frequent imputation → one-hot encoding
- Fitted on training data only (in exploratory mode: on full dataset)

**Reference date for tenure calculation:** `2025-01-01`

---

## Key Findings (from the dataset)

| Finding | Value |
|---|---|
| Total customers | 21 |
| Churned customers | 6 |
| Churn rate | 28.6% |
| Monthly contract churn rate | 54.5% |
| Annual contract churn rate | 0.0% |
| Avg CLTV — Churned | $162.8 |
| Avg CLTV — Retained | $964.7 |
| Customers with escalated tickets who churned | 4 of 6 (67%) |
| Most common cancellation reason | Switched to competitor (2) |

---

## Machine Learning Evaluation Limitations

⚠️ **The 21-customer sample is too small for valid holdout ML evaluation.**

- A standard 80/20 split produces only ~4 test observations
- Any precision, recall, or AUC from such a split has extreme statistical variance
- All ML results in this project are **in-sample (exploratory only)**
- Minimum recommended sample: 500–1000 customers

**Do not use the model output for real business decisions without a larger dataset.**

---

## Privacy & Responsible Use

- Customer names appear in the raw dataset but are **not displayed** in analyses, charts, or the dashboard
- Customer IDs (anonymized codes) are used throughout
- This dataset should not be committed to a public repository
- The workbook is provided locally — do not share it without the owner's permission
- All findings are correlational within this small cohort; no causal claims are made

---

## GitHub Setup Instructions

```bash
# Initialize git (if not cloned)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Customer Churn Analysis project"

# Add remote (replace with your GitHub username)
git remote add origin https://github.com/Umar-Rahi/Customer-Churn-Analysis.git

# Push
git push -u origin main
```

**Note:** Do not push `data/customer_churn_data_raw.xlsx` to a public repository. It is excluded by `.gitignore`.

---

## License

This project was created for academic and internship purposes.

- Source code is authored by **Umar Rahi** and is available for educational use.
- The dataset (`customer_churn_data_raw.xlsx`) is **not licensed for public redistribution** and is not included in the repository.
- Third-party libraries used are subject to their respective open-source licenses.

---

*AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 — BharatCares*
