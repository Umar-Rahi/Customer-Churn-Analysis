# Customer Churn Analysis & Intelligence

## Overview
This project analyzes a small OTT-style subscription dataset to understand customer churn, subscription patterns, cancellation reasons, customer lifetime value (CLTV), and customer-support signals.

## Included files
- `churn_analysis(2).ipynb` — original analysis notebook supplied with the project.
- `customer_churn(2).db` — SQLite database containing `db_customer`, `db_subscription`, and `db_support`.
- `customer_churn_data_raw(2).xlsx` — source workbook with the relational tables.
- `exported_churn_data.csv` — exported/flattened customer-level data.
- `requirements.txt` — Python dependencies.
- `Churn_Analysis_30_Page_Report.pdf` — project report.
- `churn_dashboard.html` — standalone interactive dashboard.

## Dataset structure
- `db_customer`: customer demographics and location.
- `db_subscription`: subscription, plan, contract, cancellation, charges, CLTV, and churn score.
- `db_support`: complaint dates, escalation indicators, CSAT scores, and comments.

The customer table has 21 rows, the subscription table has 21 rows, and the support table has 9 records in the supplied SQLite database. Support has multiple rows for some customers, so it should be aggregated before joining to a one-row-per-customer table to avoid duplicate customer counts.

## Key measured figures
These figures are computed directly from the supplied SQLite data using a cancellation date as the churn indicator:
- Customers: 21
- Churned customers: 6
- Churn rate: 28.6%
- Retention rate: 71.4%
- Sum of monthly charges for churned records (monthly-charge proxy for lost MRR): ₹73.94
- Mean monthly charge per customer record: ₹18.85
- Mean CLTV: ₹823.52

The data is a small sample and should not be treated as representative of a real production customer base. The source training PDF reports a separate set of headline figures; this README's metrics are recalculated from the attached database and may differ because of definitions or dataset version.

## Requirements
- Python 3.10 or later recommended
- Jupyter Notebook (optional, for notebook execution)
- Power BI Desktop (optional, if you want to rebuild the dashboard in Power BI)

## Setup
1. Create and activate a virtual environment.
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install packages.
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Put the project files in one folder. Keep the database and notebook in the same folder unless you update the notebook's database path.
4. Start Jupyter if needed.
   ```powershell
   jupyter notebook
   ```
5. Open the notebook and run cells from top to bottom. If a cell expects a variable created by a prior cell, run the earlier setup/import cells first.

## Dashboard
Open `churn_dashboard.html` in a modern browser. It includes filters for plan, contract, churn status, and state, plus KPI cards and charts. It uses inline sample records embedded from the supplied database; no server or package installation is needed to view it.

## KPI definitions used
- **Churn rate** = customers with a non-empty cancellation date / total customer subscription records.
- **Retention rate** = 1 − churn rate.
- **Revenue at risk (simple proxy)** = monthly charges for active customers whose churn score is greater than 70.
- **CLTV** = supplied `cltv` field; its currency/unit is not specified by the source data.
- **Support escalation** = support rows where `escalations` equals `Y` (case-insensitive).

## Limitations
- Only 21 customer/subscription records and 9 support events are present.
- The dataset does not document the currency/unit for CLTV or monthly charges; ₹ formatting in the dashboard is a display convention for this deliverable and should be confirmed before business use.
- Churn score is treated as a supplied score, not a validated probability.
- Correlation between support events and churn is descriptive only and does not prove causation.
- Some customer attributes and support fields are missing.

## Suggested next steps
- Validate definitions and currency with the dataset owner.
- Add refreshable source data and scheduled refresh if deploying.
- Confirm table relationships using `customerid`.
- Review high-score active customers and cancellation reasons before launching retention campaigns.
