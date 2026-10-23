# data/README.md — Dataset Placement Instructions

## Required Dataset

This project requires the following Excel workbook:

```
data/customer_churn_data_raw.xlsx
```

### How to Obtain the File

The dataset is **not included in the repository** because it is not licensed for public redistribution and contains personal information.

To reproduce this project:
1. Obtain the file `customer_churn_data_raw.xlsx` from the project owner (Umar Rahi).
2. Place it in this `data/` folder with the exact filename: `customer_churn_data_raw.xlsx`.
3. The notebook and dashboard reference it at the relative path `data/customer_churn_data_raw.xlsx`.

### Worksheet Structure

| Sheet | Rows | Description |
|---|---|---|
| `db_customer` | 21 | Customer ID, name, country, state, gender, DOB, interests, pincode |
| `db_subscription` | 21 | subscription_start_date, subscription_type, renewal_date, plan_type, contract_type, cancellation_date, cancellation_reason, monthly_charges, cltv, churn_score |
| `db_support` | 9 | customerid, complaint_date, escalations (Y/N), csat_score, col_1 (empty), comment |

### Privacy Note

Customer names appear in the raw data but are not displayed in any analysis, chart, or dashboard.
Only anonymized customer IDs are used in outputs.

### Data Dictionary

**db_customer**
| Column | Type | Description |
|---|---|---|
| customerid | String | Unique customer identifier (e.g., 0002-ORFBO) |
| name | String | Customer first name (not used in analysis) |
| country | String | Country of residence (some missing) |
| state | String | State or region |
| gender | String | Gender (cleaned to Male/Female) |
| dob | Date | Date of birth |
| interests | String | Stated interest category |
| pincode | String | Postal code (mostly missing) |

**db_subscription**
| Column | Type | Description |
|---|---|---|
| customerid | String | Foreign key to db_customer |
| subscription_start_date | Date | When subscription began |
| subscription_type | String | Organic / Paid / Referral |
| renewal_date | Date | Scheduled renewal date |
| plan_type | String | Basic / Standard / Premium |
| contract_type | String | Monthly / Annual |
| cancellation_date | Date | Date of cancellation (null = retained) |
| cancellation_reason | String | Stated reason (null = retained) |
| monthly_charges | Float | Monthly fee in USD |
| cltv | Integer | Customer Lifetime Value (provenance unspecified) |
| churn_score | Integer | Pre-existing churn risk score (0–100, provenance unknown) |

**db_support**
| Column | Type | Description |
|---|---|---|
| customerid | String | Foreign key to db_subscription |
| complaint_date | Date | Date ticket was opened |
| escalations | String | Y = escalated, N = not escalated |
| csat_score | Integer | Customer satisfaction score (0–100) |
| col_1 | — | Empty column (dropped during cleaning) |
| comment | String | Free-text support note |
