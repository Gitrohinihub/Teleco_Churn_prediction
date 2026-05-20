# 📊 Telco Customer Churn Analysis & Prediction

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Tool](https://img.shields.io/badge/Platform-Databricks-red)
![Language](https://img.shields.io/badge/Language-Python%20%7C%20SQL-blue)
![Dataset](https://img.shields.io/badge/Dataset-IBM%20Telco%20Churn-orange)

---

## 🧩 Problem Statement

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project addresses a critical business question:

> **Which customers are most likely to leave — and why?**

With a churn rate of **26.5%** (1 in 4 customers), the telecom company is losing **~$139,000 in monthly recurring revenue** to preventable exits. Without data-driven insight, retention efforts are broad, expensive, and ineffective.

---

## ✅ How We Solved It

We followed a structured end-to-end analytics pipeline:

```
Raw CSV Data
    ↓
Data Cleaning & Preprocessing
    ↓
Exploratory Data Analysis (EDA)
    ↓
SQL Business Questions (10 queries)
    ↓
Feature Importance Analysis
    ↓
Insights Dashboard
    ↓
Retention Recommendations
```

---

## 🛠️ Tools & Technologies

| Layer | Tool |
|---|---|
| Data Platform | **Databricks** (Apache Spark) |
| Storage | **Delta Lake** |
| Language | **Python (PySpark)**, **SQL** |
| Visualization | **Databricks SQL Dashboard** |
| Dataset Source | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Version Control | **GitHub** |

---

## 📁 Dataset Overview

**Source:** IBM Sample Dataset via Kaggle  
**Rows:** 7,043 customers | **Columns:** 21 features

| Category | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Account Info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` (Yes / No) |

---

## 🧹 Data Cleaning

### What We Found & Fixed

#### 1. `TotalCharges` — Wrong Data Type
- **Problem:** Column was loaded as `string` instead of `float`
- **Root Cause:** 11 rows had blank/whitespace values (new customers with `tenure = 0` who had not yet been billed)
- **Fix:** Cast column to `float`; filled blank values with `0.0`
- **Why:** Leaving it as string would break all numeric aggregations and ML model training

```python
df = df.withColumn(
    "TotalCharges",
    F.when(F.trim(F.col("TotalCharges")) == "", F.lit(0.0))
     .otherwise(F.col("TotalCharges").cast("float"))
)
```

#### 2. `SeniorCitizen` — Integer Without Label
- **Problem:** Stored as `0` / `1` integers — unreadable in dashboards and group-by queries
- **Fix:** Added `SeniorCitizenLabel` column with `"Yes"` / `"No"` values
- **Why:** Improves readability in all reports and visualizations

```python
df = df.withColumn(
    "SeniorCitizenLabel",
    F.when(F.col("SeniorCitizen") == 1, "Yes").otherwise("No")
)
```

#### 3. `Churn` — Categorical Target Not ML-Ready
- **Problem:** `"Yes"` / `"No"` string values cannot be used in numeric aggregations or as a model target
- **Fix:** Added `ChurnBinary` column (`1` = churned, `0` = retained)
- **Why:** Required for churn rate calculations, correlation analysis, and all supervised ML models

```python
df = df.withColumn(
    "ChurnBinary",
    F.when(F.col("Churn") == "Yes", 1).otherwise(0)
)
```

#### 4. `tenure` — Continuous Variable Needs Grouping
- **Problem:** Raw months (0–72) are hard to analyze in aggregate
- **Fix:** Added `TenureGroup` bucket column: `0–12`, `13–24`, `25–48`, `49+` months
- **Why:** Enables cohort-level analysis to identify which customer lifecycle stage has highest churn

```python
df = df.withColumn(
    "TenureGroup",
    F.when(F.col("tenure") <= 12, "0–12 months")
     .when(F.col("tenure") <= 24, "13–24 months")
     .when(F.col("tenure") <= 48, "25–48 months")
     .otherwise("49+ months")
)
```

#### 5. Outlier Check — Numeric Columns
- **Checked:** `MonthlyCharges`, `TotalCharges`, `tenure`
- **Method:** IQR (Interquartile Range) ± 1.5×
- **Result:** **No outliers detected** — all values within expected telecom business ranges
- **Action:** No capping or removal needed

#### 6. Duplicates
- **Checked:** Duplicate `customerID` values
- **Result:** **0 duplicates** found
- **Action:** None required

---

### Clean Dataset Summary

| Metric | Before | After |
|---|---|---|
| Rows | 7,043 | 7,043 |
| Null values | 11 (TotalCharges) | **0** |
| Usable numeric columns | 2 | **4** |
| Duplicate rows | 0 | 0 |
| ML-ready target column | ✗ | **✓** |

---

## ❓ Business Questions & Insights

### Q1 — What is the overall churn rate?

```sql
SELECT
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned,
    ROUND(100.0 * SUM(ChurnBinary) / COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean;
```

> **Insight:** 26.5% churn rate — 1,869 out of 7,043 customers left. This is significantly above the industry benchmark of ~15–20%.

---

### Q2 — Which contract type has the highest churn?

```sql
SELECT Contract,
       COUNT(*) AS total,
       SUM(ChurnBinary) AS churned,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
```

| Contract | Churn Rate |
|---|---|
| Month-to-month | **42.7%** |
| One year | 11.3% |
| Two year | 2.8% |

> **Insight:** Month-to-month customers churn at **15× the rate** of two-year contract customers. Contract type is the single strongest predictor of churn.

---

### Q3 — Which payment method is riskiest?

```sql
SELECT PaymentMethod,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;
```

| Payment Method | Churn Rate |
|---|---|
| Electronic check | **45.3%** |
| Mailed check | 19.1% |
| Bank transfer (auto) | 16.7% |
| Credit card (auto) | 15.2% |

> **Insight:** Electronic check users churn at **3× the rate** of auto-pay users. This likely signals lower engagement and weaker commitment to the service.

---

### Q4 — Does internet service type affect churn?

```sql
SELECT InternetService,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
```

| Internet Service | Churn Rate |
|---|---|
| Fiber optic | **41.9%** |
| DSL | 19.0% |
| No internet | 7.4% |

> **Insight:** Fiber optic — the premium product — has the highest churn. This suggests price dissatisfaction or unmet performance expectations among high-value customers.

---

### Q5 — Are churned customers paying more?

```sql
SELECT Churn,
       ROUND(AVG(MonthlyCharges), 2) AS avg_monthly,
       ROUND(AVG(TotalCharges), 2) AS avg_total,
       ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM telco_churn_clean
GROUP BY Churn;
```

| Group | Avg Monthly Charge | Avg Tenure |
|---|---|---|
| Churned | **$74.44** | 18 months |
| Retained | $61.27 | 37.6 months |

> **Insight:** Churned customers pay $13/month more on average and leave much earlier. High bills with short tenure = price-sensitive customers who never found enough value.

---

### Q6 — When in the customer lifecycle does churn peak?

```sql
SELECT TenureGroup,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
GROUP BY TenureGroup
ORDER BY churn_rate_pct DESC;
```

| Tenure Group | Churn Rate |
|---|---|
| 0–12 months | **47.4%** |
| 13–24 months | 28.7% |
| 25–48 months | 20.4% |
| 49+ months | 9.5% |

> **Insight:** Nearly half of new customers churn within their first year. The first 12 months is the most critical retention window.

---

### Q7 — Are senior citizens churning more?

```sql
SELECT SeniorCitizenLabel,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
GROUP BY SeniorCitizenLabel;
```

| Senior Citizen | Churn Rate |
|---|---|
| Yes | **41.7%** |
| No | 23.6% |

> **Insight:** Senior citizens churn at nearly double the rate of non-seniors — a significantly underserved segment likely facing usability or pricing challenges.

---

### Q8 — Does tech support reduce churn for fiber users?

```sql
SELECT TechSupport,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
WHERE InternetService != 'No'
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;
```

| Tech Support | Churn Rate |
|---|---|
| No | **41.6%** |
| Yes | 15.2% |
| No internet | 7.4% |

> **Insight:** Adding tech support cuts churn by **63%** among internet customers. It's the highest-ROI retention add-on in the dataset.

---

### Q9 — How much monthly revenue is at risk?

```sql
SELECT
    ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END), 2) AS revenue_lost_monthly,
    ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END) / SUM(MonthlyCharges), 2) AS revenue_churn_pct
FROM telco_churn_clean;
```

> **Insight:** **$139,130/month** — 30.5% of total MRR — is leaving with churned customers. Annualised, that is **~$1.67 million** in preventable revenue loss.

---

### Q10 — What is the single highest-risk customer profile?

```sql
SELECT Contract, InternetService, PaymentMethod, TechSupport,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(ChurnBinary)/COUNT(*), 2) AS churn_rate_pct
FROM telco_churn_clean
WHERE Contract = 'Month-to-month' AND InternetService = 'Fiber optic'
GROUP BY PaymentMethod, TechSupport
ORDER BY churn_rate_pct DESC
LIMIT 5;
```

> **Insight:** A customer on **month-to-month + fiber optic + electronic check + no tech support** has a **63% churn probability** — the highest-risk segment in the entire dataset.

---

## 🎯 Retention Recommendations

Based on all findings, here are six data-backed retention strategies ranked by expected impact:

### 1. 📄 Offer Contract Upgrade Incentives
**Target:** All month-to-month customers (3,875 customers, 42.7% churn rate)  
**Action:** Proactively offer 10–15% discount to upgrade to a one- or two-year contract  
**Expected Impact:** Churn rate reduction from 42.7% → under 12% for converted customers

---

### 2. 💳 Migrate Electronic Check Users to Auto-Pay
**Target:** Electronic check customers (2,365 customers, 45.3% churn rate)  
**Action:** Incentivize switch to bank transfer or credit card auto-pay with a $5/month bill credit  
**Expected Impact:** Churn rate drops from 45.3% → ~16%. The incentive pays back within 1 month retained.

---

### 3. 🎧 Bundle Free Tech Support with Fiber Plans
**Target:** Fiber optic customers without tech support (1,526 customers, 41.6% churn rate)  
**Action:** Include tech support in all fiber plans at no extra charge (or nominal fee)  
**Expected Impact:** Churn reduction from 41.6% → ~15% — a 63% improvement based on the data

---

### 4. 🕐 Launch a First-Year Onboarding Program
**Target:** All customers in months 0–12 (47.4% churn rate)  
**Action:** Trigger proactive check-in calls/emails at months 3, 6, and 9. Offer a loyalty credit at month 6.  
**Expected Impact:** Even a 20% reduction in first-year churn recovers ~180 customers per cohort

---

### 5. 👴 Create a Senior Citizen Loyalty Plan
**Target:** Senior citizens (1,142 customers, 41.7% churn rate)  
**Action:** Dedicated pricing tier with simplified billing, a priority support line, and bundled tech support  
**Expected Impact:** Addressing a ~18% higher churn rate than average for this segment

---

### 6. 💰 Proactive Outreach for High-Bill Customers
**Target:** Customers paying $70+/month on month-to-month contracts  
**Action:** Have a retention team review value and offer personalised plan adjustments before they cancel  
**Expected Impact:** Reduces the $74 avg monthly charge driver — churners are paying $13/month more than retained customers

---

## 📂 Project Structure

```
telco-churn-analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Raw dataset
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb                   # Databricks cleaning notebook
│   ├── 02_eda_sql_analysis.ipynb                # SQL business questions
│   └── 03_dashboard_prep.ipynb                  # Dashboard aggregations
│
├── dashboard/
│   └── churn_dashboard_screenshot.png           # BI dashboard export
│
└── README.md                                     # This file
```

---

## 🚀 How to Run This Project

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/telco-churn-analysis.git
   ```

2. **Upload dataset to Databricks DBFS**
   - Go to **Data → Add Data → Upload File**
   - Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv`

3. **Run notebooks in order**
   - `01_data_cleaning.ipynb` → creates the Delta table `telco_churn_clean`
   - `02_eda_sql_analysis.ipynb` → runs all 10 business queries
   - `03_dashboard_prep.ipynb` → prepares aggregated views for dashboards

4. **Build dashboard** in Databricks SQL using table `telco_churn_clean`

---

## 📌 Key Takeaways

| Finding | Impact |
|---|---|
| Month-to-month contracts = 42.7% churn | Biggest single lever |
| Electronic check = 45.3% churn | Easy fix: auto-pay migration |
| First 12 months = 47.4% churn | Onboarding program needed |
| No tech support = 41.6% churn (fiber) | Bundle it in the plan |
| Senior citizens = 41.7% churn | Dedicated loyalty tier |
| $139K MRR at risk monthly | ~$1.67M annualised loss |

---

## 👤 Author

**Rohini Singh**  
Data & BI Analyst  
[LinkedIn](https://linkedin.com)

---

*Dataset sourced from IBM Sample Data Sets via Kaggle. Used for educational and analytical purposes.*
