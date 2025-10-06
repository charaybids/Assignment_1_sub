# Duplicate Customer_ID Analysis Report

## Executive Summary

**Finding:** There are **NO duplicate Customer_IDs** in the financials or attributes datasets. Each dataset follows a specific data pattern based on its purpose.

---

## Detailed Analysis by Dataset

### 1. Financials Dataset

**Data Pattern: SNAPSHOT (One per Customer)**

| Metric | Value |
|--------|-------|
| Total records | 12,500 |
| Unique Customer_IDs | 12,500 |
| Duplicate records | **0** |
| Max records per customer | **1** |

**Timestamp Analysis:**
- Has `snapshot_date` column: ✅ YES
- Unique dates: 25 dates
- Date range: **1/1/2023 to 1/9/2024**

**Conclusion:**
- ✅ **NO duplicates** - Each customer has exactly ONE record
- Pattern: **SNAPSHOT** (latest financial state per customer)
- Data structure: **One-to-one** (1 customer = 1 record)
- Update method: **OVERWRITE** (if updated, old record replaced with new one)

---

### 2. Attributes Dataset

**Data Pattern: SNAPSHOT (One per Customer)**

| Metric | Value |
|--------|-------|
| Total records | 12,500 |
| Unique Customer_IDs | 12,500 |
| Duplicate records | **0** |
| Max records per customer | **1** |

**Timestamp Analysis:**
- Has `snapshot_date` column: ✅ YES
- Unique dates: 25 dates
- Date range: **1/1/2023 to 1/9/2024**

**Conclusion:**
- ✅ **NO duplicates** - Each customer has exactly ONE record
- Pattern: **SNAPSHOT** (latest attribute state per customer)
- Data structure: **One-to-one** (1 customer = 1 record)
- Update method: **OVERWRITE** (if updated, old record replaced with new one)

---

### 3. Loan Daily Dataset

**Data Pattern: TIME-SERIES (APPEND by Design)**

| Metric | Value |
|--------|-------|
| Total records | 137,500 |
| Unique Customer_IDs | 12,500 |
| Average records per customer | **11.0** |
| Multiple records per customer | ✅ YES (by design) |

**Sample Customer Records (CUS_0x1000):**

| snapshot_date | installment_num | overdue_amt |
|---------------|-----------------|-------------|
| 1/5/2023 | 0 | 0 |
| 1/6/2023 | 1 | 0 |
| 1/7/2023 | 2 | 0 |
| 1/8/2023 | 3 | 1000 |
| 1/10/2023 | 5 | 1000 |
| 1/11/2023 | 6 | 2000 |
| 1/12/2023 | 7 | 3000 |
| 1/1/2024 | 8 | 4000 |
| 1/2/2024 | 9 | 5000 |
| 1/3/2024 | 10 | 6000 |

**Pattern Analysis:**
- Multiple records: ✅ **YES** (this is expected and correct)
- Each record represents a different point in time (snapshot_date)
- installment_num progresses: 0 → 1 → 2 → ... → 10
- overdue_amt changes over time (0 → 1000 → 2000 → ...)

**Conclusion:**
- ✅ **Multiple records per customer are CORRECT**
- Pattern: **TIME-SERIES** (loan payment history over time)
- Data structure: **One-to-many** (1 customer = many records)
- Update method: **APPEND** (new records added for each snapshot_date)
- This tracks loan performance across installments (MOB 0 through MOB 12)

---

### 4. Clickstream Dataset

**Data Pattern: TIME-SERIES (APPEND by Design)**

| Metric | Value |
|--------|-------|
| Total records | 215,376 |
| Unique Customer_IDs | 8,974 |
| Average records per customer | **24.0** |
| Multiple records per customer | ✅ YES (by design) |

**Sample Customer Records (CUS_0x1037):**

| snapshot_date | fe_1 | fe_10 |
|---------------|------|-------|
| 1/1/2023 | 63 | 83 |
| 1/2/2023 | 55 | 113 |
| 1/10/2023 | 168 | 56 |
| 1/11/2023 | 63 | 395 |
| 1/12/2023 | 146 | 121 |
| 1/1/2024 | 239 | 210 |
| 1/2/2024 | 114 | 45 |
| 1/10/2024 | 199 | 317 |
| 1/11/2024 | 287 | 48 |
| 1/12/2024 | 145 | 24 |

**Pattern Analysis:**
- Multiple records: ✅ **YES** (this is expected and correct)
- Each record represents customer behavior at different snapshot_date
- Feature values (fe_1, fe_10, etc.) change over time
- Monthly snapshots from 1/1/2023 to 1/12/2024

**Conclusion:**
- ✅ **Multiple records per customer are CORRECT**
- Pattern: **TIME-SERIES** (clickstream behavior over time)
- Data structure: **One-to-many** (1 customer = many records)
- Update method: **APPEND** (new clickstream snapshots added monthly)
- Captures changing customer behavior patterns before loan application

---

## Why "Duplicate Customer_ID" in README?

The README mentions "Duplicate Customer_ID in financials or attributes" as a **flagging criterion**, but this does **NOT mean duplicates exist** in the raw data. 

### Clarification:

**The flagging logic checks:**
```python
# Pseudo-code from silver_utils.py
if customer appears multiple times in financials:
    flag as "bad quality"
elif customer appears multiple times in attributes:
    flag as "bad quality"
```

**Purpose:** This is a **DATA QUALITY CHECK** to ensure:
- Financials has exactly 1 record per customer
- Attributes has exactly 1 record per customer
- Any violations would indicate data corruption

**Current Status:**
- ✅ Financials: **PASSED** (no duplicates, 12,500 unique)
- ✅ Attributes: **PASSED** (no duplicates, 12,500 unique)
- ✅ No customers flagged for this reason

---

## Data Update Patterns Summary

| Dataset | Pattern | Records/Customer | Update Method | Purpose |
|---------|---------|------------------|---------------|---------|
| **Financials** | SNAPSHOT | 1 | **OVERWRITE** | Latest financial state |
| **Attributes** | SNAPSHOT | 1 | **OVERWRITE** | Latest customer attributes |
| **Loan Daily** | TIME-SERIES | 11 avg | **APPEND** | Loan payment history |
| **Clickstream** | TIME-SERIES | 24 avg | **APPEND** | Behavior over time |

---

## Timestamp Analysis

### Financials & Attributes (OVERWRITE Pattern)

**Why only 1 record per customer despite having snapshot_date?**

The datasets have 25 different snapshot_dates (1/1/2023 to 1/9/2024), but each customer appears at only ONE of these dates. This indicates:

1. **Data collection methodology:**
   - Different customers joined at different times
   - Each customer's record represents their state at the time they joined
   - OR: Latest known state across different collection windows

2. **Update strategy (if re-collected):**
   - **OVERWRITE:** If a customer's data is updated, the old record is replaced
   - Result: Always 1 record per customer (the latest)

3. **Benefit of snapshot_date:**
   - Tracks when data was collected
   - Enables time-aware analysis (don't use future information)
   - Essential for preventing data leakage in ML models

### Loan Daily & Clickstream (APPEND Pattern)

**Why multiple records per customer?**

These are **time-series** datasets by design:

1. **Loan Daily:**
   - Tracks loan performance over 12 months (11 installments on average)
   - Each record = state at a specific point in time
   - New records APPENDED for each installment

2. **Clickstream:**
   - Tracks customer behavior over 24 months
   - Each record = clickstream snapshot at a specific date
   - New records APPENDED for each month

---

## Implications for Data Pipeline

### Silver Layer Processing

**Current approach is CORRECT:**

1. **Financials/Attributes:**
   - Expect 1 record per customer ✅
   - Flag any duplicates as data quality issue ✅
   - No duplicates found = no flagging needed ✅

2. **Loan Daily/Clickstream:**
   - Expect multiple records per customer ✅
   - Do NOT flag as duplicates ✅
   - Use time-aware filtering in Gold layer ✅

### Gold Layer Feature Engineering

**Time-aware filtering prevents data leakage:**

```python
# Correct approach (already implemented)
# For prediction at MOB=0, use only data BEFORE or AT application

# Financials: Use snapshot at or before loan_start_date
financials = financials.filter(
    F.col('snapshot_date') <= F.col('loan_start_date')
)

# Clickstream: Use only data BEFORE loan application
clickstream = clickstream.filter(
    F.col('snapshot_date') < F.col('loan_start_date')
)

# Loan Daily: Use MOB=0 snapshot only (application time)
loan_features = loan_daily.filter(F.col('MOB') == 0)
```

---

## Recommendations

### ✅ Current Implementation is Correct

1. **No duplicates in Financials/Attributes** - Working as designed
2. **Multiple records in Loan Daily/Clickstream** - Correct time-series structure
3. **Flagging logic** - Appropriate quality checks
4. **Time-aware filtering** - Prevents data leakage

### 📝 Documentation Update

The README should clarify:

**Before (Potentially Confusing):**
> "2. **Duplicate Customer_ID** in financials or attributes"

**Suggested Clarification:**
> "2. **Duplicate Customer_ID detection** in financials or attributes (Note: No duplicates found in current data - this check ensures data quality by flagging any customer appearing more than once in snapshot datasets)"

### 🔍 Optional Future Enhancements

1. **Track data freshness:**
   - Add logic to use latest snapshot_date if multiple exist
   - Currently not needed (no duplicates)

2. **Temporal feature engineering:**
   - For Loan Daily: Aggregate payment patterns from MOB 0 to prediction_mob
   - For Clickstream: Time-weighted aggregations (recent behavior vs. historical)

3. **Data versioning:**
   - If data gets updated frequently, implement versioning
   - Track which snapshot_date each model was trained on

---

## Conclusion

**Key Findings:**

1. ✅ **NO duplicate Customer_IDs** in Financials dataset (12,500 unique)
2. ✅ **NO duplicate Customer_IDs** in Attributes dataset (12,500 unique)
3. ✅ **Multiple records per customer are CORRECT** in Loan Daily (time-series)
4. ✅ **Multiple records per customer are CORRECT** in Clickstream (time-series)

**Data Patterns:**
- **SNAPSHOT datasets** (Financials, Attributes): **OVERWRITE** pattern
  - 1 record per customer
  - snapshot_date tracks when data was collected
  - If updated, old records are replaced

- **TIME-SERIES datasets** (Loan Daily, Clickstream): **APPEND** pattern
  - Multiple records per customer (by design)
  - Each record represents a point in time
  - New records added over time

**Pipeline Status:**
- ✅ All data quality checks are working correctly
- ✅ No false positives from duplicate detection
- ✅ Time-aware filtering implemented properly
- ✅ No data leakage risks from duplicate handling

---

**Analysis Date:** October 6, 2025  
**Analyst:** AI Assistant  
**Data Version:** As of current workspace state
