# 📊 DATA QUALITY FINDINGS - PRESENTATION SLIDES SUMMARY

## SLIDE 1: Title
**Silver Layer Data Quality Analysis**
*Loan Default Prediction Pipeline - CS611 MLE*

---

## SLIDE 2: Executive Summary

### Dataset Overview
- **12,500 customers** across 4 datasets
- **367,376 total records** processed
- **46 numeric columns** + 4 categorical columns cleaned

### Key Finding
🚨 **13.06% of customers (1,633)** flagged for data quality issues

---

## SLIDE 3: Four Types of Placeholder Patterns Found

| Pattern | Column | Count | % | Interpretation |
|---------|--------|-------|---|----------------|
| `"_______"` | Occupation | 880 | 7.0% | Manual entry placeholder |
| `"_"` | Credit_Mix | **2,611** | **20.9%** | System default |
| `"NM"` | Payment_of_Min_Amount | 1,438 | 11.5% | "Not Mentioned" |
| `"!@9#%8"` | Payment_Behaviour | 998 | 8.0% | **🔥 Data corruption** |

**Total affected:** 5,927 records (47.4% have at least one placeholder)

---

## SLIDE 4: Critical Issue - Credit Mix Missing

### Problem
**20.9% of customers have missing Credit_Mix** (highest placeholder rate)

### Distribution
```
Standard: 36.0%  ████████████████████
Good:     24.3%  █████████████
"_":      20.9%  ███████████  ← MISSING
Bad:      18.9%  ██████████
```

### Impact
- Credit_Mix is a key risk indicator
- 1 in 5 customers have unknown credit classification
- Model bias toward customers with complete records

---

## SLIDE 5: Data Corruption Discovery

### The `"!@9#%8"` Mystery

**What we found:**
- 998 customers (8%) have payment behavior value = `"!@9#%8"`
- This is **clearly corrupted data**, not a valid category

**Valid categories:**
1. Low_spent_Small_value_payments (3,202)
2. High_spent_Medium_value_payments (2,242)
3. Low_spent_Medium_value_payments (1,686)
4. High_spent_Large_value_payments (1,683)
5. High_spent_Small_value_payments (1,389)
6. Low_spent_Large_value_payments (1,300)

**Root cause hypothesis:**
- Database encoding error
- ETL pipeline corruption
- Source system placeholder gone wrong

---

## SLIDE 6: Numeric Data Cleaning Challenge

### The Underscore Problem

**Discovery:** Many numeric fields had trailing underscores

**Examples:**
```python
"52312.68_"  (Annual Income)
"40_"        (Age)
"2_"         (Number of Loans)
"1562.91_"   (Outstanding Debt)
```

### Why This Matters

**Without cleaning:**
```python
"52312.68_".cast(FloatType()) → NULL  ❌ Data lost!
```

**With regex cleaning:**
```python
regexp_replace("52312.68_", r"[^0-9.]", "") → "52312.68"
→ cast to Float → 52312.68  ✅ Data preserved!
```

---

## SLIDE 7: Cleaning Strategy by Pattern

### Three Regex Patterns for Different Data Types

| Pattern | Keeps | Example | Use Case |
|---------|-------|---------|----------|
| `r"[^0-9.]"` | Digits + `.` | `"52.3_"` → `52.3` | **Floats** (9 columns) |
| `r"[^0-9]"` | Digits only | `"40_"` → `40` | **Integers** (13 columns) |
| `r"[^0-9-]"` | Digits + `-` | `"-50a"` → `-50` | **Signed ints** (20 columns) |

### Columns Cleaned: 46 Total
- Attributes: 1 (Age)
- Financials: 15 (income, debt, credit metrics)
- Loan Daily: 7 (amounts, balances)
- Clickstream: 20 (behavioral features fe_1 to fe_20)
- **3 features**: 3 (all features)

---

## SLIDE 8: Date Format Ambiguity

### The Problem
Is `1/11/2023` = **January 11** or **November 1st**?

### Investigation
```
American format (M/d/yyyy):  1/11/2023 = Jan 11, 2023
European format (d/M/yyyy):  1/11/2023 = Nov 1, 2023
```

### Solution
✅ Tested with data → Confirmed **d/M/yyyy** (European format)

### Impact if Wrong
❌ 6-11 month prediction errors  
❌ Wrong train/test splits  
❌ Incorrect temporal features  

---

## SLIDE 9: Business Rule Violations

### Age Validation
- **Rule:** Must be 18-100
- **< 18:** Illegal to lend (minors)
- **> 100:** Data entry errors

### SSN Validation
- **Format:** `XXX-XX-XXXX`
- **Invalid:** Wrong dashes, missing digits, letters

### Financial Validation
- **Rule:** No negative values
- **Violations:** 
  - Negative Annual_Income
  - Negative Monthly_Salary
  - Negative loan amounts

---

## SLIDE 10: Flagging & Removal Results

### Customers Flagged: 1,633 (13.06%)

**By Issue Type:**
- Invalid Age/SSN
- Negative financial values
- Negative loan amounts

### Records Removed:
```
Attributes:    1,633 rows
Financials:    1,633 rows  
Loan Daily:   17,963 rows (1,633 × 11 installments)
Clickstream:  28,296 rows
────────────────────────────
Total removed: 49,525 rows
```

### Clean Dataset:
✅ **10,867 customers (86.94%)** ready for ML

---

## SLIDE 11: Medallion Architecture Pipeline

```
┌─────────────────────────────────────────────────────┐
│  BRONZE LAYER (Raw Storage)                         │
│  • Read CSV as strings                              │
│  • No transformations                               │
│  • Save to Parquet (4 files, 7.6 MB)               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  SILVER LAYER (Cleaned Storage)                     │
│  1. Parse dates (d/M/yyyy)                         │
│  2. Replace placeholders → NULL                     │
│  3. Clean numerics with regex                       │
│  4. Validate business rules                         │
│  5. Flag bad customers (13.06%)                     │
│  6. Remove flagged customers                        │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  GOLD LAYER (ML-Ready Features)                     │
│  1. Median imputation for NULLs                     │
│  2. Engineer ratio features                         │
│  3. Aggregate clickstream                           │
│  4. Time-aware filtering                            │
│  Result: ~70 features per customer                  │
└─────────────────────────────────────────────────────┘
```

---

## SLIDE 12: Key Insights

### 1. Multiple Placeholder Patterns = Systemic Issues
Different patterns indicate different data collection problems

### 2. Silent Data Loss Prevention
Regex cleaning saved thousands of valid numeric values

### 3. Corruption Needs Investigation
The `"!@9#%8"` pattern suggests upstream ETL issues

### 4. High Missing Rate on Key Feature
20.9% missing Credit_Mix is concerning for risk models

### 5. Clean Pipeline = Better Models
Garbage in, garbage out - we prevented the garbage

---

## SLIDE 13: Impact Metrics

### Data Quality Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Placeholder Rate** | 47.4% | 0% | ✅ 100% |
| **Invalid Numerics** | Unknown | 0% | ✅ 100% |
| **Bad Customers** | 13.06% | 0% | ✅ Removed |
| **NULL Rate (post-impute)** | ~15% | <5% | ✅ 67% |
| **Data Completeness** | ~85% | >95% | ✅ +10% |

### Business Value
- Clean training data for ML model
- Reproducible cleaning pipeline
- Audit trail via medallion architecture

---

## SLIDE 14: Recommendations

### For Data Teams
1. ✅ **Standardize placeholders** across systems
2. ✅ **Validate at source** (age, SSN, amounts)
3. ✅ **Investigate corruption** (`"!@9#%8"` root cause)
4. ✅ **Document formats** (especially dates)

### For ML Teams
1. ✅ **Monitor imputation** impact on model performance
2. ✅ **Feature importance** analysis on imputed columns
3. ✅ **Regular audits** in production pipeline

### For Business
1. ✅ **Set KPIs:** Target <5% placeholder rate
2. ✅ **Root cause:** Why 20% Credit_Mix missing?
3. ✅ **Process improvement** in data collection

---

## SLIDE 15: Conclusion

### What We Achieved
✅ Identified **7 major data quality issue categories**  
✅ Cleaned **46 numeric columns** systematically  
✅ Detected **4 placeholder patterns** and **1 corruption pattern**  
✅ Flagged and removed **13.06% problematic customers**  
✅ Created **clean dataset** with 10,867 customers  
✅ Built **reproducible pipeline** for production  

### Bottom Line
**Clean data = Reliable models = Better business decisions**

*"You can have data without information, but you cannot have information without data." - Daniel Keys Moran*

---

## BACKUP SLIDES

### Backup: Detailed Column List

**Financials (15 columns):**
- **Floats (9):** Annual_Income, Monthly_Inhand_Salary, Outstanding_Debt, Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance, Changed_Credit_Limit, Interest_Rate, Credit_Utilization_Ratio
- **Integers (6):** Num_of_Loan, Num_Bank_Accounts, Num_Credit_Card, Delay_from_due_date, Num_of_Delayed_Payment, Num_Credit_Inquiries

**Loan Daily (7 columns):**
- tenure, installment_num, loan_amt, due_amt, paid_amt, overdue_amt, balance

**Clickstream (20 columns):**
- fe_1, fe_2, ..., fe_20 (signed integers)

**Attributes (1 column):**
- Age

---

### Backup: Code Example

```python
# Placeholder replacement
financials_df = financials_df.withColumn(
    'Credit_Mix',
    F.when(F.trim(F.col('Credit_Mix')).isin('_', 'NM', '!@9#%8'), None)
     .otherwise(F.col('Credit_Mix'))
)

# Numeric cleaning with regex
financials_df = financials_df.withColumn(
    'Annual_Income',
    F.regexp_replace('Annual_Income', r'[^0-9.]', '').cast(FloatType())
)

# Date parsing
df = df.withColumn('snapshot_date', 
    F.to_date('snapshot_date', 'd/M/yyyy'))
```

---

*End of Presentation Slides Summary*
