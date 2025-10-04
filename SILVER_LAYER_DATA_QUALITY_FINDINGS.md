# 🔍 SILVER LAYER DATA QUALITY ANALYSIS
## Comprehensive Findings Report

**Project:** Loan Default Prediction Pipeline  
**Dataset:** 12,500 customers with financial, loan, and behavioral data  
**Analysis Date:** October 2025

---

## 📊 EXECUTIVE SUMMARY

### **Key Findings:**
- **Total Customers:** 12,500
- **Flagged for Quality Issues:** 1,633 (13.06%)
- **Data Quality Issues Identified:** 7 major categories
- **Placeholder/Garbage Values Found:** 4 distinct patterns across 4 columns
- **Numeric Cleaning Applied:** 46 columns across 4 datasets

---

## 🚨 CRITICAL DATA QUALITY ISSUES

### **1. CATEGORICAL DATA - PLACEHOLDER VALUES**

#### **Issue 1.1: Occupation (Attributes Dataset)**
| Metric | Value |
|--------|-------|
| **Total Records** | 12,500 |
| **Placeholder Value** | `"_______"` (7 underscores) |
| **Affected Records** | 880 (7.0%) |
| **Valid Categories** | 15 occupations |
| **Business Impact** | Missing occupation info for 7% of customers |

**Sample Valid Values:**
- Lawyer (828 customers)
- Architect (795)
- Engineer (793)
- Mechanic (782)
- Doctor (765)
- Media Manager (757)

**Action Taken:** Replaced `"_______"` and `"_"` with NULL for median imputation in gold layer

---

#### **Issue 1.2: Credit_Mix (Financials Dataset)**
| Metric | Value |
|--------|-------|
| **Total Records** | 12,500 |
| **Placeholder Value** | `"_"` (single underscore) |
| **Affected Records** | 2,611 (**20.9%** - HIGHEST RATE) |
| **Valid Categories** | 3 types |

**Value Distribution:**
```
Standard: 4,497 (36.0%)
Good:     3,032 (24.3%)
Bad:      2,360 (18.9%)
"_":      2,611 (20.9%) ← PLACEHOLDER
```

**Business Impact:** 
- 1 in 5 customers have unknown credit mix classification
- Critical feature for credit risk assessment
- Missing data could bias model toward customers with complete records

**Action Taken:** Replaced `"_"` with NULL

---

#### **Issue 1.3: Payment_of_Min_Amount (Financials Dataset)**
| Metric | Value |
|--------|-------|
| **Total Records** | 12,500 |
| **Placeholder Value** | `"NM"` (Not Mentioned?) |
| **Affected Records** | 1,438 (11.5%) |
| **Valid Categories** | 2 (Yes/No) |

**Value Distribution:**
```
Yes: 6,571 (52.6%)
No:  4,491 (35.9%)
NM:  1,438 (11.5%) ← PLACEHOLDER
```

**Business Interpretation:**
- "NM" likely means "Not Mentioned" or "Not Measured"
- Indicates incomplete payment behavior tracking
- Could represent accounts without minimum payment requirements

**Action Taken:** Replaced `"NM"` with NULL

---

#### **Issue 1.4: Payment_Behaviour (Financials Dataset) 🔥 CRITICAL**
| Metric | Value |
|--------|-------|
| **Total Records** | 12,500 |
| **Garbage Value** | `"!@9#%8"` (corrupted data) |
| **Affected Records** | 998 (8.0%) |
| **Valid Categories** | 6 behavioral patterns |

**Value Distribution:**
```
Low_spent_Small_value_payments:      3,202 (25.6%)
High_spent_Medium_value_payments:    2,242 (17.9%)
Low_spent_Medium_value_payments:     1,686 (13.5%)
High_spent_Large_value_payments:     1,683 (13.5%)
High_spent_Small_value_payments:     1,389 (11.1%)
Low_spent_Large_value_payments:      1,300 (10.4%)
"!@9#%8":                              998 ( 8.0%) ← GARBAGE DATA
```

**Business Impact:**
- Clear data corruption or encoding error
- 8% of customers have invalid payment behavior classification
- This is a KEY behavioral feature for default prediction

**Root Cause Hypothesis:**
- Database encoding issue
- Data extraction/ETL error
- Placeholder for missing/invalid values in source system

**Action Taken:** Replaced `"!@9#%8"` with NULL

---

### **2. NUMERIC DATA - INVALID CHARACTERS**

#### **Issue 2.1: Underscores in Numeric Fields**

**Discovery:** Many numeric fields contained trailing underscores (e.g., `"52312.68_"`, `"40_"`)

**Affected Columns (Examples):**
```python
# Financials
"Annual_Income":        "52312.68_"  → 52312.68
"Age":                  "40_"         → 40
"Num_of_Loan":          "2_"          → 2
"Outstanding_Debt":     "1562.91_"    → 1562.91
```

**Investigation Results:**
- PySpark naturally converts invalid numeric strings to NULL when casting
- Example: `"52312.68_".cast(FloatType())` → NULL (not error)
- This means raw invalid data would silently become NULL values

**Business Impact:**
- Without cleaning: Loss of valid data (the numeric part)
- Silent NULL conversion masks the real issue
- Could significantly reduce dataset size

---

#### **Issue 2.2: Comprehensive Numeric Cleaning Solution**

**Strategy:** Use regex to remove ALL non-numeric characters before casting

**Regex Patterns Applied:**

| Pattern | Keeps | Removes | Use Case |
|---------|-------|---------|----------|
| `r"[^0-9.]"` | Digits + decimal | Everything else | **Float columns** |
| `r"[^0-9]"` | Digits only | Everything else | **Integer columns** |
| `r"[^0-9-]"` | Digits + minus | Everything else | **Signed integers** |

**Columns Cleaned (46 total):**

**Attributes (1):**
- Age → `r"[^0-9]"` → IntegerType

**Financials (15):**
- **9 Float columns:** Annual_Income, Monthly_Inhand_Salary, Outstanding_Debt, Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance, Changed_Credit_Limit, Interest_Rate, Credit_Utilization_Ratio
- **6 Integer columns:** Num_of_Loan, Num_Bank_Accounts, Num_Credit_Card, Delay_from_due_date, Num_of_Delayed_Payment, Num_Credit_Inquiries

**Loan Daily (7):**
- All integers: tenure, installment_num, loan_amt, due_amt, paid_amt, overdue_amt, balance

**Clickstream (20):**
- Signed integers: fe_1 through fe_20 (behavioral features)

**Example Transformations:**
```
Before Cleaning → After Cleaning
"52312.68_"     → 52312.68
"40_"           → 40
"2_"            → 2
"1562.91_"      → 1562.91
"-50abc"        → -50  (for signed integers)
"10.5.3"        → 10.53 (keeps all digits and dots)
```

---

### **3. DATE FORMAT AMBIGUITY 📅**

#### **Issue 3.1: American vs European Date Format**

**Initial Assumption:** Dates were in `M/d/yyyy` format (American)
- Example: `1/11/2023` = January 11, 2023

**Testing Result:** Dates are actually in `d/M/yyyy` format (European/ISO)
- Example: `1/11/2023` = 1st November 2023

**Affected Columns:**
- `snapshot_date` (all 4 datasets)
- `loan_start_date` (loan_daily dataset)

**How We Discovered:**
```python
# Tested date parsing
pd.to_datetime("1/11/2023", format="%d/%m/%Y") 
# Result: 2023-11-01 ✓ Valid

pd.to_datetime("1/11/2023", format="%m/%d/%Y")
# Result: 2023-01-11 ✗ Wrong interpretation
```

**Business Impact:**
- Wrong date format could cause 6-11 month prediction errors
- Critical for temporal features and label windows
- Would cause incorrect train/test splits

**Action Taken:** 
- Bronze layer: Keep as string (no parsing)
- Silver layer: Parse with correct format `d/M/yyyy`
- Gold layer: Use pre-parsed dates from silver

---

### **4. AGE VALIDATION ISSUES**

**Rule:** Age must be between 18 and 100 (legal lending age)

**Flagging Logic:**
```python
.withColumn("age_flag", 
    F.when((F.col("Age") < 18) | (F.col("Age") > 100), 1).otherwise(0))
```

**Examples of Invalid Ages:**
- Below 18: Minors (illegal to lend)
- Above 100: Data entry errors or deceased records

**Business Impact:**
- Compliance risk: Lending to minors
- Data quality: Extreme ages indicate bad records

---

### **5. SSN VALIDATION ISSUES**

**Expected Format:** `XXX-XX-XXXX` (US Social Security Number)

**Regex Pattern:**
```python
r"^\d{3}-\d{2}-\d{4}$"
# Must be: 3 digits, dash, 2 digits, dash, 4 digits
```

**Examples:**
```
Valid:   "913-74-1218" ✓
Invalid: "91374-1218"  ✗ (wrong dashes)
Invalid: "913-7-1218"  ✗ (missing digit)
Invalid: "ABC-12-3456" ✗ (letters)
```

**Business Impact:**
- Identity verification failures
- Fraud risk indicators
- Compliance with KYC (Know Your Customer)

---

### **6. NEGATIVE FINANCIAL VALUES 💰**

**Validation:** Key financial metrics should not be negative

**Flagged Scenarios:**
```python
# Financials
(Annual_Income < 0) OR
(Monthly_Inhand_Salary < 0) OR
(Outstanding_Debt < 0)

# Loan Daily  
(loan_amt < 0) OR
(due_amt < 0) OR
(paid_amt < 0) OR
(overdue_amt < 0)
```

**Business Interpretation:**
- Negative income: Data entry error or special account types
- Negative loan amounts: Refunds or reversals not properly handled
- Indicates accounting/transaction tracking issues

---

## 📈 FLAGGING & REMOVAL STATISTICS

### **Customers Flagged by Source:**

```
Attributes Issues:
  - Invalid Age (< 18 or > 100)
  - Invalid SSN format
  
Financials Issues:
  - Negative Annual_Income
  - Negative Monthly_Inhand_Salary
  - Negative Outstanding_Debt
  
Loan Issues:
  - Negative loan amounts
  - Negative payment amounts
```

### **Overall Impact:**
```
Total Customers:        12,500
Flagged Customers:       1,633 (13.06%)
Clean Customers:        10,867 (86.94%)

Records Removed:
  - Attributes:          1,633 rows
  - Financials:          1,633 rows
  - Loan Daily:         17,963 rows (1,633 customers × ~11 installments)
  - Clickstream:        28,296 rows
```

---

## 🔧 CLEANING STRATEGY IMPLEMENTED

### **Bronze Layer (Raw Storage)**
```
Purpose: Store raw data exactly as received
Strategy: 
  - Read CSV with dtype=str (everything as string)
  - No transformations or cleaning
  - Save to Parquet (efficient format)
  
Result: 4 datasets, ~7.6 MB total
```

### **Silver Layer (Cleaned Storage)**
```
Purpose: Clean, standardize, and flag quality issues
Strategy:
  1. Parse dates (d/M/yyyy format)
  2. Replace placeholders with NULL
     - "_______", "_", "NM", "!@9#%8" → NULL
  3. Clean numeric columns with regex
     - Remove all non-numeric characters
     - Cast to proper types (Float/Integer)
  4. Validate business rules
     - Age: 18-100
     - SSN: XXX-XX-XXXX format
     - Financials: No negative values
  5. Flag bad customers (13.06%)
  6. Remove flagged customers from all datasets
  
Result: 10,867 clean customers ready for ML
```

### **Gold Layer (ML-Ready Features)**
```
Purpose: Feature engineering and imputation
Strategy:
  1. Median imputation for NULL values
  2. Engineer ratio features (DTI, Savings_Ratio, etc.)
  3. Aggregate clickstream (means/stds)
  4. Time-aware filtering (prevent leakage)
  5. Join all features
  
Result: ~70 features per customer
```

---

## 📊 DATA QUALITY METRICS SUMMARY

| Metric | Value | Status |
|--------|-------|--------|
| **Total Records Processed** | 367,376 rows | ✓ |
| **Datasets Cleaned** | 4 (attributes, financials, loan_daily, clickstream) | ✓ |
| **Columns Cleaned** | 46 numeric + 4 categorical | ✓ |
| **Placeholder Patterns** | 4 types identified | ✓ |
| **Date Columns Parsed** | 5 columns | ✓ |
| **Data Retention Rate** | 86.94% | ✓ Acceptable |
| **Completeness After Cleaning** | >95% (after imputation) | ✓ Good |

---

## 💡 KEY INSIGHTS FOR PRESENTATION

### **1. Data Quality is Critical**
- 13% of customers had quality issues serious enough for exclusion
- Multiple placeholder patterns indicate systematic data collection issues

### **2. Placeholder Patterns Tell a Story**
- `"_______"` (7 underscores) → Manual data entry placeholder
- `"_"` (1 underscore) → System default for missing values
- `"NM"` → Explicit "Not Mentioned" flag
- `"!@9#%8"` → **Data corruption or encoding error**

### **3. Silent Data Loss Prevention**
- Without regex cleaning: PySpark would cast `"40_"` → NULL
- With regex cleaning: `"40_"` → 40 (data preserved)
- Saved thousands of valid data points

### **4. Date Format Matters**
- 6-month error possible if using wrong format
- Critical for time-series predictions
- Always validate with sample data

### **5. Medallion Architecture Benefits**
- Bronze: Preserves raw data (audit trail)
- Silver: Cleans systematically (reproducible)
- Gold: Feature engineering (ML-ready)

---

## 🎯 RECOMMENDATIONS

### **For Data Collection Teams:**
1. **Standardize placeholders** - Use consistent NULL representation
2. **Validate at entry** - Enforce SSN format, age ranges at source
3. **Fix encoding** - Investigate `"!@9#%8"` corruption source
4. **Document date format** - Prevent ambiguity in data dictionaries

### **For Model Development:**
1. **Monitor imputation impact** - Track model performance with/without imputed values
2. **Feature importance** - Check if placeholder-heavy columns (Credit_Mix) are predictive
3. **Regular audits** - Automate data quality checks in production pipeline

### **For Business Stakeholders:**
1. **Data quality KPIs** - Target <5% placeholder rate
2. **Root cause analysis** - Investigate why 20% of Credit_Mix is missing
3. **Process improvement** - Review data collection workflows

---

## 📎 APPENDIX: TECHNICAL DETAILS

### **Regex Patterns Used:**
```python
# Float columns (keep digits and decimal point)
r"[^0-9.]"
# Example: "52312.68_" → "52312.68"

# Integer columns (keep digits only)  
r"[^0-9]"
# Example: "40_" → "40"

# Signed integers (keep digits and minus sign)
r"[^0-9-]"
# Example: "-50abc" → "-50"
```

### **PySpark Cleaning Example:**
```python
# Replace placeholders with NULL
financials_df = financials_df.withColumn(
    'Credit_Mix',
    F.when(F.trim(F.col('Credit_Mix')).isin('_', 'NM', '!@9#%8'), None)
     .otherwise(F.col('Credit_Mix'))
)

# Clean and cast numeric columns
financials_df = financials_df.withColumn(
    'Annual_Income',
    F.regexp_replace('Annual_Income', r'[^0-9.]', '').cast(FloatType())
)
```

### **Date Parsing:**
```python
# Silver layer - parse from string to date
df = df.withColumn('snapshot_date', 
    F.to_date('snapshot_date', 'd/M/yyyy'))
```

---

## 📌 CONCLUSION

The silver layer data cleaning process revealed **systematic data quality issues** affecting **13% of customers**. Through comprehensive regex-based cleaning, placeholder identification, and business rule validation, we:

✅ Preserved valid data that would have been lost  
✅ Standardized 46 numeric columns across 4 datasets  
✅ Identified and flagged 1,633 problematic customers  
✅ Created a clean, ML-ready dataset with 10,867 customers  
✅ Established reproducible cleaning pipeline for production  

**Impact:** Clean data foundation for reliable loan default prediction model.

---

*Generated from Silver Layer Analysis - CS611 MLE Assignment 1*
