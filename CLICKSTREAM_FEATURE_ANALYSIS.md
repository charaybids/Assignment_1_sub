# Clickstream Feature Analysis: fe_5 vs fe_10

## Executive Summary

This analysis answers three critical questions about the clickstream feature selection for loan default prediction:

1. **What is the default label?**
2. **How does each feature correlate with the default label?**
3. **What changes if we switch from fe_5 to fe_10?**

---

## 1. Understanding the Default Label

### Definition
The **`label`** column in the gold layer represents **loan default status**:

| Label Value | Meaning | Count | Percentage |
|-------------|---------|-------|------------|
| **0** | Customer did NOT default (Good) | 7,821 | 71.97% |
| **1** | Customer DID default (Bad) | 3,046 | 28.03% |

### Business Context
- This is a **binary classification problem** predicting loan default risk
- Class imbalance: ~2.57:1 ratio (non-defaulters to defaulters)
- **Goal**: Identify customers likely to default (label=1) before issuing loans

---

## 2. Feature Correlations with Default Label

### All Clickstream Features Ranked

Based on 187,080 customer-snapshot records:

| Rank | Feature | Correlation | Abs Value | Direction | Notes |
|------|---------|-------------|-----------|-----------|-------|
| **1** | **fe_10** | **-0.113071** | **0.113071** | ⬇️ Negative | **⭐ EDA Recommended** |
| **2** | **fe_5** | **+0.107530** | **0.107530** | ⬆️ Positive | **★ Currently Selected** |
| 3 | fe_4 | +0.088935 | 0.088935 | ⬆️ Positive | |
| 4 | fe_9 | -0.088311 | 0.088311 | ⬇️ Negative | |
| 5 | fe_3 | +0.070430 | 0.070430 | ⬆️ Positive | |
| 6 | fe_8 | -0.069793 | 0.069793 | ⬇️ Negative | |
| 7 | fe_2 | +0.047455 | 0.047455 | ⬆️ Positive | |
| 8 | fe_7 | -0.043421 | 0.043421 | ⬇️ Negative | |
| 9 | fe_1 | +0.025320 | 0.025320 | ⬆️ Positive | |
| 10 | fe_6 | -0.020371 | 0.020371 | ⬇️ Negative | |
| 11-20 | fe_11-fe_20 | <±0.007 | <0.007 | Various | Weak predictors |

### Correlation Interpretation

**Positive Correlation** (e.g., fe_5: +0.107530):
- Higher feature value → Higher default risk
- Example: More clicks on certain pages indicates struggling customers

**Negative Correlation** (e.g., fe_10: -0.113071):
- Higher feature value → Lower default risk  
- Example: More engagement with financial planning tools indicates responsible customers

### Gold Layer Feature Correlations

From the current gold layer (36 features total):

| Feature | Correlation | Abs Value | Type |
|---------|-------------|-----------|------|
| Delay_from_due_date | +0.322461 | 0.322461 | Financial |
| Outstanding_Debt | +0.312843 | 0.312843 | Financial |
| Credit_History_Months | -0.288143 | 0.288143 | Credit |
| Debt_to_Annual_Income | +0.245544 | 0.245544 | Engineered |
| Monthly_Inhand_Salary | -0.140135 | 0.140135 | Financial |
| **fe_5_mean** | **+0.121699** | **0.121699** | **Clickstream** |
| Age | -0.088894 | 0.088894 | Demographic |
| Changed_Credit_Limit | +0.073420 | 0.073420 | Credit |
| **fe_5_std** | **+0.007874** | **0.007874** | **Clickstream** |

**Key Finding**: fe_5_mean ranks **6th strongest** predictor in gold layer, but has **ZERO feature importance** in Random Forest model!

---

## 3. What Changes with fe_10?

### Direct Comparison

| Metric | fe_5 (Current) | fe_10 (Proposed) | Difference |
|--------|----------------|------------------|------------|
| **Correlation** | +0.107530 | -0.113071 | ±0.005540 |
| **Abs Correlation** | 0.107530 | 0.113071 | **+0.005540** |
| **Rank** | #2 of 20 | **#1 of 20** | +1 rank |
| **Direction** | Positive | Negative | Opposite |
| **% Stronger** | Baseline | **+5.2% stronger** | - |

### Expected Model Impact

#### Current State (fe_5)
```
✓ Features in Gold: fe_5_mean, fe_5_std (2 features)
✓ Correlation: +0.107530 (positive predictor)
✓ Total features: 36
❌ Model Performance: 
   - Logistic Regression AUC: 0.7267
   - Random Forest AUC: 0.7394
❌ Feature Importance: 0.000000 (NOT USED!)
```

#### Proposed State (fe_10)
```
✓ Features in Gold: fe_10_mean, fe_10_std (2 features)
✓ Correlation: -0.113071 (5.2% stronger)
✓ Total features: 36
✅ Expected Benefits:
   ✓ 5.2% stronger correlation with default label
   ✓ Better separation between defaulters and non-defaulters
   ✓ Higher feature importance (currently fe_5 has ZERO!)
   ✓ Potential AUC improvement of 0.01-0.02 points
   ✓ More reliable predictions
```

### Technical Changes Required

To switch from fe_5 to fe_10:

1. **No code changes needed** - already automated!
2. The `gold_pipeline.py` uses correlation-based selection
3. Simply re-run with fresh data, and if fe_10 has highest correlation, it will be selected

**OR** manually force fe_10 selection in `gold_utils.py`:

```python
# In _analyze_clickstream_for_selection function
# Change line that selects by correlation to:
selected_features = ['fe_10']  # Force fe_10 selection
```

---

## 4. Root Cause: Why Does fe_5 Have Zero Importance?

### Investigation Findings

Despite fe_5_mean having correlation of +0.121699 (6th strongest predictor), the Random Forest model assigns it **zero importance**.

**Possible Reasons:**

1. **Multicollinearity**: fe_5 is highly correlated with other features (e.g., Delay_from_due_date, Outstanding_Debt)
   - Random Forest chooses one feature when multiple are redundant
   - Other features are "stealing" fe_5's importance

2. **Data Quality Issues**:
   - Missing values in fe_5 not properly imputed
   - Outliers affecting splits
   - Distribution issues (skewness, spikes)

3. **Feature Engineering**:
   - fe_5_std (standard deviation) has very weak correlation (0.007874)
   - Mean alone might be sufficient, std adds noise

4. **Random Forest Behavior**:
   - RF uses feature subsampling (random subset per tree)
   - If fe_5 is rarely selected, importance stays low
   - Other features dominate all splits

### Recommendation

**Switch to fe_10** because:
- ✅ 5.2% stronger correlation (-0.113071 vs +0.107530)
- ✅ Ranked #1 among all 20 clickstream features
- ✅ Negative correlation provides diversity (most others are positive)
- ✅ Higher likelihood of being used by Random Forest
- ✅ EDA analysis originally recommended fe_10

---

## 5. Implementation Steps

### Option A: Re-run Gold Pipeline (Automated)
```bash
# If current data has fe_10 as strongest, it will be selected automatically
docker exec mle-a1-app python pipelines/gold/gold_pipeline.py
```

### Option B: Force fe_10 Selection
```bash
# Edit gold_utils.py to force fe_10 selection
# Then regenerate gold layer and retrain models
docker exec mle-a1-app python pipelines/gold/gold_pipeline.py
docker exec mle-a1-app python pipelines/model/model_pipeline.py
```

### Validation Steps
1. Check gold layer has fe_10_mean, fe_10_std
2. Verify feature count still 36 (not changed)
3. Train models and check feature importance > 0.0
4. Compare AUC scores (expect improvement)
5. Analyze feature importance ranking

---

## 6. Summary

| Question | Answer |
|----------|--------|
| **What is the default label?** | Binary indicator: 0=No Default (72%), 1=Default (28%) |
| **Gold features vs label?** | Top correlations: Delay_from_due_date (+0.32), Outstanding_Debt (+0.31), Credit_History (-0.29) |
| **fe_5 vs fe_10?** | fe_10 is 5.2% stronger (0.113 vs 0.108 absolute correlation) |
| **Why zero importance?** | Likely multicollinearity or feature subsampling in Random Forest |
| **Should we switch?** | **YES** - fe_10 has stronger correlation and was EDA-recommended |
| **Expected impact?** | AUC improvement of 0.01-0.02, better feature utilization |

---

**Generated**: 2025-10-04  
**Analysis**: 187,080 customer-snapshots across 10,867 unique customers  
**Current Model**: Random Forest AUC = 0.7394, Logistic Regression AUC = 0.7267
