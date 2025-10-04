# EDA Visualization Summary

## Overview
This directory contains comprehensive exploratory data analysis (EDA) visualizations generated from the loan default prediction pipeline. All charts were created by running `run_eda_analysis.py`.

---

## 📊 Generated Charts

### 1. NULL Distribution Analysis
**File**: `01_null_distribution.png`

- **Purpose**: Visualizes missing data across silver layer datasets
- **Key Insights**:
  - Credit_Mix: 20.76% missing (highest)
  - Type_of_Loan: 11.77% missing
  - Payment_of_Min_Amount: 11.49% missing
  - Payment_Behaviour: 7.90% missing
  - Occupation: 7.03% missing
  - Changed_Credit_Limit: 2.12% missing
- **Impact**: NULL values are handled by median imputation in the gold layer

---

### 2. Customer Statistics
**File**: `02_customer_statistics.png`

- **Purpose**: Shows data quality filtering results
- **Visualizations**:
  - Pie chart: Distribution of removed vs. retained customers
  - Bar chart: Customer counts by quality flag
- **Key Metrics**:
  - Bronze Layer: 12,500 customers (raw data)
  - Silver Layer: 10,867 customers (after cleaning)
  - Removed: 1,633 customers (13.06%)
  - Quality issues: Invalid Age/SSN, negative financial values

---

### 3. Label Distribution
**File**: `03_label_distribution.png`

- **Purpose**: Visualizes target variable distribution for modeling
- **Visualizations**:
  - Bar chart: Counts by label
  - Pie chart: Proportion visualization
- **Key Metrics**:
  - No Default (0): 7,821 customers (71.97%)
  - Default (1): 3,046 customers (28.03%)
  - Imbalance Ratio: 2.57:1 (manageable, no need for SMOTE)
- **Modeling Impact**: Class imbalance is moderate; consider using class weights or stratified sampling

---

### 4. Feature Distributions
**File**: `04_feature_distributions.png`

- **Purpose**: Shows distribution of first 12 numerical features
- **Visualizations**: Histograms with mean lines for:
  - Age
  - Annual_Income
  - Monthly_Inhand_Salary
  - Num_Bank_Accounts
  - Num_Credit_Card
  - Interest_Rate
  - Num_of_Loan
  - Delay_from_due_date
  - Num_of_Delayed_Payment
  - Changed_Credit_Limit
  - Num_Credit_Inquiries
  - Outstanding_Debt
- **Key Observations**:
  - Most features show right-skewed distributions
  - Presence of outliers in several financial metrics
  - Some features may benefit from log transformation

---

## 🖱️ Clickstream Feature Analysis (5 Charts)

### 5. Clickstream Variance Analysis
**File**: `05_clickstream_variance_analysis.png`

- **Purpose**: Ranks all 20 clickstream features by variance and standard deviation
- **Left Chart**: Variance ranking (identifies features with most spread)
- **Right Chart**: Standard deviation ranking
- **Top 10 Features** highlighted in color
- **Use Case**: High variance features capture more information and are better predictors

---

### 6. Clickstream - All Distributions
**File**: `06_clickstream_all_distributions.png`

- **Purpose**: Comprehensive distribution analysis of all 20 clickstream features
- **Layout**: 5×4 grid (20 subplots)
- **Visualizations**: Histograms for fe_1 through fe_20 with:
  - Mean line (red dashed)
  - Median line (green dotted)
  - 50 bins for detailed distribution
- **Key Insights**:
  - fe_10: Shows distinct bimodal distribution
  - Most features: Approximately normal distributions
  - Some features: Heavy tails indicating potential outliers

---

### 7. Clickstream Box Plots
**File**: `07_clickstream_boxplots.png`

- **Purpose**: Outlier detection for all 20 clickstream features
- **Layout**: 5×4 grid (20 box plots)
- **Features Analyzed**: fe_1 through fe_20
- **Key Findings**:
  - Multiple features show outliers (points beyond whiskers)
  - Median values (orange line) vary significantly across features
  - Interquartile ranges differ, indicating varying dispersion
- **Recommendation**: Consider outlier treatment or robust scaling methods

---

### 8. Clickstream Statistics Heatmap
**File**: `08_clickstream_stats_heatmap.png`

- **Purpose**: Summary statistics visualization for all clickstream features
- **Metrics Displayed**:
  - mean
  - std (standard deviation)
  - min
  - 25% (first quartile)
  - 50% (median)
  - 75% (third quartile)
  - max
- **Features**: fe_1 through fe_20 (rows)
- **Color Scale**: Yellow to Red (YlOrRd)
- **Use Case**: Quick reference for feature ranges and central tendencies

---

### 9. Clickstream Correlation Analysis
**File**: `09_clickstream_correlation_analysis.png`

- **Purpose**: Correlation analysis with target label (default/no default)
- **Left Chart**: Absolute correlation ranking
  - Identifies features most predictive of default
  - Top 10 highlighted in red
- **Right Chart**: Raw correlation with direction
  - Green bars: Positive correlation (higher values → more likely to default)
  - Red bars: Negative correlation (higher values → less likely to default)
  - Black dashed line: Zero correlation reference
- **Key Insight**: Feature fe_10 shows highest correlation with default label

---

### 10. Clickstream Selected Features
**File**: `10_clickstream_selected_features.png`

- **Purpose**: Visualizes automated feature selection results
- **Selection Criteria**:
  - ✅ High variance (top 50%)
  - ✅ High correlation with label (top 50%)
  - ✅ Must meet BOTH criteria
- **Visualization**:
  - Green bars: Selected features
  - Gray bars: Not selected
  - Y-axis: Variance values
- **Result**: **1 feature selected** (fe_10)
- **Impact**: Reduces 40 clickstream features (20 means + 20 stds) to 2 features (1 mean + 1 std)
- **Dimensionality Reduction**: 95% reduction in clickstream features

---

## 📈 How to Use These Charts

### For Data Understanding:
1. Review charts 1-4 for data quality assessment
2. Check chart 3 for class imbalance considerations
3. Use chart 4 for feature engineering ideas

### For Clickstream Analysis:
4. Start with chart 6 (distributions) to understand feature shapes
5. Use chart 7 (box plots) to identify outliers
6. Review chart 9 (correlation) to understand feature importance
7. Check chart 10 for automated feature selection results

### For Modeling:
- **Feature Selection**: Use selected features from chart 10
- **Preprocessing**: Consider log transforms for skewed features (chart 4)
- **Outlier Treatment**: Address outliers identified in chart 7
- **Class Imbalance**: Use class weights based on chart 3 ratios

---

## 🔄 Regenerating Charts

To regenerate all charts with updated data:

```bash
# In container
docker exec mle-a1-app python /workspace/run_eda_analysis.py

# Or from host
docker exec mle-a1-app python pipelines/bronze/bronze_pipeline.py
docker exec mle-a1-app python pipelines/silver/silver_pipeline.py
docker exec mle-a1-app python pipelines/gold/gold_pipeline.py
docker exec mle-a1-app python /workspace/run_eda_analysis.py
```

Charts will be saved to: `/workspace/datamart/eda/` (inside container)
or `datamart/eda/` (on host)

---

## 📊 Chart Summary Table

| # | Filename | Purpose | Key Metric | Recommendation |
|---|----------|---------|------------|----------------|
| 1 | `01_null_distribution.png` | Missing data analysis | 20.76% max (Credit_Mix) | Use median imputation |
| 2 | `02_customer_statistics.png` | Data quality | 13.06% removed | Good cleaning |
| 3 | `03_label_distribution.png` | Target distribution | 28.03% default rate | Use class weights |
| 4 | `04_feature_distributions.png` | Feature shapes | Right-skewed features | Consider log transform |
| 5 | `05_clickstream_variance_analysis.png` | Feature variance | Top 10 by variance | Use high-variance features |
| 6 | `06_clickstream_all_distributions.png` | All distributions | fe_10 bimodal | Most predictive feature |
| 7 | `07_clickstream_boxplots.png` | Outlier detection | Multiple outliers | Use robust scaling |
| 8 | `08_clickstream_stats_heatmap.png` | Summary statistics | Complete overview | Reference for ranges |
| 9 | `09_clickstream_correlation_analysis.png` | Feature importance | fe_10 highest correlation | Focus on top correlations |
| 10 | `10_clickstream_selected_features.png` | Feature selection | 1 selected (95% reduction) | Use in final model |

---

## 💡 Key Takeaways

### Data Quality ✅
- 13.06% of customers removed due to quality issues
- Most NULL values < 20% (manageable with imputation)
- Clean data ready for modeling

### Target Variable ✅
- 28.03% default rate (realistic for loan prediction)
- 2.57:1 imbalance ratio (moderate, manageable)
- No extreme class imbalance

### Feature Engineering ✅
- 74 total features in gold layer
- Automated clickstream selection: 1 of 20 features
- Dimensionality reduction: 95% for clickstream features
- Feature `fe_10` is the most predictive

### Model Readiness ✅
- 10,867 samples available
- Balanced dataset (after quality filtering)
- Clear feature importance identified
- Ready for model training

---

## 📁 File Structure

```
datamart/eda/
├── README.md (this file)
├── 01_null_distribution.png
├── 02_customer_statistics.png
├── 03_label_distribution.png
├── 04_feature_distributions.png
├── 05_clickstream_variance_analysis.png
├── 06_clickstream_all_distributions.png
├── 07_clickstream_boxplots.png
├── 08_clickstream_stats_heatmap.png
├── 09_clickstream_correlation_analysis.png
└── 10_clickstream_selected_features.png
```

---

*Generated by: `run_eda_analysis.py`*  
*Date: October 4, 2025*  
*Data: Bronze → Silver → Gold pipeline*
