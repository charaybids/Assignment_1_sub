# Clickstream Feature Selection Guide

## 📊 For Detailed EDA: Use `utils/eda_analysis.ipynb`

The comprehensive Jupyter notebook contains ALL analysis and visualization functions for exploring data across Bronze, Silver, and Gold layers with **beautiful visualizations**.

---

## Quick Start

### Option 1: Detailed Analysis (Recommended for EDA)
Open `utils/eda_analysis.ipynb` and run the clickstream analysis section:

```python
# In notebook
clickstream = spark.read.parquet("../datamart/silver/clickstream")
label_store = spark.read.parquet("../datamart/gold/label_store")

results = analyze_clickstream_features(clickstream, label_store, top_n=10)
```

**You'll get:**
- ✅ Variance rankings with formatted tables
- ✅ Correlation with label (absolute + directional)  
- ✅ Beautiful visualizations (bar charts, rankings, color coding)
- ✅ Recommended features (high variance + high correlation)
- ✅ Low variance features (candidates for removal)

### Option 2: Pipeline Mode (Automated)
The gold pipeline can automatically select top features:

```python
# In pipelines/gold/gold_pipeline.py
gold_df = create_gold_features(
    silver_path=silver_path,
    label_store_df=label_store,
    spark_session=spark,
    analyze_clickstream=True,  # Enable automatic selection
    top_n_clickstream=10       # Keep top 10 features
)
```

---

## What's in the EDA Notebook?

### Bronze Layer Analysis:
- **`profile_bronze_data()`** - Raw data profiling

### Silver Layer Analysis:
- **`analyze_placeholder_patterns()`** - Placeholder detection
- **`analyze_null_distribution()`** - NULL distribution + charts
- **`analyze_customer_flagging()`** - Quality flags + pie charts
- **`analyze_date_parsing()`** - Date validation

### Gold Layer Analysis:
- **`analyze_label_distribution()`** - Target balance + charts
- **`analyze_feature_distributions()`** - Feature histograms
- **`analyze_clickstream_features()`** - ⭐ **Full clickstream analysis with visualizations**
- **`analyze_feature_correlations()`** - Correlation heatmap

---

## Pipeline Integration Details

### Automated Feature Selection

When you enable `analyze_clickstream=True` in the gold pipeline:

1. **Internal function** `_analyze_clickstream_for_selection()` runs
2. Calculates variance for all 20 features
3. Computes correlation with default label
4. Returns features that are BOTH high variance AND high correlation
5. Only selected features are aggregated

**Result**: Reduces from 40 features → 10-20 features automatically

### Manual Feature Selection

Use `select_top_clickstream_features()` to aggregate specific features:

```python
# After analyzing in notebook, you can manually select features
selected_features = ['fe_2', 'fe_5', 'fe_7', 'fe_9', 'fe_11']

clickstream_agg = select_top_clickstream_features(
    clickstream_df=clickstream,
    feature_list=selected_features,
    aggregation='both'  # mean + std
)
# Output: 10 columns (5 means + 5 stds)
```

---

## Complete Workflow

### Step 1: Run Bronze → Silver → Gold Pipelines
```bash
# Generate all layers first
python main.py  # Or run pipelines individually
```

### Step 2: Open EDA Notebook
```bash
# Open Jupyter
jupyter notebook utils/eda_analysis.ipynb
```

### Step 3: Explore Data
Run cells in the notebook to:
- Profile raw data quality (Bronze)
- Validate cleaning (Silver)
- Analyze clickstream features (Gold)
- Identify top features with visualizations

### Step 4: Update Pipeline (Optional)
Based on analysis, enable automatic selection:

```python
# In pipelines/gold/gold_pipeline.py
gold_df = gold_utils.create_gold_features(
    silver_path=silver_path,
    label_store_df=label_store,
    spark_session=spark,
    analyze_clickstream=True,  # Enable based on notebook findings
    top_n_clickstream=10       # Adjust based on analysis
)
```

### Step 5: Re-run Gold Pipeline
```bash
python pipelines/gold/gold_pipeline.py
```

---

## Configuration Options

### `create_gold_features()` Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_clickstream` | bool | `False` | Enable automatic feature selection |
| `top_n_clickstream` | int | `10` | Number of top features to keep |

**Example Configurations:**

```python
# Use all 20 features (40 total with mean+std)
create_gold_features(..., analyze_clickstream=False)

# Auto-select top 10 features (20 total)  
create_gold_features(..., analyze_clickstream=True, top_n_clickstream=10)

# Auto-select top 5 features (10 total) - more aggressive
create_gold_features(..., analyze_clickstream=True, top_n_clickstream=5)
```

---

## Feature Selection Criteria

### High Variance Features
- Large standard deviation across customers
- Indicates significant variation
- More information content

### High Correlation Features  
- Strong relationship with default label
- Positive or negative correlation both useful
- Directly predictive of target

### Recommended Features
Features that are **BOTH**:
1. In top 50% by variance
2. In top 50% by correlation with label

These are the "sweet spot" features.

### Low Variance Features
- Bottom 25% by variance
- Don't vary much across customers  
- Candidates for removal (low information)

---

## Expected Benefits

| Metric | Before | After (top 10) | Improvement |
|--------|--------|----------------|-------------|
| **Features** | 40 | 20 | 50% reduction |
| **Training Time** | Baseline | ~30-40% faster | Significant |
| **Overfitting Risk** | Higher | Lower | Reduced |
| **Interpretability** | Complex | Simpler | Improved |

---

## Testing

### Test Automated Selection:
Open `utils/eda_analysis.ipynb` and run the "🧪 Pipeline Feature Selection Test" section.

This shows which features would be selected by the pipeline (lightweight, no charts).

### Full Analysis with Visualizations:
Run the "Gold Layer Analysis" section in the notebook for detailed charts and analysis.

---

## Files Reference

| File | Purpose | Use When |
|------|---------|----------|
| `utils/eda_analysis.ipynb` | **All EDA, visualizations, and testing** | Exploring data, presentations, testing |
| `utils/gold_utils.py` | Pipeline functions | Running gold layer |
| `CLICKSTREAM_FEATURE_SELECTION_GUIDE.md` | This guide | Reference |

---

## Pro Tips

1. **Always start with the notebook** - Visualizations help understand your data
2. **Check class balance** - Ensure label distribution is reasonable
3. **Review correlations** - Negative correlations are useful too!
4. **Test different top_n** - Compare model performance with 5, 10, 15 features
5. **Document findings** - Note which features were selected and why

---

## Troubleshooting

**Q: Notebook won't open?**  
A: Make sure Jupyter is installed: `pip install jupyter matplotlib seaborn`

**Q: No visualizations showing?**  
A: Add `%matplotlib inline` at the top of notebook

**Q: Pipeline still uses 40 features?**  
A: Check `analyze_clickstream=True` is set in `create_gold_features()`

**Q: Want more/fewer features?**  
A: Adjust `top_n_clickstream` parameter (5-15 recommended)

---

## Summary

- 📊 **For EDA**: Use `utils/eda_analysis.ipynb` (detailed + visualizations)
- 🔧 **For Pipeline**: Enable `analyze_clickstream=True` (automated)
- 🧪 **For Testing**: Run `test_clickstream_analysis.py` (quick check)
- 📚 **For Reference**: Read this guide

**Recommended workflow**: Notebook → Analyze → Configure Pipeline → Run → Model Training
  - Number of top features to keep when `analyze_clickstream=True`

**Example Usage**:

```python
# Option 1: Use all 20 features (40 total with mean+std)
gold_df = create_gold_features(
    silver_path="datamart/silver",
    label_store_df=label_store,
    spark_session=spark,
    analyze_clickstream=False  # Default
)

# Option 2: Analyze and select top 10 features (20 total with mean+std)
gold_df = create_gold_features(
    silver_path="datamart/silver",
    label_store_df=label_store,
    spark_session=spark,
    analyze_clickstream=True,
    top_n_clickstream=10
)

# Option 3: Select even fewer features (e.g., top 5 = 10 total)
gold_df = create_gold_features(
    silver_path="datamart/silver",
    label_store_df=label_store,
    spark_session=spark,
    analyze_clickstream=True,
    top_n_clickstream=5
)
```

---

## Testing Workflow

### Step 1: Run Analysis Only
Use the test script to see which features are recommended:

```bash
python test_clickstream_analysis.py
```

This will:
1. Load clickstream data
2. Create label store
3. Run variance and correlation analysis
4. Print recommended features
5. Show low-variance features (candidates for removal)

### Step 2: Review Results
Look at the output:
- **Variance Rankings**: Features with high spread/variation
- **Correlation Rankings**: Features most related to default
- **Recommended Features**: Intersection of high variance + high correlation
- **Low Variance Features**: Features with little variation (consider removing)

### Step 3: Update Gold Pipeline
Once you're happy with the selection, update your gold pipeline:

```python
# In pipelines/gold/gold_pipeline.py
gold_df = gold_utils.create_gold_features(
    silver_path=silver_path,
    label_store_df=label_store,
    spark_session=spark,
    analyze_clickstream=True,  # Enable analysis
    top_n_clickstream=10       # Keep top 10
)
```

---

## Feature Selection Criteria

### High Variance Features
- Features with large standard deviation
- Indicates the feature varies significantly across customers
- More likely to contain useful information

### High Correlation Features
- Features with strong relationship to default label
- Positive or negative correlation both useful
- Directly related to prediction target

### Recommended Features
- Features that are BOTH:
  1. In top 50% by variance
  2. In top 50% by correlation with label
- These are the "sweet spot" features

### Low Variance Features
- Bottom 25% by variance
- Features that don't vary much across customers
- Candidates for removal (low information content)

---

## Example Output

When you run the test script, you'll see output like:

```
=== TOP 10 FEATURES BY VARIANCE ===
Rank | Feature | Mean      | Std       | Variance  | Range
-----|---------|-----------|-----------|-----------|----------
1    | fe_5    | 125.4     | 78.3      | 6130.9    | 450.2
2    | fe_12   | -45.2     | 62.1      | 3856.4    | 320.5
3    | fe_18   | 210.3     | 55.8      | 3113.6    | 280.1
...

=== TOP 10 FEATURES BY CORRELATION ===
Rank | Feature | |Correlation| | Raw Correlation | Direction
-----|---------|---------------|-----------------|----------
1    | fe_5    | 0.342         | -0.342          | Negative
2    | fe_18   | 0.289         | 0.289           | Positive
3    | fe_12   | 0.267         | 0.267           | Positive
...

=== RECOMMENDED FEATURES (High Variance + High Correlation) ===
['fe_5', 'fe_12', 'fe_18', 'fe_3', 'fe_9', 'fe_15', 'fe_7', 'fe_20']

=== LOW VARIANCE FEATURES (Bottom 25%) ===
['fe_1', 'fe_4', 'fe_8', 'fe_11', 'fe_16']
```

---

## Recommendations

1. **Start with analysis**: Run `test_clickstream_analysis.py` first
2. **Review results**: Look at variance and correlation rankings
3. **Choose top_n**: Based on model complexity vs performance trade-off
   - Top 5: Very simple model (10 features total)
   - Top 10: Balanced (20 features total) - **recommended**
   - Top 15: More features (30 features total)
4. **Update pipeline**: Enable `analyze_clickstream=True` in gold pipeline
5. **Compare models**: Train with different `top_n_clickstream` values
6. **Monitor performance**: Check if reduced features maintain accuracy

---

## Benefits

1. **Reduced dimensionality**: 40 → 10-20 features
2. **Faster training**: Fewer features = faster model training
3. **Less overfitting**: Fewer features reduce overfitting risk
4. **Better interpretability**: Focus on most important behavioral metrics
5. **Data-driven selection**: Based on variance and correlation, not arbitrary

---

## Notes

- Clickstream negatives are VALID (behavioral metrics)
- Analysis is done at the customer level (aggregated first)
- Correlation requires label_store (won't work without labels)
- Variance analysis works even without labels
- Recommended to keep both mean and std for selected features
