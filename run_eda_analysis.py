"""
Run EDA Analysis in Docker Container with Visualizations
Executes the main analysis functions from eda_analysis.ipynb
Generates and saves all charts to datamart/eda/
"""

import os
import sys

# Set working directory to /workspace (container path)
if os.path.exists('/workspace'):
    os.chdir('/workspace')
    sys.path.insert(0, '/workspace')
else:
    # Running locally
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.spark_utils import create_spark_session
from utils.config import PREDICTION_MOB, OBSERVATION_MOB, LABEL_STRATEGY, OVERDUE_THRESHOLD
import pyspark.sql.functions as F
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for container
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = "/workspace/datamart/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Charts will be saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("EDA ANALYSIS - Running in Docker Container")
print("="*80 + "\n")

# Initialize Spark
print("Initializing Spark session...")
spark = create_spark_session(app_name="EDA_Analysis")
print(f"✅ Spark session initialized: {spark.version}\n")

# =============================================================================
# SECTION 1: BRONZE LAYER ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("1️⃣ BRONZE LAYER ANALYSIS")
print("="*80 + "\n")

datasets = ['attributes', 'financials', 'clickstream', 'loan_daily']

for dataset in datasets:
    print(f"\n📋 Dataset: {dataset}")
    print("-" * 80)
    
    df = spark.read.parquet(f"/workspace/datamart/bronze/{dataset}.parquet")
    
    # Basic info
    row_count = df.count()
    col_count = len(df.columns)
    print(f"Shape: {row_count:,} rows × {col_count} columns")
    print(f"Columns: {', '.join(df.columns[:10])}{', ...' if col_count > 10 else ''}")
    
    # Show sample
    print(f"\nSample Data (first 2 rows):")
    df.show(2, truncate=True)

# =============================================================================
# SECTION 2: SILVER LAYER ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("2️⃣ SILVER LAYER ANALYSIS")
print("="*80 + "\n")

# Check for NULL values in silver layer
print("NULL Distribution in Silver Layer:")
print("-" * 80)

all_nulls = []

for dataset in ['attributes', 'financials']:
    df = spark.read.parquet(f"/workspace/datamart/silver/{dataset}")
    total_rows = df.count()
    
    print(f"\n📋 {dataset} ({total_rows:,} rows)")
    
    # Calculate null counts
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    null_data = null_counts.collect()[0].asDict()
    
    # Show columns with nulls
    cols_with_nulls = [(col, count) for col, count in null_data.items() if count > 0]
    
    if cols_with_nulls:
        cols_with_nulls.sort(key=lambda x: x[1], reverse=True)
        for col, count in cols_with_nulls[:10]:  # Top 10
            pct = (count / total_rows * 100)
            print(f"  {col}: {count:,} ({pct:.2f}%)")
            all_nulls.append({'dataset': dataset, 'column': col, 'null_count': count, 'null_pct': pct})
    else:
        print("  ✅ No NULL values found")

# Visualize NULL distribution
if all_nulls:
    print("\n📊 Creating NULL distribution chart...")
    null_df = pd.DataFrame(all_nulls)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c' if x > 10 else '#f39c12' if x > 5 else '#3498db' for x in null_df['null_pct']]
    
    bars = ax.barh(range(len(null_df)), null_df['null_pct'], color=colors)
    ax.set_yticks(range(len(null_df)))
    ax.set_yticklabels([f"{row['dataset']}.{row['column']}" for _, row in null_df.iterrows()])
    ax.set_xlabel('NULL Percentage (%)', fontsize=12)
    ax.set_title('NULL Distribution in Silver Layer', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (_, row) in enumerate(null_df.iterrows()):
        ax.text(row['null_pct'] + 0.5, i, f"{row['null_pct']:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_null_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 01_null_distribution.png")

# Customer flagging analysis
print("\n" + "-" * 80)
print("Customer Statistics:")
print("-" * 80)

attributes = spark.read.parquet("/workspace/datamart/silver/attributes")

total_customers = attributes.select("Customer_ID").distinct().count()
print(f"Total Customers in Silver: {total_customers:,}")
print(f"(Flagged customers were already removed in silver layer)")

# Compare with bronze
bronze_attrs = spark.read.parquet("/workspace/datamart/bronze/attributes.parquet")
bronze_customers = bronze_attrs.select("Customer_ID").distinct().count()
removed = bronze_customers - total_customers

print(f"\nBronze Layer: {bronze_customers:,} customers")
print(f"Silver Layer: {total_customers:,} customers")
print(f"Removed: {removed:,} customers ({removed/bronze_customers*100:.2f}%)")

# Visualize customer statistics
print("\n📊 Creating customer statistics chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors = ['#e74c3c', '#2ecc71']
explode = (0.1, 0)
ax1.pie([removed, total_customers], 
        labels=['Removed\n(Flagged)', 'Retained\n(Clean)'],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90)
ax1.set_title('Customer Quality Distribution', fontsize=14, fontweight='bold')

# Bar chart
categories = ['Removed\n(Flagged)', 'Retained\n(Clean)']
values = [removed, total_customers]
bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Customers', fontsize=12)
ax2.set_title('Customer Counts by Quality', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/bronze_customers*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_customer_statistics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 02_customer_statistics.png")

# =============================================================================
# SECTION 3: GOLD LAYER ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3️⃣ GOLD LAYER ANALYSIS")
print("="*80 + "\n")

# Load label store
print("Loading label store...")
label_store = spark.read.parquet("/workspace/datamart/gold/label_store")
print(f"✅ Loaded label store: {label_store.count():,} records\n")

# Label distribution
print("Label Distribution:")
print("-" * 80)
label_dist = label_store.groupBy('label').count().orderBy('label').collect()
total = sum(row['count'] for row in label_dist)

for row in label_dist:
    label_name = "Default (1)" if row['label'] == 1 else "No Default (0)"
    count = row['count']
    pct = (count / total * 100)
    print(f"{label_name}: {count:,} ({pct:.2f}%)")

if len(label_dist) == 2:
    counts = [row['count'] for row in label_dist]
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

# Visualize label distribution
print("\n📊 Creating label distribution charts...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

labels = ['No Default (0)', 'Default (1)']
counts_list = [label_dist[0]['count'], label_dist[1]['count']]
colors = ['#2ecc71', '#e74c3c']

# Bar chart
bars = ax1.bar(labels, counts_list, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Customers', fontsize=12)
ax1.set_title('Label Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/total*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Pie chart
ax2.pie(counts_list, labels=labels, autopct='%1.1f%%', colors=colors, 
       startangle=90, explode=(0, 0.1))
ax2.set_title('Label Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_label_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 03_label_distribution.png")

# Load gold features
print("\n" + "-" * 80)
print("Loading gold features...")
gold_features = spark.read.parquet("/workspace/datamart/gold/features")
print(f"✅ Loaded gold features: {gold_features.count():,} records, {len(gold_features.columns)} columns\n")

# Feature statistics
print("Feature Statistics (first 10 features):")
print("-" * 80)

numeric_cols = [field.name for field in gold_features.schema.fields 
                if str(field.dataType) in ['IntegerType', 'FloatType', 'DoubleType', 'DoubleType()']
                and field.name not in ['Customer_ID', 'label']][:10]

if numeric_cols:
    stats = gold_features.select(numeric_cols).summary("count", "mean", "stddev", "min", "max")
    stats.show(truncate=False)
    
    # Create distribution plots for first 12 features
    print("\n📊 Creating feature distribution charts...")
    pdf = gold_features.select(numeric_cols[:12]).toPandas()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols[:12]):
        ax = axes[i]
        data = pdf[col].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.grid(alpha=0.3)
            
            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.legend(fontsize=7)
    
    plt.suptitle('Feature Distributions (First 12 Features)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 04_feature_distributions.png")

# =============================================================================
# SECTION 4: CLICKSTREAM FEATURE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4️⃣ CLICKSTREAM FEATURE ANALYSIS")
print("="*80 + "\n")

from utils.gold_utils import _analyze_clickstream_for_selection

# Load clickstream
clickstream = spark.read.parquet("/workspace/datamart/silver/clickstream")
print(f"Loaded {clickstream.count():,} clickstream records\n")

# =============================================================================
# DETAILED CLICKSTREAM ANALYSIS WITH VISUALIZATIONS
# =============================================================================

print("="*80)
print("DETAILED CLICKSTREAM FEATURE ANALYSIS")
print("="*80 + "\n")

fe_cols = [f"fe_{i}" for i in range(1, 21)]

# 1. Calculate statistics for all 20 features
print("📊 Calculating statistics for 20 clickstream features...\n")
variance_stats = []

for col_name in fe_cols:
    stats = clickstream.agg(
        F.mean(col_name).alias('mean'),
        F.stddev(col_name).alias('std'),
        F.min(col_name).alias('min'),
        F.max(col_name).alias('max'),
        F.count(col_name).alias('count')
    ).collect()[0]
    
    variance = stats['std'] ** 2 if stats['std'] is not None else 0
    
    variance_stats.append({
        'feature': col_name,
        'mean': stats['mean'],
        'std': stats['std'],
        'variance': variance,
        'min': stats['min'],
        'max': stats['max'],
        'range': stats['max'] - stats['min']
    })

variance_stats_sorted = sorted(variance_stats, key=lambda x: x['variance'], reverse=True)

# Display top 10 by variance
print("Top 10 Features by Variance:")
print("-" * 90)
print(f"{'Rank':<6} {'Feature':<10} {'Mean':<12} {'Std':<12} {'Variance':<15} {'Range':<15}")
print("-" * 90)

for i, stat in enumerate(variance_stats_sorted[:10], 1):
    print(f"{i:<6} {stat['feature']:<10} {stat['mean']:>11.2f} {stat['std']:>11.2f} "
          f"{stat['variance']:>14.2f} {stat['range']:>14.2f}")

# 2. Create comprehensive clickstream visualizations
print("\n📊 Creating comprehensive clickstream charts...\n")

# Chart 1: Variance Ranking
variance_df = pd.DataFrame(variance_stats_sorted)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Variance bar chart
colors_var = ['#e74c3c' if i < 10 else '#95a5a6' for i in range(len(variance_df))]
ax1.barh(range(len(variance_df)), variance_df['variance'], color=colors_var)
ax1.set_yticks(range(len(variance_df)))
ax1.set_yticklabels(variance_df['feature'])
ax1.set_xlabel('Variance', fontsize=12)
ax1.set_title('Clickstream Features Ranked by Variance', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Standard deviation bar chart
colors_std = ['#3498db' if i < 10 else '#95a5a6' for i in range(len(variance_df))]
ax2.barh(range(len(variance_df)), variance_df['std'], color=colors_std)
ax2.set_yticks(range(len(variance_df)))
ax2.set_yticklabels(variance_df['feature'])
ax2.set_xlabel('Standard Deviation', fontsize=12)
ax2.set_title('Clickstream Features Ranked by Std Dev', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_clickstream_variance_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 05_clickstream_variance_analysis.png")

# Chart 2: Distribution of all 20 clickstream features
print("  📊 Creating distribution plots for all 20 features...")
clickstream_pdf = clickstream.select(fe_cols).toPandas()

fig, axes = plt.subplots(5, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(fe_cols):
    ax = axes[i]
    data = clickstream_pdf[col].dropna()
    
    if len(data) > 0:
        ax.hist(data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')
        ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Distribution of All 20 Clickstream Features', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_clickstream_all_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 06_clickstream_all_distributions.png")

# Chart 3: Box plots for outlier detection
print("  📊 Creating box plots for outlier analysis...")
fig, axes = plt.subplots(5, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(fe_cols):
    ax = axes[i]
    data = clickstream_pdf[col].dropna()
    
    if len(data) > 0:
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(alpha=0.3, axis='y')

plt.suptitle('Box Plots: Outlier Analysis for All 20 Clickstream Features', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_clickstream_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 07_clickstream_boxplots.png")

# Chart 4: Heatmap of feature statistics
print("  📊 Creating summary statistics heatmap...")
stats_matrix = clickstream_pdf.describe().T
stats_matrix = stats_matrix[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(stats_matrix.values, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(stats_matrix.columns)))
ax.set_yticks(np.arange(len(stats_matrix.index)))
ax.set_xticklabels(stats_matrix.columns, fontsize=10)
ax.set_yticklabels(stats_matrix.index, fontsize=10)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Value', rotation=270, labelpad=20, fontsize=11)

# Add text annotations
for i in range(len(stats_matrix.index)):
    for j in range(len(stats_matrix.columns)):
        text = ax.text(j, i, f'{stats_matrix.values[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Clickstream Features: Summary Statistics Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_clickstream_stats_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 08_clickstream_stats_heatmap.png")

# Chart 5: Correlation with label
print("  📊 Creating correlation analysis with labels...")

# Run automated feature selection
print("\nRunning automated feature selection...")
print("-" * 80)

results = _analyze_clickstream_for_selection(
    clickstream_df=clickstream,
    label_store_df=label_store,
    top_n=10
)

# Get correlation data for visualization
clickstream_agg = clickstream.groupBy("Customer_ID").agg(
    *[F.mean(c).alias(f"{c}_mean") for c in fe_cols]
)

data_with_labels = label_store.select("Customer_ID", "label") \
    .join(clickstream_agg, "Customer_ID", "inner")

pdf_corr = data_with_labels.toPandas()

# Calculate correlations
correlations = []
for col_name in fe_cols:
    mean_col = f"{col_name}_mean"
    if mean_col in pdf_corr.columns:
        corr = pdf_corr[mean_col].corr(pdf_corr['label'])
        correlations.append({
            'feature': col_name,
            'correlation': abs(corr) if pd.notna(corr) else 0,
            'correlation_raw': corr if pd.notna(corr) else 0
        })

correlations_sorted = sorted(correlations, key=lambda x: x['correlation'], reverse=True)
corr_df = pd.DataFrame(correlations_sorted)

# Create correlation visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Absolute correlation
colors1 = ['#e74c3c' if i < 10 else '#95a5a6' for i in range(len(corr_df))]
ax1.barh(range(len(corr_df)), corr_df['correlation'], color=colors1)
ax1.set_yticks(range(len(corr_df)))
ax1.set_yticklabels(corr_df['feature'])
ax1.set_xlabel('|Correlation with Label|', fontsize=12)
ax1.set_title('Clickstream Features by Absolute Correlation', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Raw correlation (showing direction)
colors2 = ['#2ecc71' if x > 0 else '#e74c3c' for x in corr_df['correlation_raw']]
ax2.barh(range(len(corr_df)), corr_df['correlation_raw'], color=colors2)
ax2.set_yticks(range(len(corr_df)))
ax2.set_yticklabels(corr_df['feature'])
ax2.set_xlabel('Raw Correlation with Label', fontsize=12)
ax2.set_title('Clickstream Correlation Direction', fontsize=14, fontweight='bold')
ax2.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_clickstream_correlation_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 09_clickstream_correlation_analysis.png")

# Chart 6: Selected features comparison
print("  📊 Creating selected features comparison chart...")
selected_features = results['recommended_features']

fig, ax = plt.subplots(figsize=(14, 8))

# Create comparison data
all_features = [s['feature'] for s in variance_stats_sorted]
variance_values = [s['variance'] for s in variance_stats_sorted]
is_selected = ['Selected' if f in selected_features else 'Not Selected' for f in all_features]

colors_selection = ['#2ecc71' if f in selected_features else '#95a5a6' for f in all_features]
bars = ax.barh(range(len(all_features)), variance_values, color=colors_selection, alpha=0.7)

ax.set_yticks(range(len(all_features)))
ax.set_yticklabels(all_features)
ax.set_xlabel('Variance', fontsize=12)
ax.set_title(f'Feature Selection Results: {len(selected_features)} Selected Features (Green)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.7, label=f'Selected ({len(selected_features)})'),
    Patch(facecolor='#95a5a6', alpha=0.7, label=f'Not Selected ({20 - len(selected_features)})')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_clickstream_selected_features.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: 10_clickstream_selected_features.png")

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80 + "\n")

print(f"📊 Selected Features: {sorted(results['recommended_features'])}")
print(f"📈 Total Selected: {len(results['recommended_features'])} features")
print(f"📉 Reduction: {20 - len(results['recommended_features'])} features removed")
print(f"\nThese features have both high variance AND high correlation with default label.")

# =============================================================================
# CLEANUP
# =============================================================================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80 + "\n")

spark.stop()
print("Spark session stopped.")
