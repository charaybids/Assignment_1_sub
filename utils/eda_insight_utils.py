"""
Targeted EDA insights utilities:
 - Default rate by feature bins (quantile-based)
 - Default rate by category for categorical columns
 - Correlation matrix for numeric features
 - Removal summary between bronze and silver layers
"""
from typing import Dict, List, Tuple
import os
import json

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def select_numeric_columns(df: DataFrame, exclude: List[str] = None) -> List[str]:
    exclude = set(exclude or [])
    numeric_types = ("byte", "short", "int", "bigint", "float", "double", "decimal")
    cols = []
    for f in df.schema.fields:
        dt = f.dataType.simpleString().lower()
        if any(dt.startswith(t) for t in numeric_types) and f.name not in exclude:
            cols.append(f.name)
    return cols


def default_rate_by_quantile_bins(df: DataFrame, feature: str, label: str = "label", bins: int = 10, sample: int = 200_000):
    """Return a pandas DataFrame with default rate by quantile bins for a numeric feature."""
    pdf = df.select(feature, label).dropna().limit(sample).toPandas()
    if pdf.empty:
        return None
    import pandas as pd
    try:
        pdf["bin"], bins_used = pd.qcut(pdf[feature], q=bins, duplicates="drop", retbins=True)
    except Exception:
        return None
    agg = pdf.groupby("bin").agg(default_rate=(label, "mean"), count=(label, "size")).reset_index()
    agg["bin_left"] = agg["bin"].apply(lambda iv: iv.left)
    agg["bin_right"] = agg["bin"].apply(lambda iv: iv.right)
    return agg


def plot_default_rate_bins(agg_pdf, out_path: str, title: str, xlabel: str):
    if agg_pdf is None or agg_pdf.empty:
        return
    plt.figure(figsize=(8, 4))
    x = [f"[{l:.2g},{r:.2g}]" for l, r in zip(agg_pdf["bin_left"], agg_pdf["bin_right"])]
    if _HAS_SEABORN:
        import seaborn as sns
        sns.barplot(x=x, y=agg_pdf["default_rate"], color="#4C78A8")
    else:
        plt.bar(x, agg_pdf["default_rate"], color="#4C78A8")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Default rate")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def default_rate_by_category(df: DataFrame, col: str, label: str = "label", top_k: int = 12):
    counts = df.groupBy(col).agg(F.count("*").alias("count"), F.mean(label).alias("default_rate")).orderBy(F.desc("count")).limit(top_k)
    return counts.toPandas()


def plot_default_rate_category(agg_pdf, out_path: str, title: str, ylabel: str = "Default rate"):
    if agg_pdf is None or agg_pdf.empty:
        return
    plt.figure(figsize=(8, 4 + 0.25 * len(agg_pdf)))
    y = agg_pdf.columns[0]
    if _HAS_SEABORN:
        import seaborn as sns
        sns.barplot(data=agg_pdf, x="default_rate", y=y, color="#59A14F")
    else:
        plt.barh(agg_pdf[y].astype(str), agg_pdf["default_rate"], color="#59A14F")
    plt.xlabel(ylabel)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_numeric_correlation(df: DataFrame, exclude: List[str] = None, sample: int = 200_000):
    cols = select_numeric_columns(df, exclude=exclude)
    if not cols:
        return None
    pdf = df.select(*cols).limit(sample).toPandas()
    if pdf.empty:
        return None
    return pdf.corr(numeric_only=True)


def save_correlation_matrix(corr, out_csv: str, out_png: str = None, title: str = "Correlation (numeric)"):
    if corr is None:
        return
    _ensure_dir(os.path.dirname(out_csv))
    corr.to_csv(out_csv)
    if out_png:
        plt.figure(figsize=(min(16, 0.5 * len(corr) + 6), min(12, 0.5 * len(corr) + 5)))
        if _HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(corr, cmap="coolwarm", center=0)
        else:
            import numpy as np
            plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
            plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
            plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()


def summarize_removals(bronze_path: str, silver_path: str, spark: SparkSession) -> Dict:
    """Compute total rows removed from bronze->silver and customers removed in totality."""
    datasets = ["attributes", "financials", "loan_daily", "clickstream"]
    per_ds = []
    bronze_union = None
    silver_union = None
    for name in datasets:
        bp = os.path.join(bronze_path, name)
        sp = os.path.join(silver_path, name)
        if not os.path.isdir(bp) or not os.path.isdir(sp):
            continue
        bdf = spark.read.parquet(bp)
        sdf = spark.read.parquet(sp)
        bcount = bdf.count()
        scount = sdf.count()
        per_ds.append({
            "dataset": name,
            "bronze_rows": bcount,
            "silver_rows": scount,
            "removed_rows": bcount - scount,
        })
        # customer union
        bcu = bdf.select("Customer_ID").distinct()
        scu = sdf.select("Customer_ID").distinct()
        bronze_union = bcu if bronze_union is None else bronze_union.union(bcu)
        silver_union = scu if silver_union is None else silver_union.union(scu)

    bronze_union = bronze_union.distinct() if bronze_union is not None else None
    silver_union = silver_union.distinct() if silver_union is not None else None
    customers_removed = 0
    if bronze_union is not None and silver_union is not None:
        customers_removed = bronze_union.join(silver_union, on="Customer_ID", how="left_anti").count()

    total_removed_rows = sum(d["removed_rows"] for d in per_ds)
    return {"per_dataset": per_ds, "total_removed_rows": total_removed_rows, "customers_removed": customers_removed}


def save_removal_summary(summary: Dict, out_dir: str):
    _ensure_dir(out_dir)
    # JSON
    with open(os.path.join(out_dir, "removal_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # CSV per dataset
    import csv
    with open(os.path.join(out_dir, "removal_summary_per_dataset.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "bronze_rows", "silver_rows", "removed_rows"])
        writer.writeheader()
        for row in summary.get("per_dataset", []):
            writer.writerow(row)
