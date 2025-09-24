"""
EDA plotting utilities for Spark DataFrames.

Generates:
 - Histograms for numeric columns
 - Top-K bar charts for categorical columns
 - Correlation heatmap for numeric columns

Saves PNGs to a given output directory using a non-interactive backend.
"""
from typing import List, Tuple
import os

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

try:
    # Reuse column splitting from EDA utils if available
    from utils.eda_utils import _split_columns  # type: ignore
except Exception:
    def _split_columns(df: DataFrame) -> Tuple[List[str], List[str]]:
        numeric_types = {"byte", "short", "int", "bigint", "float", "double", "decimal"}
        num_cols, cat_cols = [], []
        for f in df.schema.fields:
            dt = f.dataType.simpleString().lower()
            if any(dt.startswith(t) for t in numeric_types):
                num_cols.append(f.name)
            else:
                cat_cols.append(f.name)
        return num_cols, cat_cols


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def plot_numeric_distributions(df: DataFrame, out_dir: str, dataset_name: str, bins: int = 30, sample_rows: int = 200_000):
    _ensure_dir(out_dir)
    num_cols, _ = _split_columns(df)
    if not num_cols:
        return
    # Sample for plotting efficiency
    pdf = df.select(*num_cols).dropna(how="all").limit(sample_rows).toPandas()
    if pdf.empty:
        return
    # Individual histograms
    for col in num_cols:
        if col not in pdf.columns:
            continue
        series = pdf[col].dropna()
        if series.empty:
            continue
        plt.figure(figsize=(6, 4))
        if _HAS_SEABORN:
            sns.histplot(series, bins=bins, kde=True, stat="density")
        else:
            plt.hist(series, bins=bins, density=True, alpha=0.7)
        plt.title(f"{dataset_name}: {col} distribution")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{_sanitize(dataset_name)}__{_sanitize(col)}__hist.png")
        plt.savefig(fname)
        plt.close()

    # Correlation heatmap
    if len(num_cols) >= 2:
        corr = pdf[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(min(12, 0.6 * len(num_cols) + 4), min(10, 0.6 * len(num_cols) + 3)))
        if _HAS_SEABORN:
            sns.heatmap(corr, cmap="coolwarm", center=0)
        else:
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=6)
            plt.yticks(range(len(num_cols)), num_cols, fontsize=6)
        plt.title(f"{dataset_name}: Numeric correlation heatmap")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{_sanitize(dataset_name)}__correlation_heatmap.png")
        plt.savefig(fname)
        plt.close()


def plot_categorical_topk(df: DataFrame, out_dir: str, dataset_name: str, top_k: int = 20):
    _ensure_dir(out_dir)
    _, cat_cols = _split_columns(df)
    if not cat_cols:
        return
    for col in cat_cols:
        # Skip obviously continuous or date-like columns that ended up as string
        # but still allow plotting if someone wants to see them.
        counts = df.groupBy(col).count().orderBy(F.desc("count")).limit(top_k)
        pdf = counts.toPandas()
        if pdf.empty:
            continue
        plt.figure(figsize=(8, 4 + 0.25 * len(pdf)))
        if _HAS_SEABORN:
            sns.barplot(data=pdf, x="count", y=col, orient="h")
        else:
            plt.barh(pdf[col].astype(str), pdf["count"]) 
        plt.title(f"{dataset_name}: Top-{top_k} values for {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{_sanitize(dataset_name)}__{_sanitize(col)}__top{top_k}.png")
        plt.savefig(fname)
        plt.close()


def generate_plots_for_dataset(df: DataFrame, output_root: str, dataset_name: str, top_k: int = 20, bins: int = 30):
    ds_dir = os.path.join(output_root, _sanitize(dataset_name))
    _ensure_dir(ds_dir)
    plot_numeric_distributions(df, ds_dir, dataset_name, bins=bins)
    plot_categorical_topk(df, ds_dir, dataset_name, top_k=top_k)
