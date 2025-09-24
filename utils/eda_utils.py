"""
EDA utilities for Spark DataFrames.

Generates distribution summaries for numeric and categorical columns and
persists them as CSV files for later inspection.
"""
from typing import Dict, List, Tuple
import os
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


def _split_columns(df: DataFrame) -> Tuple[List[str], List[str]]:
    numeric_types = {"byte", "short", "int", "bigint", "float", "double", "decimal"}
    num_cols, cat_cols = [], []
    for f in df.schema.fields:
        dt = f.dataType.simpleString().lower()
        # decimal(x,y) starts with 'decimal'
        if any(dt.startswith(t) for t in numeric_types):
            num_cols.append(f.name)
        else:
            # Exclude obviously non-feature columns when summarizing (dates, ids) but keep for missingness
            cat_cols.append(f.name)
    return num_cols, cat_cols


def summarize_dataframe(df: DataFrame, top_k: int = 20) -> Dict[str, DataFrame]:
    """
    Build EDA summary dataframes:
      - missingness: null counts and ratios
      - numeric_summary: count, mean, stddev, min, percentiles, max per numeric col
      - categorical_topk: top K frequent values per categorical col

    Returns dict of name -> DataFrame
    """
    # Missingness
    total = df.count()
    miss_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    miss_row = df.agg(*miss_exprs)
    # transform to long format
    miss_cols = (
        miss_row.select(F.expr("stack(%d, %s) as (column, missing_count)" % (
            len(df.columns), ", ".join([f"'{c}', `{c}`" for c in df.columns])
        )))
        .withColumn("total_rows", F.lit(total))
        .withColumn("missing_ratio", F.col("missing_count") / F.col("total_rows"))
        .orderBy(F.desc("missing_ratio"))
    )

    # Numeric summary
    num_cols, cat_cols = _split_columns(df)
    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    numeric_summary_parts: List[DataFrame] = []
    for c in num_cols:
        # approx_quantile is fast; returns list matching percentiles order
        qs = df.approxQuantile(c, percentiles, 0.01) if total > 0 else [None] * len(percentiles)
        base = df.select(
            F.count(F.col(c)).alias("count"),
            F.mean(F.col(c)).alias("mean"),
            F.stddev(F.col(c)).alias("stddev"),
            F.min(F.col(c)).alias("min"),
            F.max(F.col(c)).alias("max"),
        ).withColumn("column", F.lit(c))
        # add percentile columns as literals to avoid collect
        for p, q in zip(percentiles, qs):
            base = base.withColumn(f"p{int(p*100)}", F.lit(q))
        numeric_summary_parts.append(base.select(
            "column", "count", "mean", "stddev", "min",
            "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"
        ))

    numeric_summary = (
        numeric_summary_parts[0].sparkSession.createDataFrame([], schema=numeric_summary_parts[0].schema)
        if not numeric_summary_parts else numeric_summary_parts[0]
    )
    for part in numeric_summary_parts[1:]:
        numeric_summary = numeric_summary.unionByName(part)

    # Categorical top-k frequencies (skip large high-cardinality numerics already handled above)
    cat_topk_parts: List[DataFrame] = []
    for c in cat_cols:
        # For dates/timestamps/arrays/structs this will still work as string display
        topk = (
            df.groupBy(c).count().orderBy(F.desc("count")).limit(top_k)
            .withColumnRenamed(c, "value").withColumn("column", F.lit(c))
        )
        cat_topk_parts.append(topk.select("column", "value", "count"))

    if cat_topk_parts:
        categorical_topk = cat_topk_parts[0]
        for part in cat_topk_parts[1:]:
            categorical_topk = categorical_topk.unionByName(part)
    else:
        categorical_topk = df.sparkSession.createDataFrame([], schema="column string, value string, count long")

    return {
        "missingness": miss_cols,
        "numeric_summary": numeric_summary,
        "categorical_topk": categorical_topk,
    }


def save_eda_reports(eda: Dict[str, DataFrame], output_dir: str, dataset_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, d in eda.items():
        # Write as single CSV for easier reading
        path = os.path.join(output_dir, f"{dataset_name}__{key}")
        (d.coalesce(1)
         .write.mode("overwrite")
         .option("header", True)
         .csv(path))


def run_eda_on_path(spark: SparkSession, path: str, dataset_name: str, output_root: str, top_k: int = 20):
    df = spark.read.parquet(path)
    eda = summarize_dataframe(df, top_k=top_k)
    save_eda_reports(eda, output_root, dataset_name)
    # brief stdout preview
    print(f"EDA written for {dataset_name} at {output_root}")
    eda["missingness"].show(10, truncate=False)
    if not eda["numeric_summary"].rdd.isEmpty():
        eda["numeric_summary"].orderBy(F.desc("p95")).show(5, truncate=False)
    eda["categorical_topk"].show(10, truncate=False)
