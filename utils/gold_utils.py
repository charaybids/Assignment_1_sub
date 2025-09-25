"""
Gold layer utilities: label store creation and feature engineering.
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from utils.config import MAX_LOAN_MONTHS


def create_label_store(loan_daily_df, prediction_months, label_window_months):
    """
    Create label store for training data
    
    Args:
        loan_daily_df (DataFrame): Loan daily data
        prediction_months (int): Months after loan start to make prediction
    label_window_months (int): Months window to check for defaults
        
    Returns:
        DataFrame: Label store with Customer_ID, loan_id, prediction_date, and label
    """
    # Auto-clamp label window so that prediction_months + label_window_months <= MAX_LOAN_MONTHS
    if prediction_months + label_window_months > MAX_LOAN_MONTHS:
        effective_window = max(0, MAX_LOAN_MONTHS - prediction_months)
        print(
            f"[WARN] pm ({prediction_months}) + wm ({label_window_months}) exceeds MAX_LOAN_MONTHS ({MAX_LOAN_MONTHS}). "
            f"Clamping window to {effective_window}."
        )
        label_window_months = effective_window
    else:
        effective_window = label_window_months

    print(
        f"--- Creating Label Store: Predicting at {prediction_months} months with a {effective_window}-month window ---"
    )
    
    loan_daily_with_start_date = loan_daily_df.withColumn(
        "loan_start_date_dt", F.to_date(F.col("loan_start_date"), "M/d/yyyy")
    )
    
    loan_info = loan_daily_with_start_date.groupBy("loan_id", "Customer_ID").agg(
        F.first("loan_start_date_dt").alias("start_date")
    ).filter(F.col("start_date").isNotNull())
    
    labels_df = loan_info.withColumn(
        "prediction_date", F.add_months(F.col("start_date"), prediction_months)
    ).withColumn(
    "label_window_end", F.add_months(F.col("prediction_date"), label_window_months)
    )
    
    loan_events = loan_daily_df.select("loan_id", "snapshot_date", "overdue_amt")
    default_events = labels_df.join(loan_events, on="loan_id") \
        .filter(
            (F.col("snapshot_date") >= F.col("prediction_date")) & 
            (F.col("snapshot_date") <= F.col("label_window_end")) & 
            (F.col("overdue_amt") > 0)
        ) \
        .select("loan_id").distinct().withColumn("defaulted_flag", F.lit(1))
        
    final_labels = labels_df.join(default_events, "loan_id", "left_outer") \
        .withColumn("label", F.when(F.col("defaulted_flag").isNotNull(), 1).otherwise(0)) \
        .select("Customer_ID", "loan_id", "prediction_date", "label")
    
    print("--- Label Store Creation Complete ---")
    final_labels.groupBy("label").count().show()
    return final_labels


def create_gold_features(silver_path, label_store_df, spark_session):
    """
    Create gold layer features with time-aware feature engineering
    
    Args:
        silver_path (str): Path to silver layer data
        label_store_df (DataFrame): Label store DataFrame
        spark_session: Spark session
        
    Returns:
        DataFrame: Gold features DataFrame
    """
    print("--- Starting Gold Pipeline: Time-Aware Feature Engineering ---")
    
    # Load silver data
    attributes_df = spark_session.read.parquet(os.path.join(silver_path, 'attributes'))
    financials_df = spark_session.read.parquet(os.path.join(silver_path, 'financials'))
    loan_daily_df = spark_session.read.parquet(os.path.join(silver_path, 'loan_daily'))
    clickstream_df = spark_session.read.parquet(os.path.join(silver_path, 'clickstream'))
    
    # 1. Filter historical data to be BEFORE the prediction_date to prevent leakage
    loan_history = loan_daily_df.join(
        label_store_df.select("loan_id", "prediction_date"), "loan_id"
    ).filter(F.col("snapshot_date") <= F.col("prediction_date"))
    
    clickstream_history = clickstream_df.join(
        label_store_df.select("Customer_ID", "prediction_date"), "Customer_ID"
    ).filter(F.col("snapshot_date") <= F.col("prediction_date"))

    # 2. Aggregate time-aware history
    loan_agg = loan_history.groupBy("Customer_ID").agg(
        F.sum("paid_amt").alias("hist_total_paid"),
        F.sum("due_amt").alias("hist_total_due"),
        F.sum("overdue_amt").alias("hist_total_overdue_amount")
    ).withColumn(
        "hist_Loan_Payment_Ratio", 
        F.when(F.col("hist_total_due") > 0, F.col("hist_total_paid") / F.col("hist_total_due")).otherwise(1.0)
    )

    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    agg_exprs = [F.mean(c).alias(f"{c}_mean") for c in fe_cols] + \
                [F.stddev(c).alias(f"{c}_std") for c in fe_cols]
    clickstream_agg = clickstream_history.groupBy("Customer_ID").agg(*agg_exprs)
    
    # 3. Get the latest attribute/financials data as of the prediction date
    attributes_as_of = attributes_df.join(
        label_store_df.select("Customer_ID", "prediction_date"), "Customer_ID"
    ).filter(F.col("snapshot_date") <= F.col("prediction_date")) \
     .groupBy("Customer_ID").agg(F.max('snapshot_date').alias('latest_snapshot'))
    
    attributes_latest = attributes_df.join(
        attributes_as_of, 
        on=[attributes_df.Customer_ID == attributes_as_of.Customer_ID, 
            attributes_df.snapshot_date == attributes_as_of.latest_snapshot]
    ).select(attributes_df["*"])
    
    financials_as_of = financials_df.join(
        label_store_df.select("Customer_ID", "prediction_date"), "Customer_ID"
    ).filter(F.col("snapshot_date") <= F.col("prediction_date")) \
     .groupBy("Customer_ID").agg(F.max('snapshot_date').alias('latest_snapshot'))
    
    financials_latest = financials_df.join(
        financials_as_of, 
        on=[financials_df.Customer_ID == financials_as_of.Customer_ID, 
            financials_df.snapshot_date == financials_as_of.latest_snapshot]
    ).select(financials_df["*"])
    
    # 4. Engineer features on the latest financial snapshot
    years_col = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType())
    months_col = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType())
    
    financials_features = financials_latest.withColumn(
        "Credit_History_Months", 
        F.coalesce(years_col, F.lit(0)) * 12 + F.coalesce(months_col, F.lit(0))
    ).withColumn(
        "DTI", F.col("Total_EMI_per_month") / F.col("Monthly_Inhand_Salary")
    ).withColumn(
        "Savings_Ratio", F.col("Amount_invested_monthly") / F.col("Monthly_Inhand_Salary")
    ).withColumn(
        "Monthly_Surplus", 
        F.col("Monthly_Inhand_Salary") - F.col("Total_EMI_per_month") - F.col("Amount_invested_monthly")
    ).withColumn(
        "Debt_to_Annual_Income", F.col("Outstanding_Debt") / F.col("Annual_Income")
    )

    # 5. Join all features together
    model_data = label_store_df.join(attributes_latest, "Customer_ID", "inner") \
                               .join(financials_features.drop("snapshot_date", "year", "month"), "Customer_ID", "left") \
                               .join(loan_agg, "Customer_ID", "left") \
                               .join(clickstream_agg, "Customer_ID", "left")

    print("--- Gold Pipeline Complete ---")
    return model_data


def check_gold_exists(gold_path):
    """
    Return True only if the gold layer directory contains data files.
    """
    if not os.path.isdir(gold_path):
        return False
    for _root, _dirs, files in os.walk(gold_path):
        if files:
            return True
    return False


def check_label_store_exists(label_store_path):
    """
    Return True only if the label store directory contains data files.
    """
    if not os.path.isdir(label_store_path):
        return False
    for _root, _dirs, files in os.walk(label_store_path):
        if files:
            return True
    return False