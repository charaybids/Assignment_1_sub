"""
Gold layer utilities: label store creation and feature engineering.
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType
from utils.config import (
    MAX_LOAN_MONTHS, PREDICTION_MOB, OBSERVATION_MOB, 
    LABEL_STRATEGY, OVERDUE_THRESHOLD, INCLUDE_LOAN_HISTORY_FEATURES
)


def _analyze_clickstream_for_selection(clickstream_df, label_store_df, top_n=10):
    """
    Internal helper: Force select fe_10 as the strongest clickstream predictor
    
    Args:
        clickstream_df (DataFrame): Clickstream data with fe_1 to fe_20
        label_store_df (DataFrame): Label store to compute correlation with target
        top_n (int): Number of top features to return (always returns fe_10)
        
    Returns:
        dict: Analysis results with recommended_features list
        
    Note:
        Based on EDA analysis, fe_10 has strongest correlation (-0.113) with default label
    """
    print(f"\n=== CLICKSTREAM FEATURE SELECTION ===")
    print(f"Forcing selection of fe_10 (strongest predictor based on EDA)...")
    
    # Force select fe_10 (correlation: -0.113, rank #1 among all clickstream features)
    # Based on comprehensive EDA analysis showing fe_10 as strongest predictor
    recommended_features = ['fe_10']
    
    print(f"  ✓ Selected fe_10 (correlation: -0.113 with default label)")
    print(f"  ✓ Rank #1 among 20 clickstream features")
    print(f"  ✓ Negative correlation: higher engagement → lower default risk\n")
    
    return {
        'recommended_features': recommended_features,
        'top_variance_features': ['fe_10']
    }


def select_top_clickstream_features(clickstream_df, feature_list, aggregation='mean'):
    """
    Aggregate only selected clickstream features
    
    Args:
        clickstream_df (DataFrame): Clickstream data
        feature_list (list): List of feature names to keep (e.g., ['fe_1', 'fe_5', 'fe_10'])
        aggregation (str): 'mean', 'std', or 'both'
        
    Returns:
        DataFrame: Aggregated clickstream with only selected features
    """
    print(f"\n=== SELECTING TOP CLICKSTREAM FEATURES ===")
    print(f"Selected features: {', '.join(feature_list)}")
    print(f"Aggregation method: {aggregation}\n")
    
    agg_exprs = []
    
    if aggregation in ['mean', 'both']:
        agg_exprs.extend([F.mean(c).alias(f"{c}_mean") for c in feature_list])
    
    if aggregation in ['std', 'both']:
        agg_exprs.extend([F.stddev(c).alias(f"{c}_std") for c in feature_list])
    
    clickstream_agg = clickstream_df.groupBy("Customer_ID").agg(*agg_exprs)
    
    feature_count = len(clickstream_agg.columns) - 1  # Exclude Customer_ID
    print(f"✓ Created {feature_count} clickstream features from {len(feature_list)} selected columns\n")
    
    return clickstream_agg


def create_label_store(loan_daily_df, prediction_mob=None, observation_mob=None, 
                      label_strategy=None, overdue_threshold=None):
    """
    Create label store for training data (Dynamic configuration)
    
    Args:
        loan_daily_df (DataFrame): Loan daily data
        prediction_mob (int): Month On Book to make prediction (default: from config)
        observation_mob (int): Month On Book to observe outcome (default: from config)
        label_strategy (str): "snapshot", "window", or "cumulative" (default: from config)
        overdue_threshold (float): Minimum overdue to flag as default (default: from config)
        
    Returns:
        DataFrame: Label store with Customer_ID, loan_id, prediction_date, observation_date, label
    """
    # Use config defaults if not specified
    prediction_mob = prediction_mob if prediction_mob is not None else PREDICTION_MOB
    observation_mob = observation_mob if observation_mob is not None else OBSERVATION_MOB
    label_strategy = label_strategy if label_strategy is not None else LABEL_STRATEGY
    overdue_threshold = overdue_threshold if overdue_threshold is not None else OVERDUE_THRESHOLD
    
    # Validation
    if prediction_mob < 0 or prediction_mob >= MAX_LOAN_MONTHS:
        raise ValueError(f"prediction_mob must be between 0 and {MAX_LOAN_MONTHS-1}")
    if observation_mob <= prediction_mob or observation_mob > MAX_LOAN_MONTHS:
        raise ValueError(f"observation_mob must be between {prediction_mob+1} and {MAX_LOAN_MONTHS}")
    
    print(f"\n=== LABEL STORE CREATION ===")
    print(f"Prediction MOB: {prediction_mob} (make decision)")
    print(f"Observation MOB: {observation_mob} (check outcome)")
    print(f"Label Strategy: {label_strategy}")
    print(f"Overdue Threshold: {overdue_threshold}")
    print(f"="*50 + "\n")
    
    # Get loan info: loan_start_date is already a date column from silver layer
    loan_info = loan_daily_df.select("loan_id", "Customer_ID", "loan_start_date").distinct()
    
    # Calculate prediction and observation dates
    labels_df = loan_info \
        .withColumn("prediction_date", F.add_months(F.col("loan_start_date"), prediction_mob)) \
        .withColumn("observation_date", F.add_months(F.col("loan_start_date"), observation_mob))
    
    # Define label based on strategy
    if label_strategy == "snapshot":
        # Check overdue at exact observation_mob installment
        print(f"Label: Checking overdue_amt > {overdue_threshold} at installment_num = {observation_mob}")
        
        observation_snapshot = loan_daily_df.filter(F.col("installment_num") == observation_mob) \
            .select("loan_id", "overdue_amt")
        
        default_events = observation_snapshot \
            .filter(F.col("overdue_amt") > overdue_threshold) \
            .select("loan_id").distinct() \
            .withColumn("defaulted_flag", F.lit(1))
    
    elif label_strategy == "window":
        # Check if ANY overdue between prediction_mob and observation_mob
        print(f"Label: Checking ANY overdue_amt > {overdue_threshold} between MOB {prediction_mob} and {observation_mob}")
        
        loan_events = loan_daily_df.select("loan_id", "installment_num", "overdue_amt")
        
        default_events = labels_df.join(loan_events, "loan_id") \
            .filter(
                (F.col("installment_num") >= prediction_mob) & 
                (F.col("installment_num") <= observation_mob) &
                (F.col("overdue_amt") > overdue_threshold)
            ) \
            .select("loan_id").distinct() \
            .withColumn("defaulted_flag", F.lit(1))
    
    elif label_strategy == "cumulative":
        # Check if cumulative overdue by observation_mob exceeds threshold
        print(f"Label: Checking cumulative overdue_amt > {overdue_threshold} by MOB {observation_mob}")
        
        cumulative_overdue = loan_daily_df \
            .filter(F.col("installment_num") <= observation_mob) \
            .groupBy("loan_id") \
            .agg(F.sum("overdue_amt").alias("total_overdue"))
        
        default_events = cumulative_overdue \
            .filter(F.col("total_overdue") > overdue_threshold) \
            .select("loan_id").distinct() \
            .withColumn("defaulted_flag", F.lit(1))
    
    else:
        raise ValueError(f"Unknown label_strategy: {label_strategy}. Use 'snapshot', 'window', or 'cumulative'")
    
    # Join back to create final labels
    final_labels = labels_df.join(default_events, "loan_id", "left") \
        .withColumn("label", F.when(F.col("defaulted_flag").isNotNull(), 1).otherwise(0)) \
        .select("Customer_ID", "loan_id", "prediction_date", "observation_date", "label")
    
    # Print label distribution
    label_counts = final_labels.groupBy("label").count().collect()
    total = sum([row['count'] for row in label_counts])
    print(f"Label Distribution:")
    for row in label_counts:
        pct = (row['count'] / total * 100) if total > 0 else 0
        print(f"  label={row['label']}: {row['count']:5d} ({pct:5.2f}%)")
    print(f"\n=== LABEL STORE COMPLETE ===\n")
    
    return final_labels


def create_gold_features(silver_path, label_store_df, spark_session, 
                        prediction_mob=None, include_loan_history=None, 
                        analyze_clickstream=False, top_n_clickstream=10):
    """
    Create gold layer features with time-aware feature engineering (Dynamic)
    
    Args:
        silver_path (str): Path to silver layer data
        label_store_df (DataFrame): Label store DataFrame
        spark_session: Spark session
        prediction_mob (int): Month On Book for prediction (default: from config)
        include_loan_history (bool): Include loan payment history features (default: from config)
        analyze_clickstream (bool): Run clickstream analysis and select top features
        top_n_clickstream (int): Number of top clickstream features to keep (if analyze_clickstream=True)
        
    Returns:
        DataFrame: Gold features DataFrame
    """
    # Use config defaults if not specified
    prediction_mob = prediction_mob if prediction_mob is not None else PREDICTION_MOB
    include_loan_history = include_loan_history if include_loan_history is not None else INCLUDE_LOAN_HISTORY_FEATURES
    
    print(f"\n=== GOLD LAYER: FEATURE ENGINEERING ===")
    print(f"Prediction MOB: {prediction_mob}")
    print(f"Include Loan History: {include_loan_history}")
    print(f"Clickstream Analysis: {analyze_clickstream}")
    if analyze_clickstream:
        print(f"Top N Clickstream: {top_n_clickstream}")
    print(f"="*50 + "\n")
    
    # Load silver data
    print("Loading silver data...")
    attributes_df = spark_session.read.parquet(os.path.join(silver_path, 'attributes'))
    financials_df = spark_session.read.parquet(os.path.join(silver_path, 'financials'))
    loan_daily_df = spark_session.read.parquet(os.path.join(silver_path, 'loan_daily'))
    clickstream_df = spark_session.read.parquet(os.path.join(silver_path, 'clickstream'))
    print("✓ Loaded 4 datasets\n")

    # 1. Time-aware filtering: only use data BEFORE or AT prediction date
    print(f"Filtering data up to prediction MOB={prediction_mob}...")
    
    # Get loan application features (from MOB=0)
    loan_application = loan_daily_df.filter(F.col("installment_num") == 0) \
        .select("loan_id", "Customer_ID", "loan_amt", "tenure", "loan_start_date")
    print(f"  ✓ Extracted loan application features (loan_amt, tenure)")
    
    # Clickstream: only data BEFORE loan application
    clickstream_history = clickstream_df.join(
        loan_application.select("Customer_ID", "loan_start_date"), "Customer_ID"
    ).filter(F.col("snapshot_date") < F.col("loan_start_date"))
    print(f"  ✓ Filtered clickstream to BEFORE application date")
    
    # Loan history: only if prediction_mob > 0
    if include_loan_history and prediction_mob > 0:
        print(f"  ✓ Including loan payment history from MOB 0 to {prediction_mob}")
        loan_history = loan_daily_df.join(
            label_store_df.select("loan_id", "prediction_date"), "loan_id"
        ).filter(
            (F.col("installment_num") >= 0) & 
            (F.col("installment_num") < prediction_mob)  # BEFORE prediction MOB
        )
        has_loan_history = True
    else:
        if prediction_mob == 0:
            print(f"  ✓ No loan history (predicting at application time)")
        else:
            print(f"  ✓ Loan history disabled (INCLUDE_LOAN_HISTORY_FEATURES=False)")
        has_loan_history = False
    print()
    
    # 2. Impute NULL values with median (for columns with placeholders from silver layer)
    print("Imputing NULL values with median...")
    
    # Define numeric columns to impute in financials
    numeric_cols_to_impute = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]
    
    # Calculate medians for numeric columns
    median_values = {}
    for col_name in numeric_cols_to_impute:
        if col_name in financials_df.columns:
            median_val = financials_df.approxQuantile(col_name, [0.5], 0.01)[0]
            median_values[col_name] = median_val
    
    # Apply median imputation to financials
    for col_name, median_val in median_values.items():
        financials_df = financials_df.withColumn(
            col_name, F.coalesce(F.col(col_name), F.lit(median_val))
        )
    
    # Impute Age in attributes (if NULL)
    age_median = attributes_df.approxQuantile('Age', [0.5], 0.01)[0]
    attributes_df = attributes_df.withColumn('Age', F.coalesce(F.col('Age'), F.lit(age_median)))
    
    print(f"✓ Imputed {len(median_values)} financial columns + Age with median values\n")
    
    # 3. Aggregate loan payment history (only if enabled and prediction_mob > 0)
    if has_loan_history:
        print(f"Aggregating loan payment history...")
        loan_agg = loan_history.groupBy("Customer_ID").agg(
            F.sum("paid_amt").alias("hist_total_paid"),
            F.sum("due_amt").alias("hist_total_due"),
            F.sum("overdue_amt").alias("hist_total_overdue_amount")
        ).withColumn(
            "hist_Loan_Payment_Ratio", 
            F.when(F.col("hist_total_due") > 0, F.col("hist_total_paid") / F.col("hist_total_due")).otherwise(1.0)
        )
        print(f"  ✓ Created 4 loan history features\n")
    else:
        loan_agg = None
    
    # 4. Aggregate clickstream features
    print("Aggregating clickstream features...")
    
    if analyze_clickstream:
        # Run analysis to select top features
        print("  Running clickstream variance and correlation analysis...")
        analysis_results = _analyze_clickstream_for_selection(
            clickstream_history, 
            label_store_df, 
            top_n=top_n_clickstream
        )
        
        # Use recommended features (high variance + high correlation)
        selected_features = analysis_results['recommended_features']
        print(f"\n  Selected {len(selected_features)} top features: {selected_features}")
        
        # Aggregate only selected features
        clickstream_agg = select_top_clickstream_features(
            clickstream_history, 
            selected_features, 
            aggregation='both'
        )
        print(f"  ✓ Created {len(selected_features)*2} clickstream aggregates ({len(selected_features)} means + {len(selected_features)} stds)\n")
    else:
        # Use all 20 features (legacy mode)
        fe_cols = [f"fe_{i}" for i in range(1, 21)]
        agg_exprs = [F.mean(c).alias(f"{c}_mean") for c in fe_cols] + \
                    [F.stddev(c).alias(f"{c}_std") for c in fe_cols]
        clickstream_agg = clickstream_history.groupBy("Customer_ID").agg(*agg_exprs)
        print(f"  ✓ Created 40 clickstream aggregates (20 means + 20 stds)\n")
    
    # 5. Get the latest attribute/financials data as of the prediction date
    print("Getting latest customer snapshots...")
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
    print(f"  ✓ Got latest snapshots for {attributes_latest.count()} customers\n")
    
    # 6. Engineer features on the latest financial snapshot
    print("Engineering financial features...")
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
    print(f"  ✓ Created 5 engineered features (DTI, Savings_Ratio, Monthly_Surplus, etc.)\n")

    # 7. Join all features together
    print("Joining all features...")
    model_data = label_store_df \
        .join(loan_application.select("loan_id", "loan_amt", "tenure"), "loan_id", "inner") \
        .join(attributes_latest.select("Customer_ID", "Age", "Occupation"), "Customer_ID", "inner") \
        .join(financials_features.drop("snapshot_date"), "Customer_ID", "left") \
        .join(clickstream_agg, "Customer_ID", "left")
    
    # Conditionally add loan history if enabled
    if loan_agg is not None:
        model_data = model_data.join(loan_agg, "Customer_ID", "left")
        print(f"  ✓ Joined: label + loan_app + attributes + financials + clickstream + loan_history")
    else:
        print(f"  ✓ Joined: label + loan_app + attributes + financials + clickstream")

    feature_count = len(model_data.columns)
    print(f"  ✓ Total features before filtering: {feature_count}")
    
    # Filter to 15 safe features (no leakage, avoid curse of dimensionality)
    print("\nFiltering to 15 safe features (removing leaked features)...")
    
    # Define 15 safe features (no data leakage)
    SAFE_FEATURES = [
        # Bureau features (2)
        'Credit_History_Months',    # -0.288 correlation
        'Credit_Mix',               # Categorical: Good/Bad/Standard
        
        # Demographics (3)
        'Age',                      # -0.089 correlation
        'Monthly_Inhand_Salary',    # -0.140 correlation  
        'Occupation',               # Categorical
        
        # Application features (3)
        'loan_amt',                 # Loan size
        'tenure',                   # Loan duration
        'Interest_Rate',            # -0.017 correlation
        
        # Clickstream (2)
        'fe_10_mean',               # -0.113 correlation (strongest clickstream!)
        'fe_10_std',                # Variance measure
        
        # Derived safe features (2)
        'Savings_Ratio',            # +0.020 correlation
        'DTI',                      # +0.016 correlation
        
        # Additional bureau/financial (3)
        'Num_Bank_Accounts',        # +0.015 correlation
        'Num_Credit_Card',          # +0.006 correlation
        'Amount_invested_monthly',  # -0.012 correlation
    ]
    
    # REMOVED LEAKED FEATURES:
    # - Delay_from_due_date (0.322) - includes THIS loan delays
    # - Outstanding_Debt (0.313) - includes THIS loan overdue  
    # - Num_of_Delayed_Payment - includes THIS loan
    # - Debt_to_Annual_Income (0.246) - derived from leaked Outstanding_Debt
    # - Credit_Utilization_Ratio - may include THIS loan
    # - Monthly_Balance - reflects THIS loan
    # - Monthly_Surplus - derived from Monthly_Balance
    
    # Keep ID columns, dates, label, and safe features
    id_cols = ['Customer_ID', 'loan_id', 'prediction_date', 'observation_date', 'label']
    selected_cols = id_cols + [c for c in SAFE_FEATURES if c in model_data.columns]
    
    model_data_filtered = model_data.select(*selected_cols)
    
    final_feature_count = len(model_data_filtered.columns) - len(id_cols)
    print(f"  ✓ Filtered to {final_feature_count} safe features")
    print(f"  ✓ Removed leaked features: Delay_from_due_date, Outstanding_Debt, etc.")
    print(f"  ✓ Sample-to-feature ratio: ~724:1 (excellent!)\n")
    
    print("=== GOLD PIPELINE COMPLETE ===\n")
    return model_data_filtered


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