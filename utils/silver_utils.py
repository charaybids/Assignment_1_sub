"""
Silver layer data processing utilities
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType


def clean_silver_data(bronze_path, spark_session):
    """
    Clean and flag data quality issues in bronze data
    
    Args:
        bronze_path (str): Path to bronze layer data
        spark_session: Spark session
        
    Returns:
        tuple: (cleaned_dataframes_dict, flagged_customer_ids_df)
    """
    print("--- Starting Silver Pipeline: Cleaning and Flagging Data ---")
    
    # Load data from the bronze layer
    attributes_df = spark_session.read.parquet(os.path.join(bronze_path, 'attributes'))
    financials_df = spark_session.read.parquet(os.path.join(bronze_path, 'financials'))
    loan_daily_df = spark_session.read.parquet(os.path.join(bronze_path, 'loan_daily'))
    clickstream_df = spark_session.read.parquet(os.path.join(bronze_path, 'clickstream'))

    # --- Inspection Step ---
    print("\n--- Inspecting Bronze Data: 'attributes' ---")
    attributes_df.printSchema()
    attributes_df.show(5)

    print("\n--- Inspecting Bronze Data: 'financials' ---")
    financials_df.printSchema()
    financials_df.show(5)

    # --- 1. Clean and Flag Attributes Data ---
    attributes_cleaned_placeholders = attributes_df.withColumn("Occupation", 
        F.when(F.trim(F.col("Occupation")).isin("_______", "_"), None)
         .otherwise(F.col("Occupation"))
    )

    attributes_cleaned = attributes_cleaned_placeholders \
        .withColumn("Age", F.regexp_replace(F.col("Age").cast("string"), "_", "").cast(IntegerType())) \
        .withColumn("age_flag", F.when((F.col("Age") < 18) | (F.col("Age") > 100), 1).otherwise(0)) \
        .withColumn("ssn_flag", F.when(F.trim(F.col("SSN")).rlike(r"^\d{3}-\d{2}-\d{4}$"), 0).otherwise(1)) \
        .withColumn("data_quality_issue", F.when((F.col("age_flag") == 1) | (F.col("ssn_flag") == 1), 1).otherwise(0))
    
    # --- 2. Clean and Flag Financials Data ---
    financials_cleaned_placeholders = financials_df
    string_cols_to_clean = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
    for col_name in string_cols_to_clean:
        financials_cleaned_placeholders = financials_cleaned_placeholders.withColumn(col_name, 
            F.when(F.trim(F.col(col_name)).isin("_______", "_", "NM"), None)
             .otherwise(F.col(col_name))
        )

    # Clean numeric columns by removing underscores and casting to float
    numeric_cols_to_clean = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_of_Loan', 
                            'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    for col_name in numeric_cols_to_clean:
        financials_cleaned_placeholders = financials_cleaned_placeholders.withColumn(
            col_name, F.regexp_replace(F.col(col_name).cast("string"), "_", "").cast(FloatType())
        )

    financials_cleaned = financials_cleaned_placeholders \
        .withColumn("negative_financials_flag", F.when(
            (F.col("Annual_Income") < 0) | 
            (F.col("Monthly_Inhand_Salary") < 0) | 
            (F.col("Outstanding_Debt") < 0),
            1
        ).otherwise(0)) \
        .withColumn("data_quality_issue", F.col("negative_financials_flag"))

    # --- 3. Clean and Flag Loan Data ---
    loan_daily_cleaned = loan_daily_df \
        .withColumn("negative_loan_vals_flag", F.when(
            (F.col("loan_amt") < 0) | 
            (F.col("due_amt") < 0) | 
            (F.col("paid_amt") < 0) | 
            (F.col("overdue_amt") < 0),
            1
        ).otherwise(0)) \
        .withColumn("data_quality_issue", F.col("negative_loan_vals_flag"))
    
    # --- 4. Consolidate Flagged Customer IDs ---
    flagged_customers_attr = attributes_cleaned.filter(F.col("data_quality_issue") == 1).select("Customer_ID")
    flagged_customers_fin = financials_cleaned.filter(F.col("data_quality_issue") == 1).select("Customer_ID")
    flagged_customers_loan = loan_daily_cleaned.filter(F.col("data_quality_issue") == 1).select("Customer_ID")
    
    all_flagged_customers = flagged_customers_attr.union(flagged_customers_fin).union(flagged_customers_loan).distinct()
    print(f"Found {all_flagged_customers.count()} customers with data quality issues.")

    # Prepare dictionary of cleaned dataframes (before filtering)
    silver_dfs = {
        'attributes': attributes_cleaned.drop("age_flag", "ssn_flag", "data_quality_issue"),
        'financials': financials_cleaned.drop("negative_financials_flag", "data_quality_issue"),
        'loan_daily': loan_daily_cleaned.drop("negative_loan_vals_flag", "data_quality_issue"),
        'clickstream': clickstream_df
    }
    
    print("--- Silver Pipeline: Cleaning and Flagging Complete ---")
    return silver_dfs, all_flagged_customers


def remove_flagged_customers(silver_dfs, flagged_customer_ids):
    """
    Remove data for flagged customers from all datasets
    
    Args:
        silver_dfs (dict): Dictionary of cleaned DataFrames
        flagged_customer_ids (DataFrame): DataFrame with flagged Customer_IDs
        
    Returns:
        dict: Dictionary of filtered DataFrames
    """
    print("--- Removing Flagged Customer Data ---")
    filtered_silver_dfs = {}
    for name, df in silver_dfs.items():
        initial_count = df.count()
        # Use a left-anti join to keep only rows where Customer_ID is NOT in the flagged list
        filtered_df = df.join(flagged_customer_ids, on="Customer_ID", how="left_anti")
        final_count = filtered_df.count()
        print(f"Removed {initial_count - final_count} rows from '{name}'.")
        filtered_silver_dfs[name] = filtered_df
    
    print("--- Removal of Flagged Customer Data Complete ---")
    return filtered_silver_dfs


def save_data(data_dfs, output_path):
    """
    Save DataFrames to parquet format
    
    Args:
        data_dfs (dict): Dictionary of DataFrames to save
        output_path (str): Output directory path
    """
    print(f"--- Saving Data to {output_path} ---")
    for name, df in data_dfs.items():
        try:
            output_file_path = os.path.join(output_path, name)
            df.write.mode("overwrite").parquet(output_file_path)
            print(f"Successfully saved '{name}' to '{output_file_path}'.")
        except Exception as e:
            print(f"Error saving data for '{name}': {e}")
    print("--- Save Complete ---")


def check_silver_exists(silver_path):
    """
    Return True only if the silver layer directory contains data files.
    """
    if not os.path.isdir(silver_path):
        return False
    for _root, _dirs, files in os.walk(silver_path):
        if files:
            return True
    return False