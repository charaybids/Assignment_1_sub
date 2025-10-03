"""
Silver layer utilities - PySpark data cleaning
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType


def clean_silver_data(bronze_path, spark_session):
    """Clean bronze data and flag quality issues"""
    print("\n=== SILVER LAYER: DATA CLEANING ===\n")
    
    # Load bronze data
    print("Loading bronze data...")
    attributes_df = spark_session.read.parquet(os.path.join(bronze_path, 'attributes.parquet'))
    financials_df = spark_session.read.parquet(os.path.join(bronze_path, 'financials.parquet'))
    loan_daily_df = spark_session.read.parquet(os.path.join(bronze_path, 'loan_daily.parquet'))
    clickstream_df = spark_session.read.parquet(os.path.join(bronze_path, 'clickstream.parquet'))
    print("✓ Loaded 4 datasets\n")

    # Parse dates (d/M/yyyy format)
    print("Parsing dates with format d/M/yyyy...")
    attributes_df = attributes_df.withColumn('snapshot_date', F.to_date('snapshot_date', 'd/M/yyyy'))
    financials_df = financials_df.withColumn('snapshot_date', F.to_date('snapshot_date', 'd/M/yyyy'))
    loan_daily_df = loan_daily_df.withColumn('snapshot_date', F.to_date('snapshot_date', 'd/M/yyyy')) \
                                 .withColumn('loan_start_date', F.to_date('loan_start_date', 'd/M/yyyy'))
    clickstream_df = clickstream_df.withColumn('snapshot_date', F.to_date('snapshot_date', 'd/M/yyyy'))
    print("✓ Parsed 5 date columns\n")

    # --- ATTRIBUTES CLEANING ---
    print("Cleaning attributes dataset...")
    # Replace placeholders with NULL
    attributes_df = attributes_df.withColumn('Occupation', 
        F.when(F.trim(F.col('Occupation')).isin('_______', '_'), None).otherwise(F.col('Occupation')))
    print("  ✓ Replaced placeholders ('_______', '_') with NULL in Occupation")
    
    # Clean Age: remove non-digits, cast to integer
    attributes_df = attributes_df.withColumn('Age', 
        F.regexp_replace('Age', r'[^0-9]', '').cast(IntegerType()))
    print("  ✓ Cleaned Age with r\"[^0-9]\" → IntegerType")
    
    # Flag quality issues
    attributes_df = attributes_df \
        .withColumn('age_flag', F.when((F.col('Age') < 18) | (F.col('Age') > 100), 1).otherwise(0)) \
        .withColumn('ssn_flag', F.when(F.trim(F.col('SSN')).rlike(r'^\d{3}-\d{2}-\d{4}$'), 0).otherwise(1)) \
        .withColumn('data_quality_issue', F.when((F.col('age_flag') == 1) | (F.col('ssn_flag') == 1), 1).otherwise(0))
    print("  ✓ Flagged invalid Age/SSN\n")
    
    # --- FINANCIALS CLEANING ---
    print("Cleaning financials dataset...")
    # Replace categorical placeholders with NULL
    for col_name in ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']:
        financials_df = financials_df.withColumn(col_name,
            F.when(F.trim(F.col(col_name)).isin('_______', '_', 'NM', '!@9#%8'), None).otherwise(F.col(col_name)))
    print("  ✓ Replaced placeholders ('_', 'NM', '!@9#%8') with NULL in 3 categorical columns")
    
    # Clean float columns: keep digits and decimal point only
    float_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 
                  'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
                  'Changed_Credit_Limit', 'Interest_Rate', 'Credit_Utilization_Ratio']
    for col_name in float_cols:
        financials_df = financials_df.withColumn(col_name,
            F.regexp_replace(col_name, r'[^0-9.]', '').cast(FloatType()))
    print(f"  ✓ Cleaned {len(float_cols)} columns with r\"[^0-9.]\" → FloatType")
    
    # Clean integer columns: keep digits only
    int_cols = ['Num_of_Loan', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']
    for col_name in int_cols:
        financials_df = financials_df.withColumn(col_name,
            F.regexp_replace(col_name, r'[^0-9]', '').cast(IntegerType()))
    print(f"  ✓ Cleaned {len(int_cols)} columns with r\"[^0-9]\" → IntegerType")
    
    # Flag quality issues
    financials_df = financials_df \
        .withColumn('negative_financials_flag', F.when(
            (F.col('Annual_Income') < 0) | (F.col('Monthly_Inhand_Salary') < 0) | (F.col('Outstanding_Debt') < 0), 
            1).otherwise(0)) \
        .withColumn('data_quality_issue', F.col('negative_financials_flag'))
    print("  ✓ Flagged negative financial values\n")

    # --- LOAN_DAILY CLEANING ---
    print("Cleaning loan_daily dataset...")
    # Clean integer columns: keep digits only
    loan_int_cols = ['tenure', 'installment_num', 'loan_amt', 'due_amt', 'paid_amt', 'overdue_amt', 'balance']
    for col_name in loan_int_cols:
        loan_daily_df = loan_daily_df.withColumn(col_name,
            F.regexp_replace(col_name, r'[^0-9]', '').cast(IntegerType()))
    print(f"  ✓ Cleaned {len(loan_int_cols)} columns with r\"[^0-9]\" → IntegerType")
    
    # Flag quality issues
    loan_daily_df = loan_daily_df \
        .withColumn('negative_loan_vals_flag', F.when(
            (F.col('loan_amt') < 0) | (F.col('due_amt') < 0) | (F.col('paid_amt') < 0) | (F.col('overdue_amt') < 0),
            1).otherwise(0)) \
        .withColumn('data_quality_issue', F.col('negative_loan_vals_flag'))
    print("  ✓ Flagged negative loan values\n")
    
    # --- CLICKSTREAM CLEANING ---
    print("Cleaning clickstream dataset...")
    # Clean signed integer columns (keep digits and minus sign)
    for i in range(1, 21):
        clickstream_df = clickstream_df.withColumn(f'fe_{i}',
            F.regexp_replace(f'fe_{i}', r'[^0-9-]', '').cast(IntegerType()))
    print("  ✓ Cleaned 20 features (fe_1 to fe_20) with r\"[^0-9-]\" → IntegerType\n")
    
    # --- FLAG CUSTOMERS ---
    print("Identifying flagged customers...")
    flagged_attr = attributes_df.filter(F.col('data_quality_issue') == 1).select('Customer_ID')
    flagged_fin = financials_df.filter(F.col('data_quality_issue') == 1).select('Customer_ID')
    flagged_loan = loan_daily_df.filter(F.col('data_quality_issue') == 1).select('Customer_ID')
    all_flagged = flagged_attr.union(flagged_fin).union(flagged_loan).distinct()
    
    # Calculate flagging statistics
    total_customers = attributes_df.select('Customer_ID').union(
        financials_df.select('Customer_ID')).union(
        loan_daily_df.select('Customer_ID')).union(
        clickstream_df.select('Customer_ID')).distinct().count()
    flagged_count = all_flagged.count()
    pct = (flagged_count / total_customers * 100.0) if total_customers else 0.0
    print(f"✓ Flagged {flagged_count}/{total_customers} customers ({pct:.2f}%) for removal\n")
    
    # Prepare clean datasets (drop quality flag columns)
    silver_dfs = {
        'attributes': attributes_df.drop('age_flag', 'ssn_flag', 'data_quality_issue'),
        'financials': financials_df.drop('negative_financials_flag', 'data_quality_issue'),
        'loan_daily': loan_daily_df.drop('negative_loan_vals_flag', 'data_quality_issue'),
        'clickstream': clickstream_df
    }
    
    print("=== SILVER CLEANING COMPLETE ===\n")
    return silver_dfs, all_flagged


def remove_flagged_customers(silver_dfs, flagged_customers):
    """Remove flagged customers from all datasets"""
    print("=== REMOVING FLAGGED CUSTOMERS ===\n")
    filtered_dfs = {}
    for name, df in silver_dfs.items():
        before = df.count()
        filtered_df = df.join(flagged_customers, on='Customer_ID', how='left_anti')
        after = filtered_df.count()
        removed = before - after
        print(f"  {name:15} - Removed {removed:5} rows ({before} → {after})")
        filtered_dfs[name] = filtered_df
    print("\n=== REMOVAL COMPLETE ===\n")
    return filtered_dfs


def save_data(data_dfs, output_path):
    """Save DataFrames to parquet"""
    print(f"=== SAVING TO {output_path} ===\n")
    for name, df in data_dfs.items():
        output_file = os.path.join(output_path, name)
        df.write.mode('overwrite').parquet(output_file)
        print(f"  ✓ Saved {name}")
    print("\n=== SAVE COMPLETE ===\n")


def check_silver_exists(silver_path):
    """Check if silver layer already exists"""
    if not os.path.isdir(silver_path):
        return False
    for _root, _dirs, files in os.walk(silver_path):
        if files:
            return True
    return False