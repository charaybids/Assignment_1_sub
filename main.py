"""
Main entry point to run the data pipeline (Bronze -> Silver -> Gold -> Models).

What it does:
- Creates the datamart structure (bronze/silver/gold) and model_store if missing
- Bronze: ingest CSVs to parquet partitioned by year/month
- Silver: clean data, flag bad Customer_IDs, remove them, and save cleaned datasets
- Gold: build features and a label store for modeling
- Models: train Logistic Regression and Random Forest and save them

Notes for readers:
- All explanatory comments live here at the top to keep code compact below.
- We intentionally removed EDA code and verbose prints for clarity.
"""
import os
import sys

# Make sure we can import utils from local folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from utils.config import (
    RAW_DATA_PATHS,
    BRONZE_PATH,
    SILVER_PATH,
    GOLD_PATH,
    LABEL_STORE_PATH,
    MODEL_PATH,
    PREDICTION_MONTHS,
    LABEL_WINDOW_MONTHS,
    SPARK_CONFIG,
)
from utils.spark_utils import create_spark_session, stop_spark_session
from utils.bronze_utils import ingest_and_partition_bronze_data, check_bronze_exists
from utils.silver_utils import clean_silver_data, remove_flagged_customers, save_data, check_silver_exists
from utils.gold_utils import create_label_store, create_gold_features, check_gold_exists, check_label_store_exists
from utils.model_utils import prepare_model_data, create_ml_pipeline, train_and_evaluate_model, get_feature_importance


def ensure_datamart_structure():
    os.makedirs(BRONZE_PATH, exist_ok=True)
    os.makedirs(SILVER_PATH, exist_ok=True)
    os.makedirs(GOLD_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    print(f"Ensured datamart structure at: {os.path.join(BASE_DIR, 'datamart')} and model_store")


def run_bronze(spark):
    print("\n=== Running Bronze Pipeline ===")
    if check_bronze_exists(BRONZE_PATH):
        print(f"Bronze already exists at {BRONZE_PATH}; skipping ingest.")
    else:
        ingest_and_partition_bronze_data(RAW_DATA_PATHS, BRONZE_PATH, spark)


def run_silver(spark):
    print("\n=== Running Silver Pipeline ===")
    if check_silver_exists(SILVER_PATH):
        print(f"Silver already exists at {SILVER_PATH}; skipping cleaning.")
        return
    unfiltered_silver_data, flagged_customers = clean_silver_data(BRONZE_PATH, spark)
    silver_data = remove_flagged_customers(unfiltered_silver_data, flagged_customers)
    save_data(silver_data, SILVER_PATH)


def run_gold(spark):
    print("\n=== Running Gold Pipeline ===")
    # Label store
    if check_label_store_exists(LABEL_STORE_PATH):
        label_store_df = spark.read.parquet(LABEL_STORE_PATH)
    else:
        silver_loan_daily = spark.read.parquet(os.path.join(SILVER_PATH, 'loan_daily'))
        label_store_df = create_label_store(silver_loan_daily, PREDICTION_MONTHS, LABEL_WINDOW_MONTHS)
        label_store_df.write.mode("overwrite").parquet(LABEL_STORE_PATH)
        print(f"Saved label store -> {LABEL_STORE_PATH}")

    # Gold features
    if check_gold_exists(GOLD_PATH):
        print(f"Gold already exists at {GOLD_PATH}; skipping features.")
        gold_features_df = spark.read.parquet(GOLD_PATH)
    else:
        gold_features_df = create_gold_features(SILVER_PATH, label_store_df, spark)
        gold_features_df.write.mode("overwrite").parquet(GOLD_PATH)
        print(f"Saved gold features -> {GOLD_PATH}")

    return gold_features_df


def run_modeling(gold_features_df):
    print("\n=== Running Model Training ===")
    model_data, categorical_cols, numerical_cols = prepare_model_data(gold_features_df)
    train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

    # Logistic Regression
    lr_pipeline = create_ml_pipeline(categorical_cols, numerical_cols, "logistic_regression")
    lr_model, lr_auc = train_and_evaluate_model(lr_pipeline, train_data, test_data, MODEL_PATH, 'logistic_regression_pipeline')

    # Random Forest
    rf_pipeline = create_ml_pipeline(categorical_cols, numerical_cols, "random_forest")
    rf_model, rf_auc = train_and_evaluate_model(rf_pipeline, train_data, test_data, MODEL_PATH, 'random_forest_pipeline')
    _ = get_feature_importance(rf_model, categorical_cols, numerical_cols)

    print(f"\nAUC - Logistic Regression: {lr_auc:.4f}")
    print(f"AUC - Random Forest:      {rf_auc:.4f}")


def main():
    ensure_datamart_structure()

    spark = create_spark_session(
        app_name=SPARK_CONFIG['app_name'],
        master=SPARK_CONFIG['master'],
        driver_memory=SPARK_CONFIG['driver_memory'],
        log_level=SPARK_CONFIG['log_level']
    )

    try:
        run_bronze(spark)
        run_silver(spark)
        gold_features_df = run_gold(spark)
        run_modeling(gold_features_df)
        print("\nPipeline complete.")
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
