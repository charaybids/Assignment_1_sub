#!/usr/bin/env python3
"""
Gold Layer Pipeline - Feature Engineering and Label Store Creation
"""
import sys
import os
sys.path.append('/app/utils')

from spark_utils import create_spark_session, stop_spark_session
from gold_utils import create_label_store, create_gold_features, check_gold_exists, check_label_store_exists
from silver_utils import save_data
from config import SILVER_PATH, GOLD_PATH, LABEL_STORE_PATH, PREDICTION_MONTHS, LABEL_WINDOW_DAYS, SPARK_CONFIG


def main():
    """Main function for gold pipeline"""
    print("Starting Gold Layer Pipeline...")
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG['app_name'] + "_Gold",
        master=SPARK_CONFIG['master'],
        driver_memory=SPARK_CONFIG['driver_memory'],
        log_level=SPARK_CONFIG['log_level']
    )
    
    try:
        # Execute Label Store Creation
        if check_label_store_exists(LABEL_STORE_PATH):
            print(f"Label store already exists at '{LABEL_STORE_PATH}'. Loading...")
            label_store_df = spark.read.parquet(LABEL_STORE_PATH)
        else:
            silver_loan_daily = spark.read.parquet(os.path.join(SILVER_PATH, 'loan_daily'))
            label_store_df = create_label_store(silver_loan_daily, PREDICTION_MONTHS, LABEL_WINDOW_DAYS)
            
            # Save label store
            label_store_df.write.mode("overwrite").parquet(LABEL_STORE_PATH)
            print(f"Successfully saved label store to '{LABEL_STORE_PATH}'.")

        # Execute Gold Features Creation
        if check_gold_exists(GOLD_PATH):
            print(f"Gold layer already exists at '{GOLD_PATH}'. Skipping.")
        else:
            gold_features_df = create_gold_features(SILVER_PATH, label_store_df, spark)
            gold_features_df.write.mode("overwrite").parquet(GOLD_PATH)
            print(f"Successfully saved gold features to '{GOLD_PATH}'.")
            
            # Show sample data
            print("Sample of Final Model Data:")
            gold_features_df.show(5, vertical=True)
            print("\nGold pipeline completed successfully!")
            
    except Exception as e:
        print(f"Error in gold pipeline: {e}")
        sys.exit(1)
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()