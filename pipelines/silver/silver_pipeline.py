#!/usr/bin/env python3
"""
Silver Layer Pipeline - Data Cleaning and Validation
"""
import sys
import os
sys.path.append('/app/utils')

from spark_utils import create_spark_session, stop_spark_session
from silver_utils import clean_silver_data, remove_flagged_customers, save_data, check_silver_exists
from config import BRONZE_PATH, SILVER_PATH, SPARK_CONFIG, EDA_OUTPUT_PATH, EDA_TOP_K
from eda_utils import run_eda_on_path


def main():
    """Main function for silver pipeline"""
    print("Starting Silver Layer Pipeline...")
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG['app_name'] + "_Silver",
        master=SPARK_CONFIG['master'],
        driver_memory=SPARK_CONFIG['driver_memory'],
        log_level=SPARK_CONFIG['log_level']
    )
    
    try:
        # Check if silver layer already exists
        if check_silver_exists(SILVER_PATH):
            print(f"Silver layer already exists at '{SILVER_PATH}'. Skipping.")
            for name in ["attributes", "financials", "loan_daily", "clickstream"]:
                path = os.path.join(SILVER_PATH, name)
                if os.path.isdir(path):
                    run_eda_on_path(spark, path, f"silver_{name}", EDA_OUTPUT_PATH, top_k=EDA_TOP_K)
        else:
            # Execute silver pipeline
            unfiltered_silver_data, flagged_customers = clean_silver_data(BRONZE_PATH, spark)
            silver_data = remove_flagged_customers(unfiltered_silver_data, flagged_customers)
            save_data(silver_data, SILVER_PATH)
            for name in silver_data.keys():
                path = os.path.join(SILVER_PATH, name)
                run_eda_on_path(spark, path, f"silver_{name}", EDA_OUTPUT_PATH, top_k=EDA_TOP_K)
            
            # Show sample data
            print("Sample of Cleaned Attributes Data:")
            silver_data['attributes'].show(10)
            print("\nSample of Cleaned Financials Data:")
            silver_data['financials'].show(10)
            print("\nSilver pipeline completed successfully!")
            
    except Exception as e:
        print(f"Error in silver pipeline: {e}")
        sys.exit(1)
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()