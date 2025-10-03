#!/usr/bin/env python3
"""Silver Layer Pipeline - Data Cleaning"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from spark_utils import create_spark_session, stop_spark_session
from silver_utils import clean_silver_data, remove_flagged_customers, save_data, check_silver_exists
from config import BRONZE_PATH, SILVER_PATH, SPARK_CONFIG


def main():
    """Execute silver layer pipeline"""
    print("\n" + "="*60)
    print("SILVER LAYER PIPELINE")
    print("="*60 + "\n")
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG['app_name'] + "_Silver",
        master=SPARK_CONFIG['master'],
        driver_memory=SPARK_CONFIG['driver_memory'],
        log_level=SPARK_CONFIG['log_level']
    )
    
    try:
        # Check if silver layer exists
        if check_silver_exists(SILVER_PATH):
            print(f"✓ Silver layer exists at '{SILVER_PATH}' - Skipping\n")
        else:
            # Execute cleaning pipeline
            unfiltered_data, flagged_customers = clean_silver_data(BRONZE_PATH, spark)
            filtered_data = remove_flagged_customers(unfiltered_data, flagged_customers)
            save_data(filtered_data, SILVER_PATH)
            
            # Show sample
            print("=== SAMPLE DATA ===\n")
            print("Attributes:")
            filtered_data['attributes'].show(5, truncate=False)
            print("\nFinancials:")
            filtered_data['financials'].show(5, truncate=False)
            
            print("\n" + "="*60)
            print("✓ SILVER PIPELINE COMPLETE")
            print("="*60 + "\n")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        sys.exit(1)
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()