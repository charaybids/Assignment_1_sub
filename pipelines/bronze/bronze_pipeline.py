#!/usr/bin/env python3
"""
Bronze Layer Pipeline - Data Ingestion and Partitioning
"""
import sys
import os
sys.path.append('/app/utils')

from spark_utils import create_spark_session, stop_spark_session
from bronze_utils import ingest_and_partition_bronze_data, check_bronze_exists
from config import RAW_DATA_PATHS, BRONZE_PATH, SPARK_CONFIG


def main():
    """Main function for bronze pipeline"""
    print("Starting Bronze Layer Pipeline...")
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG['app_name'] + "_Bronze",
        master=SPARK_CONFIG['master'],
        driver_memory=SPARK_CONFIG['driver_memory'],
        log_level=SPARK_CONFIG['log_level']
    )
    
    try:
        # Check if bronze layer already exists
        if check_bronze_exists(BRONZE_PATH):
            print(f"Bronze layer already exists at '{BRONZE_PATH}'. Skipping ingestion.")
        else:
            # Execute bronze pipeline
            ingest_and_partition_bronze_data(RAW_DATA_PATHS, BRONZE_PATH, spark)
            print("Bronze pipeline completed successfully!")
            
    except Exception as e:
        print(f"Error in bronze pipeline: {e}")
        sys.exit(1)
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()