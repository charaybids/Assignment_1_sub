#!/usr/bin/env python3
"""
Bronze Layer Pipeline - Data Ingestion and Partitioning
"""
import sys
import os
sys.path.append('/app/utils')

from bronze_utils import ingest_and_partition_bronze_data, check_bronze_exists
from config import RAW_DATA_PATHS, BRONZE_PATH


def main():
    """Main function for bronze pipeline"""
    print("Starting Bronze Layer Pipeline...")
    
    try:
        # Check if bronze layer already exists
        if check_bronze_exists(BRONZE_PATH):
            print(f"Bronze layer already exists at '{BRONZE_PATH}'. Skipping ingestion.")
        else:
            # Execute bronze pipeline
            ingest_and_partition_bronze_data(RAW_DATA_PATHS, BRONZE_PATH)
            print("Bronze pipeline completed successfully!")
            
    except Exception as e:
        print(f"Error in bronze pipeline: {e}")
        sys.exit(1)
    finally:
        pass


if __name__ == "__main__":
    main()