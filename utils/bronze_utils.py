"""
Bronze layer utilities: ingest CSVs and write partitioned parquet.
"""
import os
import pyspark.sql.functions as F


def ingest_and_partition_bronze_data(file_paths, output_path, spark_session):
    """
    Ingest raw CSV files and partition them by year and month
    
    Args:
        file_paths (dict): Dictionary of file names and paths
        output_path (str): Output path for bronze layer
        spark_session: Spark session
    """
    print("--- Starting Bronze Pipeline: Ingesting Raw Data ---")
    
    for name, path in file_paths.items():
        try:
            # Read CSV with header and inferred schema
            df = spark_session.read.csv(path, header=True, inferSchema=True)
            
            # Resiliently drop year/month if they exist from the source
            if 'year' in df.columns:
                df = df.drop('year')
            if 'month' in df.columns:
                df = df.drop('month')

            # Standardize snapshot_date column format and extract year/month for partitioning
            df = df.withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "M/d/yyyy")) \
                   .withColumn("year", F.year(F.col("snapshot_date"))) \
                   .withColumn("month", F.month(F.col("snapshot_date")))
            
            # Write data to bronze layer, partitioned by year and month
            partition_output_path = os.path.join(output_path, name)
            df.write.partitionBy("year", "month").mode("overwrite").parquet(partition_output_path)
            
            print(f"Successfully ingested and partitioned '{name}' to '{partition_output_path}'.")
            
        except Exception as e:
            print(f"Error processing file '{path}': {e}")
            
    print("--- Bronze Pipeline Complete ---")


def check_bronze_exists(bronze_path):
    """
    Return True only if the bronze layer directory contains data files
    (not just an empty folder structure).
    """
    if not os.path.isdir(bronze_path):
        return False
    for _root, _dirs, files in os.walk(bronze_path):
        if files:
            return True
    return False