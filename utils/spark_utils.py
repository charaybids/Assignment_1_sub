"""
Spark session utilities
"""
import pyspark as ps
from pyspark.sql import SparkSession


def create_spark_session(app_name, master="local[*]", driver_memory="16g", log_level="ERROR"):
    """
    Create and configure a Spark session
    
    Args:
        app_name (str): Name of the Spark application
        master (str): Spark master URL
        driver_memory (str): Driver memory allocation
        log_level (str): Logging level
    
    Returns:
        SparkSession: Configured Spark session
    """
    spark = ps.sql.SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", driver_memory) \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel(log_level)
    
    print(f"Spark session '{app_name}' created successfully.")
    return spark


def stop_spark_session(spark):
    """
    Stop the Spark session
    
    Args:
        spark (SparkSession): Spark session to stop
    """
    if spark:
        spark.stop()
        print("Spark session stopped.")