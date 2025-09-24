"""
Configuration settings for the data processing pipeline.

Only essential settings are kept for clarity.
"""

# Pipeline Configuration
PREDICTION_MONTHS = 0
LABEL_WINDOW_DAYS = 90

# Data paths (relative to project root)
RAW_DATA_PATHS = {
    'financials': 'data/features_financials.csv',
    'attributes': 'data/features_attributes.csv',
    'loan_daily': 'data/lms_loan_daily.csv',
    'clickstream': 'data/feature_clickstream.csv'
}

# Data layer paths
BRONZE_PATH = 'datamart/bronze/'
SILVER_PATH = 'datamart/silver/'
GOLD_PATH = 'datamart/gold/'
LABEL_STORE_PATH = 'datamart/label_store/'
MODEL_PATH = 'model_store/'

# EDA removed

# Spark Configuration
SPARK_CONFIG = {
    'app_name': 'DefaultPredictionPipeline',
    'master': 'local[*]',
    'driver_memory': '16g',
    'log_level': 'ERROR'
}