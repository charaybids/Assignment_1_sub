"""
Configuration settings for the data processing pipeline.

Only essential settings are kept for clarity.
"""

# Pipeline Configuration
# Predict as-of loan start + PREDICTION_MONTHS; label defaults within LABEL_WINDOW_MONTHS after that
PREDICTION_MONTHS = 1
LABEL_WINDOW_MONTHS = 12

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