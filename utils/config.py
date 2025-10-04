"""
Configuration settings for the data processing pipeline.

Only essential settings are kept for clarity.
"""

# ============================================================
# PREDICTION STRATEGY 
# ============================================================

# When to make the prediction 
# MOB=0: At application time (no payment history)
# MOB=1: After first payment
# MOB=3: After 3 payments, etc.
PREDICTION_MOB = 0

# When to observe the outcome 
# MOB=6: Check default at 6th installment
# MOB=12: Check default at end of loan
OBSERVATION_MOB = 6

# Label definition
# "snapshot" - Check if overdue at exact OBSERVATION_MOB
# "window" - Check if ANY overdue between PREDICTION_MOB and OBSERVATION_MOB
# "cumulative" - Check if total overdue > threshold by OBSERVATION_MOB
LABEL_STRATEGY = "snapshot"

# For cumulative strategy: minimum overdue amount to flag as default
OVERDUE_THRESHOLD = 0  # Any overdue > 0 is considered default

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Include loan history features only works if PREDICTION_MOB > 0
# If True, aggregate payment history from MOB=0 to MOB=PREDICTION_MOB
INCLUDE_LOAN_HISTORY_FEATURES = False  # Set True if predicting at MOB > 0

# Clickstream aggregation window (days before loan application)
CLICKSTREAM_LOOKBACK_DAYS = None  # None = use all available data

# Business constraint: maximum loan period in months
MAX_LOAN_MONTHS = 12

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

# Modeling / evaluation configuration
# Use the last N calendar months (by prediction_date) as out-of-time test set
TEST_LAST_N_MONTHS = 3