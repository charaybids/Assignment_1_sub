# Containerized Data Processing Pipeline for Default Prediction

This project provides a simple, single-container setup with JupyterLab to run a unified pipeline (`main.py`) using PySpark. The pipeline follows the Medallion Architecture (Bronze, Silver, Gold layers) and then trains models, all orchestrated by `main.py`.

## Project Structure

```
Assignment_1_sub/
├── README.md                     # This file
├── docker-compose.yml            # Single container exposing JupyterLab
├── Dockerfile                    # Image with Python, PySpark, JupyterLab
├── requirements.txt              # Python dependencies
├── .dockerignore               # Docker ignore file
├── datamart/                   # Data storage (Bronze, Silver, Gold layers)
│   ├── bronze/                 # Raw ingested data
│   ├── silver/                 # Cleaned and validated data
│   └── gold/                   # Feature-engineered data
├── model_store/                # Trained models storage
├── utils/                      # Shared utility functions
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── spark_utils.py         # Spark session utilities
│   ├── bronze_utils.py        # Bronze layer processing
│   ├── silver_utils.py        # Silver layer processing
│   ├── gold_utils.py          # Gold layer processing
│   └── model_utils.py         # Model training utilities
└── pipelines/                 # Optional: previous modular scripts (no EDA included)
```

## Pipeline Architecture

### 1. Bronze Layer
- **Purpose**: Ingest raw CSV files and partition by date
- **Container**: `bronze-pipeline`
- **Input**: Raw CSV files from `data/` directory
- **Output**: Partitioned Parquet files in `datamart/bronze/`
- **Operations**:
  - Standardize date formats
  - Add year/month partitioning columns
  - Convert to Parquet format for better performance

### 2. Silver Layer
- **Purpose**: Clean and validate data quality
- **Container**: `silver-pipeline`
- **Input**: Bronze layer data
- **Output**: Cleaned data in `datamart/silver/`
- **Operations**:
  - Flag data quality issues (invalid ages, SSN format, negative values)
  - Remove customers with quality issues across all datasets
  - Clean placeholder values and standardize data types

### 3. Gold Layer
- **Purpose**: Feature engineering and label creation
- **Container**: `gold-pipeline`
- **Input**: Silver layer data
- **Output**: Feature-engineered data in `datamart/gold/` and `datamart/label_store/`
- **Operations**:
  - Create time-aware features to prevent data leakage
  - Generate financial stability metrics (DTI, Savings Ratio)
  - Aggregate historical loan and clickstream data
  - Create target labels for default prediction

### 4. Model Training
- **Purpose**: Train and evaluate machine learning models
- **Container**: `model-pipeline`
- **Input**: Gold layer features
- **Output**: Trained models in `model_store/`
- **Operations**:
  - Train Logistic Regression and Random Forest models
  - Evaluate using Area Under ROC (AUC)
  - Extract feature importance for Random Forest

## Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM available for Spark processing
- Raw data files in the `data/` directory:
  - `features_financials.csv`
  - `features_attributes.csv`
  - `lms_loan_daily.csv`
  - `feature_clickstream.csv`

## Setup and Execution

### 1. Prepare Data Directory

Create a `data/` directory in the project root and place your raw CSV files:

```powershell
mkdir data
# Copy your CSV files to the data/ directory
```

### 2. Run the Pipeline

Run these commands from the project root (where `docker-compose.yml` lives):

```powershell
docker-compose build
docker-compose up
```

When the container starts, it will print a JupyterLab URL like:

```
http://127.0.0.1:8888/lab?token=<token>
```

Open the link in your browser, then:
- In JupyterLab, open a Terminal and run:

```powershell
python main.py
```
You can also run from a Python console inside JupyterLab: `!python main.py`.

### 3. Monitor Progress

The pipeline prints concise logs of key steps, row counts, and model metrics.

## Configuration

Pipeline settings can be modified in `utils/config.py`:

```python
# Pipeline Configuration
PREDICTION_MONTHS = 0          # Months after loan start for prediction
LABEL_WINDOW_DAYS = 90         # Days to check for defaults

# Spark Configuration
SPARK_CONFIG = {
    'app_name': 'DefaultPredictionPipeline',
    'master': 'local[*]',
    'driver_memory': '16g',
    'log_level': 'ERROR'
}
```

## Output

After successful execution, you'll have:

1. **Datamart Structure**:
   - `datamart/bronze/`: Partitioned raw data
   - `datamart/silver/`: Cleaned datasets
   - `datamart/gold/`: Feature-engineered data
   - `datamart/label_store/`: Training labels

2. **Trained Models**:
   - `model_store/logistic_regression_pipeline/`
   - `model_store/random_forest_pipeline/`

3. **Performance Metrics**:
   - AUC scores for both models
   - Feature importance rankings

4. EDA and charts were intentionally removed to keep the codebase compact and easy to follow.
