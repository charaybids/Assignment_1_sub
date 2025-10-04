#!/usr/bin/env python3
"""
Model Training Pipeline - Train and Evaluate ML Models
"""
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.spark_utils import create_spark_session
from utils.model_utils import (prepare_model_data, create_ml_pipeline, train_and_evaluate_model, 
                        get_feature_importance)
from utils.config import TEST_LAST_N_MONTHS
import pyspark.sql.functions as F


def main():
    """Main function for model training pipeline"""
    print("\n" + "="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Define paths
    gold_path = "datamart/gold/features"
    model_path = "model_store"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Create Spark session
    spark = create_spark_session(
        app_name="DefaultPredictionPipeline_Model",
        master="local[*]",
        driver_memory="16g",
        log_level="ERROR"
    )
    
    try:
        # Load gold features
        print("Loading gold features...")
        gold_features_df = spark.read.parquet(gold_path)
        print(f"✓ Loaded {gold_features_df.count():,} records with {len(gold_features_df.columns)} features\n")
        
        # Prepare data for modeling
        print("Preparing data for modeling...")
        model_data, categorical_cols, numerical_cols = prepare_model_data(gold_features_df)
        print(f"✓ Categorical columns: {len(categorical_cols)}")
        print(f"✓ Numerical columns: {len(numerical_cols)}\n")
        
        # Time-based split using prediction_date (last N calendar months as OOT test)
        print(f"Splitting data (last {TEST_LAST_N_MONTHS} months for test)...")
        if 'prediction_date' in model_data.columns:
            # Convert to first-of-month for grouping; handle both date and string types safely
            md = model_data.withColumn(
                'pred_date_dt', F.to_date(F.col('prediction_date').cast('string'))
            ).withColumn(
                'pred_month', F.date_format(F.col('pred_date_dt'), 'yyyy-MM')
            )

            months_df = md.select('pred_month').distinct().orderBy('pred_month')
            months = [r['pred_month'] for r in months_df.collect()]

            if len(months) == 0:
                print("No prediction months found; falling back to random split.")
                train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
            else:
                k = min(TEST_LAST_N_MONTHS, len(months))
                test_months = set(months[-k:])
                print(f"Time-based split: using last {k} months for test: {sorted(list(test_months))}")
                md = md.cache()
                test_data = md.filter(F.col('pred_month').isin(list(test_months)))
                train_data = md.filter(~F.col('pred_month').isin(list(test_months)))
                # Drop helper columns
                train_data = train_data.drop('pred_date_dt', 'pred_month')
                test_data = test_data.drop('pred_date_dt', 'pred_month')
        else:
            print("prediction_date column not found; using random split.")
            train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
        
        # Train Logistic Regression Model
        lr_pipeline = create_ml_pipeline(categorical_cols, numerical_cols, "logistic_regression")
        lr_model, lr_auc = train_and_evaluate_model(
            lr_pipeline, train_data, test_data, model_path, 'logistic_regression_pipeline'
        )
        
        # Train Random Forest Model
        rf_pipeline = create_ml_pipeline(categorical_cols, numerical_cols, "random_forest")
        rf_model, rf_auc = train_and_evaluate_model(
            rf_pipeline, train_data, test_data, model_path, 'random_forest_pipeline'
        )
        
        # Get feature importance for Random Forest
        feature_importance_df = get_feature_importance(rf_model, categorical_cols, numerical_cols)
        
        print("\n=== Model Training Pipeline Complete ===")
        print(f"Logistic Regression AUC: {lr_auc:.4f}")
        print(f"Random Forest AUC: {rf_auc:.4f}")
        
    except Exception as e:
        print(f"Error in model training pipeline: {e}")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()