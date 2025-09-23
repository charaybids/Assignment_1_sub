#!/usr/bin/env python3
"""
Model Training Pipeline - Train and Evaluate ML Models
"""
import sys
import os
sys.path.append('/app/utils')

from spark_utils import create_spark_session, stop_spark_session
from model_utils import (prepare_model_data, create_ml_pipeline, train_and_evaluate_model, 
                        get_feature_importance, check_model_exists)
from config import GOLD_PATH, MODEL_PATH, SPARK_CONFIG


def main():
    """Main function for model training pipeline"""
    print("Starting Model Training Pipeline...")
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG['app_name'] + "_Model",
        master=SPARK_CONFIG['master'],
        driver_memory=SPARK_CONFIG['driver_memory'],
        log_level=SPARK_CONFIG['log_level']
    )
    
    try:
        # Load gold features
        gold_features_df = spark.read.parquet(GOLD_PATH)
        
        # Prepare data for modeling
        model_data, categorical_cols, numerical_cols = prepare_model_data(gold_features_df)
        
        # Split data
        train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
        
        # Train Logistic Regression Model
        lr_pipeline = create_ml_pipeline(categorical_cols, numerical_cols, "logistic_regression")
        lr_model, lr_auc = train_and_evaluate_model(
            lr_pipeline, train_data, test_data, MODEL_PATH, 'logistic_regression_pipeline'
        )
        
        # Train Random Forest Model
        rf_pipeline = create_ml_pipeline(categorical_cols, numerical_cols, "random_forest")
        rf_model, rf_auc = train_and_evaluate_model(
            rf_pipeline, train_data, test_data, MODEL_PATH, 'random_forest_pipeline'
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
        stop_spark_session(spark)


if __name__ == "__main__":
    main()