"""
Model training and evaluation utilities
"""
import os
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StringType


def prepare_model_data(gold_features_df):
    """
    Prepare data for model training by identifying feature columns
    
    Args:
        gold_features_df (DataFrame): Gold features DataFrame
        
    Returns:
        tuple: (model_data, categorical_cols, numerical_cols)
    """
    model_data = gold_features_df
    ignore_cols = ['Customer_ID', 'Name', 'SSN', 'snapshot_date', 'loan_id', 
                   'Credit_History_Age', 'label', 'prediction_date', 'start_date'] 
    
    categorical_cols = [c for c in model_data.columns 
                       if isinstance(model_data.schema[c].dataType, StringType) and c not in ignore_cols]
    numerical_cols = [c for c in model_data.columns if c not in categorical_cols + ignore_cols]
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    return model_data, categorical_cols, numerical_cols


def create_ml_pipeline(categorical_cols, numerical_cols, model_type="logistic_regression"):
    """
    Create ML pipeline with preprocessing and model
    
    Args:
        categorical_cols (list): List of categorical column names
        numerical_cols (list): List of numerical column names
        model_type (str): Type of model ("logistic_regression" or "random_forest")
        
    Returns:
        Pipeline: ML Pipeline
    """
    # Create preprocessing stages
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") 
                for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") 
                for c in categorical_cols]
    imputer = Imputer(inputCols=numerical_cols, 
                     outputCols=[f"{c}_imputed" for c in numerical_cols]).setStrategy("median")
    
    assembler_inputs = [f"{c}_ohe" for c in categorical_cols] + [f"{c}_imputed" for c in numerical_cols]
    vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    
    # Create model
    if model_type == "logistic_regression":
        model = LogisticRegression(featuresCol="features", labelCol="label")
    elif model_type == "random_forest":
        model = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(stages=indexers + encoders + [imputer, vector_assembler, model])
    
    return pipeline


def train_and_evaluate_model(pipeline, train_data, test_data, model_path, model_name):
    """
    Train and evaluate the model
    
    Args:
        pipeline (Pipeline): ML Pipeline
        train_data (DataFrame): Training data
        test_data (DataFrame): Test data
        model_path (str): Path to save the model
        model_name (str): Name of the model for saving
        
    Returns:
        tuple: (trained_model, auc_score)
    """
    print(f"\n--- Training and Evaluating {model_name} ---")
    
    # Create full model save path
    full_model_path = os.path.join(model_path, model_name)
    
    # Train and save the model
    if not os.path.exists(full_model_path):
        model = pipeline.fit(train_data)
        model.write().overwrite().save(full_model_path)
        print(f"Successfully saved trained model to '{full_model_path}'.")
    else:
        print(f"Model already exists at '{full_model_path}'. Loading model.")
        model = PipelineModel.load(full_model_path)

    # Make predictions and evaluate
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    print(f"Area Under ROC (AUC) on Test Data: {auc:.4f}")
    
    return model, auc


def get_feature_importance(model, categorical_cols, numerical_cols):
    """
    Extract and display feature importance for Random Forest models
    
    Args:
        model (PipelineModel): Trained model pipeline
        categorical_cols (list): List of categorical column names
        numerical_cols (list): List of numerical column names
        
    Returns:
        pd.DataFrame: Feature importance DataFrame
    """
    # Check if the model is Random Forest
    rf_model = model.stages[-1]
    if not hasattr(rf_model, 'featureImportances'):
        print("Feature importance not available for this model type.")
        return None
    
    print("\n--- Top 10 Most Important Features ---")
    
    # Get the list of feature names from the VectorAssembler stage
    assembler = model.stages[-2]
    feature_names = assembler.getInputCols()

    # Create a pandas DataFrame for better visualization
    importances = rf_model.featureImportances.toArray()
    feature_importance_df = pd.DataFrame(
        list(zip(feature_names, importances)), 
        columns=['feature', 'importance']
    )
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print(feature_importance_df.head(10))
    return feature_importance_df


def check_model_exists(model_path):
    """
    Check if model already exists
    
    Args:
        model_path (str): Path to model
        
    Returns:
        bool: True if exists, False otherwise
    """
    return os.path.exists(model_path)