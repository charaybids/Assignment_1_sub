#!/usr/bin/env python3
"""
Gold Layer Pipeline - Feature Engineering and Label Store Creation
"""
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.spark_utils import create_spark_session
from utils.gold_utils import create_label_store, create_gold_features
from utils.config import PREDICTION_MOB, OBSERVATION_MOB, LABEL_STRATEGY, OVERDUE_THRESHOLD


def main():
    """Main function for gold pipeline"""
    print("\n" + "="*80)
    print("GOLD LAYER PIPELINE")
    print("="*80 + "\n")
    
    # Create Spark session
    spark = create_spark_session(
        app_name="DefaultPredictionPipeline_Gold",
        master="local[*]",
        driver_memory="16g",
        log_level="ERROR"
    )
    
    try:
        # Define paths
        silver_path = "datamart/silver"
        gold_path = "datamart/gold"
        label_store_path = f"{gold_path}/label_store"
        features_path = f"{gold_path}/features"
        
        # Create gold directory if it doesn't exist
        os.makedirs(gold_path, exist_ok=True)
        
        # Load silver layer data
        print("Loading silver layer data...")
        loan_daily = spark.read.parquet(f"{silver_path}/loan_daily")
        print(f"  ✓ Loaded {loan_daily.count():,} loan records\n")
        
        # Create Label Store
        print("Creating label store...")
        label_store_df = create_label_store(
            loan_daily_df=loan_daily,
            prediction_mob=PREDICTION_MOB,
            observation_mob=OBSERVATION_MOB,
            label_strategy=LABEL_STRATEGY,
            overdue_threshold=OVERDUE_THRESHOLD
        )
        
        # Save label store
        label_store_df.write.mode("overwrite").parquet(label_store_path)
        print(f"  ✓ Saved label store to '{label_store_path}'\n")
        print(f"  Label store: {label_store_df.count():,} customers")
        label_store_df.groupBy('label').count().show()
        
        # Create Gold Features
        print("\nCreating gold features...")
        gold_features_df = create_gold_features(
            silver_path=silver_path,
            label_store_df=label_store_df,
            spark_session=spark,
            prediction_mob=PREDICTION_MOB,
            analyze_clickstream=True,   # Enable automated feature selection
            top_n_clickstream=1          # Use only fe_10 (top feature)
        )
        
        # Save gold features
        gold_features_df.write.mode("overwrite").parquet(features_path)
        print(f"  ✓ Saved gold features to '{features_path}'\n")
        
        # Show sample
        print("Sample of Final Model Data:")
        print(f"Shape: {gold_features_df.count():,} rows × {len(gold_features_df.columns)} columns")
        gold_features_df.show(5, truncate=False)
        
        print("\n" + "="*80)
        print("GOLD PIPELINE COMPLETE!")
        print("="*80 + "\n")
            
    except Exception as e:
        print(f"\n❌ Error in gold pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()
        print("Spark session stopped.")


if __name__ == "__main__":
    main()