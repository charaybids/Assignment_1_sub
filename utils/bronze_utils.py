"""
Bronze layer utilities: Raw CSV to Parquet ingestion only.

Bronze layer philosophy:
- Raw data ingestion with NO transformations
- Store data exactly as received from source
- Simple CSV → Parquet conversion for efficiency
- Display .info() and .describe() for validation
"""
import os
import pandas as pd


def ingest_bronze_data(file_paths, output_path):
    """
    Raw data ingestion: Read CSV and write Parquet with no transformations.
    
    Args:
        file_paths (dict): Mapping of dataset name -> CSV path
        output_path (str): Root bronze output directory
    """
    print("=" * 80)
    print("Bronze Layer: Raw Data Ingestion (No Transformations)")
    print("=" * 80)
    
    os.makedirs(output_path, exist_ok=True)
    
    for name, csv_path in file_paths.items():
        print(f"\n{'─' * 80}")
        print(f"Processing: {name}")
        print(f"{'─' * 80}")
        
        try:
            # Read CSV as-is (no parsing, no transformations)
            df = pd.read_csv(csv_path, dtype=str)  # Keep everything as strings
            print(f"✓ Loaded {len(df):,} rows from {csv_path}")
            
            # Display DataFrame info
            print(f"\n📊 DataFrame Info for '{name}':")
            print("─" * 40)
            df.info()
            
            # Display descriptive statistics (will show string stats)
            print(f"\n📈 Basic Statistics for '{name}':")
            print("─" * 40)
            print(df.describe(include='all'))
            
            # Write to Parquet (raw format)
            output_file = os.path.join(output_path, f'{name}.parquet')
            df.to_parquet(output_file, engine='pyarrow', index=False)
            print(f"\n✓ Saved to: {output_file}")
            
        except Exception as e:
            print(f"✗ Error processing '{csv_path}': {e}")
            raise
    
    print(f"\n{'=' * 80}")
    print("Bronze Layer Complete! (Raw data stored)")
    print(f"{'=' * 80}\n")


def check_bronze_exists(bronze_path):
    """
    Return True if the bronze layer directory contains .parquet files.
    """
    if not os.path.isdir(bronze_path):
        return False
    # Check for any .parquet files
    parquet_files = [f for f in os.listdir(bronze_path) if f.endswith('.parquet')]
    return len(parquet_files) > 0