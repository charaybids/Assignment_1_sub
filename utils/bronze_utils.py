"""
Bronze layer utilities: ingest CSVs with pandas and write Parquet partitioned by year/month.

Assumptions:
- snapshot_date (and loan_start_date when present) are in dd/mm/yyyy format.
"""
import os
import shutil
import pandas as pd


def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    s = df[date_col].astype(str).str.strip()
    # Strictly parse dd/mm/yyyy; raises no errors, coerces invalid to NaT
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

def ingest_and_partition_bronze_data(file_paths, output_path, overwrite: bool = True):
    """
    Ingest raw CSV files with pandas and write them partitioned by year and month as Parquet.

    Args:
        file_paths (dict): Mapping of dataset name -> CSV path
        output_path (str): Root bronze output directory
        overwrite (bool): If True, delete existing dataset folder before writing (full refresh). If False, keep
                          existing folders; only overwrite the specific partition files we write.
    """

    print("--- Starting Bronze Pipeline (pandas): Ingesting Raw Data ---")

    os.makedirs(output_path, exist_ok=True)

    for name, path in file_paths.items():
        try:
            # Read CSV with pandas
            df = pd.read_csv(path)

            # Drop any pre-existing partition columns from source
            for c in ["year", "month"]:
                if c in df.columns:
                    df = df.drop(columns=[c])

            # Parse dates
            df["snapshot_date"] = _parse_dates(df, "snapshot_date")
            if "loan_start_date" in df.columns:
                df["loan_start_date"] = _parse_dates(df, "loan_start_date")

            # Derive partitions
            df["year"] = df["snapshot_date"].dt.year
            df["month"] = df["snapshot_date"].dt.month

            # Avoid writing NaT-partitions
            valid = df.dropna(subset=["year", "month"]).copy()

            # Log distribution
            try:
                dist = (
                    df.groupby(["year", "month"], dropna=True)
                      .size()
                      .reset_index(name="count")
                      .sort_values(["year", "month"])
                )
                print(f"Partition month distribution for '{name}':\n{dist.to_string(index=False)}")
            except Exception as log_err:
                print(f"[WARN] Could not display month distribution for '{name}': {log_err}")

            # Prepare output dataset path
            dataset_out = os.path.join(output_path, name)
            os.makedirs(dataset_out, exist_ok=True)

            # Targeted deletion: if overwrite=True, remove only the partitions (year, month)
            # that appear in this batch instead of deleting the entire dataset folder.
            if overwrite:
                unique_parts = (
                    valid[["year", "month"]]
                    .dropna()
                    .drop_duplicates()
                    .astype({"year": int, "month": int})
                    .itertuples(index=False, name=None)
                )
                removed = 0
                for y, m in unique_parts:
                    part_dir = os.path.join(dataset_out, f"year={y}", f"month={m}")
                    if os.path.isdir(part_dir):
                        shutil.rmtree(part_dir)
                        removed += 1
                if removed:
                    print(f"Removed {removed} existing partitions for '{name}' prior to write.")

            # Write per-partition parquet files using pyarrow
            # Directory layout: <output>/<name>/year=YYYY/month=M/part-00000.parquet
            engine = "pyarrow"
            # Avoid writing NaT-partitions
            valid = df.dropna(subset=["year", "month"]).copy()
            for (y, m), part in valid.groupby(["year", "month"], sort=True):
                part_dir = os.path.join(dataset_out, f"year={int(y)}", f"month={int(m)}")
                os.makedirs(part_dir, exist_ok=True)
                part_out = os.path.join(part_dir, "part-00000.parquet")
                # Do not include partition columns in file body (Spark will add them from path)
                cols = [c for c in part.columns if c not in ("year", "month")]
                out_df = part[cols].copy()
                # Ensure date columns are stored as DATE (not TIMESTAMP[ns]) for Spark compatibility
                for dc in ["snapshot_date", "loan_start_date"]:
                    if dc in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df[dc]):
                        out_df[dc] = out_df[dc].dt.date
                out_df.to_parquet(part_out, engine=engine, index=False)

            print(f"Successfully ingested and partitioned '{name}' -> {dataset_out}")

        except Exception as e:
            print(f"Error processing file '{path}': {e}")

    print("--- Bronze Pipeline Complete ---")


def check_bronze_exists(bronze_path):
    """
    Return True only if the bronze layer directory contains data files
    (not just an empty folder structure).
    """
    if not os.path.isdir(bronze_path):
        return False
    for _root, _dirs, files in os.walk(bronze_path):
        if files:
            return True
    return False