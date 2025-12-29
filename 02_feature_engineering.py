import polars as pl
import numpy as np
from pathlib import Path
import gc

# --- Configuration ---
DATA_DIR = Path("data/raw")
TRAIN_DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_merge_data():
    """
    Load processed train data and raw test data.
    Strategy: Concatenate Train and Test to allow Lag features to flow continously across the boundary.
    """
    print("Loading datasets...")
    
    # Load Train (output of 01)
    df_train = pl.read_parquet(TRAIN_DATA_DIR / "train_initial.parquet")
    
    # Load Test (raw) and align columns
    df_test = pl.read_csv(DATA_DIR / "test.csv").with_columns(
        pl.col("date").str.strptime(pl.Date),
        pl.lit(None).cast(pl.Float64).alias("sales"), 
        pl.lit("test").alias("split")
    )
    
    # join Oil and Stores to Test set
    oil = pl.scan_csv(DATA_DIR / "oil.csv").with_columns(pl.col("date").str.strpyime(pl.Date)).collect()
    oil = oil.with_columns(pl.col("dcoilwtico").interpolate().fill_null(strategy="backward").alias("oil_price")).drop("dcoilwtico")
    
    stores = pl.read_csv(DATA_DIR / "stores.csv")
    
    df_test = df_test.join(oil, on="date", how="left").join(stores, on="store_nbr", how="left")
    
    # Add split column to train for identification
    df_train = df_train.with_columns(pl.lit("train").alias("split"))
    
    # Concatenate
    common_cols = [c for c in df_train.columns if c in df_test.columns]
    
    print(f"Stacking Train ({df_train.shape[0]}) and Test ({df_test.shape[0]})...")
    df_full = pl.concat([
        df_train.select(common_cols),
        df_test.select(common_cols)
    ], how="diagonal")
    
    return df_full

def create_date_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract cyclic date features.
    """
    print("Engineering Date Features")
    