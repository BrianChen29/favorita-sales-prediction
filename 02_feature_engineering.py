import polars as pl
import numpy as np
from pathlib import Path
import gc

# --- Configuration ---
DATA_DIR = Path("data/raw")
TRAIN_DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")
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
    oil = pl.scan_csv(DATA_DIR / "oil.csv").with_columns(pl.col("date").str.strptime(pl.Date)).collect()
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
    df = df.with_columns([
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.day().alias("day_of_month"),
        
        # Payroll Logic: 15th and End-of-Month are typical paydays in Ecuador
        ((pl.col("date").dt.day() == 15) | (pl.col("date").dt.day() == pl.col("date").dt.month_end().dt.day()))
        .cast(pl.Int8).alias("is_payday")
    ])
    return df

def create_holiday_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Complex Holiday Logic:
    Mathches holiday based on 'Locale' (National vs Regional vs Local).
    This ensures a holiday in Quito doesn't affect a store in Guayaquil.
    """
    print("Engineering Holiday Features...")
    
    # Load raw holidays and remove transferred ones
    holidays = (
        pl.scan_csv(DATA_DIR / "holidays_events.csv")
        .with_columns(pl.col("date").str.strptime(pl.Date))
        .filter(pl.col("transferred") == False)
        .collect()
    )
    
    # Split holidays by locale for precise joining
    nat_hols = holidays.filter(pl.col("locale") == "National").select("date", "description").rename({"description": "nat_desc"})
    reg_hols = holidays.filter(pl.col("locale") == "Regional").select("date", "locale_name", "description").rename({"description": "reg_desc", "locale_name": "state"})
    loc_hols = holidays.filter(pl.col("locale") == "Local").select("date", "locale_name", "description").rename({"description": "loc_desc", "locale_name": "city"})
    
    # National Join
    df = df.join(nat_hols, on="date", how="left")
    
    # Regional Join
    df = df.join(reg_hols, on=["date", "state"], how="left")
    
    # Local Join
    df = df.join(loc_hols, on=["date", "city"], how="left")
    
    # Consolidate into a single boolean flag
    df = df.with_columns([
        (pl.col("nat_desc").is_not_null() | pl.col("reg_desc").is_not_null() | pl.col("loc_desc").is_not_null())
        .cast(pl.Int8).alias("is_holiday")
    ])
    
    # Cleanup text columns to save memory
    df = df.drop(["nat_desc", "reg_desc", "loc_desc"])
    
    return df

def create_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create Time-Series Features (Lags & Rolling Windows).
    CRITICAL: Data must be sorted by Entity (Store, Family) and Time (Date).
    """
    print("Engineering Lag Features (Computationally Heavy)...")
    
    # Sorting is mandatory for shift/rolling operations
    df = df.sort(["store_nbr", "family", "date"])
    
    # Lag Features (Shifted Sales)
    # We start from Lag-16 because the Test set horizon is 16 days.
    lags = [16, 30, 60]
    cols = []
    for lag in lags: cols.append(
        pl.col("sales").shift(lag).over(["store_nbr", "family"]).alias(f"lag_{lag}")
    )
    
    # Rolling Features (Moving Averages)
    # Logic: Shift(16) first to prevent data leakage, then calculate rolling mean.
    for window in [30, 60]:
        cols.append(
            pl.col("sales")
            .shift(16)
            .rolling_mean(window_size=window)
            .over(["store_nbr", "family"])
            .alias(f"rolling_mean_{window}")
        )
        
    df = df.with_columns(cols)
    return df

def main():
    df = load_and_merge_data()
    
    df = create_date_features(df)
    
    df = create_holiday_features(df)
    
    # Earthquake Feature
    print("Adding Earthquake Logic...")
    earthquake_date = pl.date(2016, 4, 16)
    df = df.with_columns(
        ((pl.col("date") >= earthquake_date) & (pl.col("date") <= earthquake_date.dt.offset_by("30d")))
        .cast(pl.Int8).alias("is_earthquake")
    )
    
    # Lag Features
    df = create_lag_features(df)
    
    # Filter & Save
    # Drop the initial rows of Train set where Lags are NaN
    print("Saving Features to Parquet...")
    
    # Drop rows where lag_60 is null (cannot train on them), but keep all test rows
    df_final = df.filter(
        (pl.col("split") == "test") | (pl.col("lag_60").is_not_null())
    )
    
    # Fill remaining NaNs with 0
    df_final = df_final.fill_nan(0)
    
    output_path = OUTPUT_DIR / "train_features.parquet"
    df_final.write_parquet(output_path)
    
    print("-" * 30)
    print(f"Phase 2 Complete! Feature Engineering Done.")
    print(f"Saved to: {output_path}")
    print(f"Final Shape: {df_final.shape}")
    print("Generated Columns:", df_final.columns)
    
if __name__ == "__main__":
    main()