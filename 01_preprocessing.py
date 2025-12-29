import polars as pl
import os
from pathlib import Path

# --- Configuration ---
# Define paths for raw input and processed output
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_oil(filepath: Path) -> pl.DataFrame:
    """
    Loads oil data and handles missing values.
    Logic: 
    Oil prices often have gaps (weekends/holidays). We use Linear Interpolation to fill these gaps to ensure we have a continuous macro feature.
    """
    print(f"Processing Oil Data from {filepath}...")
    q = (
        pl.scan_csv(filepath)
        .with_columns(pl.col("date").str.strptime(pl.Date))
        .sort("date")
    )
    
    # Collect (eager execution) to perform interpolation accuratel
    df_oil = q.collect()
    
    # Linear Interpolation: Fills missing values linearly between valid points
    # 'backward' strategy fills any leading NaNs at the start of the time series
    df_oil = df_oil.with_columns(
        pl.col("dcoilwtico").interpolate().fill_null(strategy="backward").alias("oil_price")
    ).drop("dcoilwtico")
    
    return df_oil

def load_main_data(filepath: Path) -> pl.LazyFrame:
    """
    Loads the main train dataset lazily.
    
    The training data has millions of rows. Using scan_csv allows Polars to 
    optimize the query plan and only load data when necessary (.collect()).
    """
    print(f"Loading Train Data (Lazy) from {filepath}...")
    return (
        pl.scan_csv(filepath)
        .with_columns(pl.col("date").str.strptime(pl.Date))
    )
    
def main():
    # Sanity Check: Ensure files exist
    if not (DATA_DIR / "train.csv").exists():
        print(f"Error: 'train.csv' not found in {DATA_DIR.absolute()}")
        return
    
    # 1. Process Auxiliary Data
    # Process Oil first.
    oil_df = process_oil(DATA_DIR / "oil.csv")
    stores_df = pl.read_csv(DATA_DIR / "stores.csv")
    
    # 2. Load Main Training Data
    train_lazy = load_main_data(DATA_DIR / "train.csv") 
    
    # 3. Merging Strategy
    # Merge Oil Data (Macroeconomic feature) -> left join on Date
    print("Merging Oil Data...")
    train_enriched = train_lazy.join(
        oil_df.lazy(),
        on="date",
        how="left"
    )
    
    # Merge Store Metadata (City, State, Type, Cluster)
    print("Merging Store Metadata...")
    train_enriched = train_enriched.join(
        stores_df.lazy(),
        on="store_nbr",
        how="left"
    )
    
    # 4. Save to Parquet
    print("Coleecting and Saving to Parquet...")
    
    final_df = train_enriched.collect()
    output_file = OUTPUT_DIR / "train_initial.parquet"
    
    final_df.write_parquet(output_file)
    
    print("-" * 30)
    print(f"Phase 1 Complete! Data saved to: {output_file}")
    print(f"Final Shape: {final_df.shape}")
    print("Columns:", final_df.columns)
    print("Preview:")
    print(final_df.head())
    
if __name__ == "__main__":
    main()