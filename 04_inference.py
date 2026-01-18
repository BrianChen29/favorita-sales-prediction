import polars as pl
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path

# --- Configuration ---
INPUT_PATH = Path("data/processed/train_features.parquet")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

def load_artifacts():
    """
    Load the trained model and the categorical encoder.
    """
    print("Loading model and artifacts...")
    model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    return model, encoder

def make_predictions():
    """
    prediction process
    """
    # 1. Load Data (Filter for Test set only)
    print("Loading test data...")
    df = pl.read_parquet(INPUT_PATH)
    df_test = df.filter(pl.col("split") == "test")
    
    print(f"Test Data Shape: {df_test.shape}")
    
    # 2. Prepare Features
    exclude_cols = ["id", "date", "sales", "split"]
    features = [c for c in df_test.columns if c not in exclude_cols]  # match the training features
    cat_cols = ["family", "store_nbr", "city", "state", "type", "cluster"]
    
    # Convert to Pandas for encoding/inference (compatible with training pipeline)
    X_test_pd = df_test.select(features).to_pandas()
    ids = df_test["id"].to_numpy()  # Keep IDs for submission file
    
    # 3. Load Model & Apply Encoding
    model, encoder = load_artifacts()
    
    print("Encoding categorical features...")
    # Transforming test data using the SAME mapping as training
    X_test_pd[cat_cols] = encoder.transform(X_test_pd[cat_cols])
    
    # 4. Prediction
    print("Running inference...")
    preds_log = model.predict(X_test_pd)
    
    # Inverse Log Transformation (exp(x) - 1)
    # (Train on log1p(sales))
    preds = np.expm1(preds_log)
    
    # Clip negative predictions (Sales cannot be negative)
    preds = np.maximum(preds, 0)
    
    # 5. Create Submission File
    print("Creating submission file...")
    submission = pl.DataFrame({
        "id": ids,
        "sales": preds
    })
    
    # Save
    submission.write_csv(SUBMISSION_PATH)
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print(submission.head())
    
if __name__ == "__main__":
    make_predictions()