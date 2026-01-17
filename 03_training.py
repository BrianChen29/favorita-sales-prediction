import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OrdinalEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
INPUT_PATH = Path("data/processed/train_features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load the feature-engineered dataset.
    """
    print(f"Loading data from {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)
    return df

def train_model(df: pl.DataFrame):
    """
    Train XGBoost Regressor with Time-Series Split logic.
    """
    print("Preparing data for training...")
    
    # 1. Separate Train and Test sets (based on the 'split' column)
    # Train set: Used for training and validation
    # Test set: The future data we need to predict (Submission)
    df_train_full = df.filter(pl.col("split") == "train")
    df_submission = df.filter(pl.col("split") == "test")
    
    # 2. Time-Series Split for Validation
    # Strategy: Use the last 30 days of the training set as a Validation Set
    max_date = df_train_full["date"].max()
    val_start_date = max_date - np.timedelta64(30, 'D')
    
    print(f"Training Data End Date: {max_date}")
    print(f"Validation Start Date: {val_start_date}")
    
    train_mask = df_train_full["date"] < val_start_date
    val_mask = df_train_full["date"] >= val_start_date
    
    X_train = df_train_full.filter(train_mask)
    X_val = df_train_full.filter(val_mask)
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")
    
    # 3. Define Feature and Target
    # Drop non-feature columns
    exlude_cols = ["id", "date", "sales", "split"]
    features = [c for c in df_train_full.columns if c not in exlude_cols]
    target = "sales"
    
    print(f"Training with {len(features)} features: {features}")
    
    # 4. Handle Categorical Columns (XGBoost needs numeric input)
    # Use Ordinal Encoding for high-cardinality categories like 'family', 'city'
    cat_cols = ["family", "store_nbr", "city", "state", "type", "cluster"]
    
    # Simple Ordinal Encoding using Pandas for compatibility with Sklearn/XGBoost
    X_train_pd = X_train.select(features).to_pandas()
    y_train = X_train.select(target).to_pandas().values.ravel()
    
    X_val_pd = X_val.select(features).to_pandas()
    y_val = X_val.select(target).to_pandas().values.ravel()
    
    # Encode categoricals
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_pd[cat_cols] = encoder.fit_transform(X_train_pd[cat_cols])
    X_val_pd[cat_cols] = encoder.transform(X_val_pd[cat_cols])
    
    # 5. Train XGBoost
    # Using RMSLE objective (reg:squaredlogerror)
    # Standard RMSE on log-transformed target for stability
    print("Training XGBoost Model...")
    # Log-transform target: log(1 + y) to avoid issues with zero sales
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    model = xgb.XGBRegressor(
        n_estimators=1000,  # Number of trees
        learning_rate=0.05,  # Step size
        max_depth=8,  # Tree depth
        subsample=0.8,  # Row Sampling
        colsample_bytree=0.8,  # Feature sampling
        early_stopping_rounds=50,  # Stop if validation score doesn't improve
        n_jobs=1,
        random_state=42
    )
    
    model.fit(
        X_train_pd, y_train_log,
        eval_set=[(X_train_pd, y_train_log), (X_val_pd, y_val_log)],
        verbose=100
    )
    
    # 6. Evaluation
    print("Evaluating Model...")
    preds_log = model.predict(X_val_pd)
    # Inverse log transform to get actual sales prediction: exp(y) - 1
    preds = np.expm1(preds_log)
    
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
    print(f"Validation RMSLE: {rmsle: .4f}")
    
    # 7. Feature Importance Plot
    plot_importance(model, features)
    
    # 8. Save Model
    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
    print(f"Model Saved to {MODEL_DIR / 'xgboost_model.pkl'}")
    
def plot_importance(model, feature_names):
    """
    Plot Feature Importance to understand what drives sales.
    """
    plt.figure(figsize=(10, 8))
    # XGBoost provides importance scores
    importance = model.feature_importances_
    # Sort features
    indices = np.argsort(importance)[::-1]
    
    plt.title("Feature Importance (XGBoost)")
    plt.barh(range(len(indices)), importance[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png")
    print(f"Feature importance plot saved to {MODEL_DIR / 'feature_importance.png'}")
    plt.show()
    
if __name__ == "__main__":
    df = load_data()
    train_model(df)