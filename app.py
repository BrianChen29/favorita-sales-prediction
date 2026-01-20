from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import polars as pl
import joblib
import numpy as np
import os
from mangum import Mangum

app = FastAPI(title="Favorita Sales Forecaster")

# Load Models Globally
MODEL_PATH = "models/xgboost_model.pkl"
ENCODER_PATH = "models/encoder.pkl"

model = None
encoder = None

try:
    if os.path.exists(MODEL_PATH):
        print("Loading model and encoder...")
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Model loaded succesfully.")
    else:
        print("Warning: Model files not found. API will not work correctly.")
except Exception as e:
    print(f"Error loading model: {e}")
    
# Define Schema
class PredictionInput(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    # key features
    lag_16: float
    lag_30: float
    rolling_mean_30: float
    rolling_mean_60: float
    
@app.get("/")
def root():
    """Health Check"""
    return {"status": "ok", "message": "Favorita Sales Forecasting API is ready!"}

@app.post("/predict")
def predict(data: PredictionInput):
    """
    Main Prediction Endpoint
    """
    if not model or not encoder:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    # Prepare data -> DataFrame
    input_data = {
        "store_nbr": [data.store_nbr],
        "family": [data.family],
        "onpromotion": [data.onpromotion],
        "lag_16": [data.lag_16],
        "lag_30": [data.lag_30],
        "rolling_mean_30": [data.rolling_mean_30],
        "rolling_mean_60": [data.rolling_mean_60],
        # other features with default value
        "city": ["Quito"],
        "state": ["Pichincha"],
        "type": ["D"],
        "cluster": [13],
        "month": [1],
        "day_of_week": [1],
        "day_of_month": [15],
        "is_payday": [1],
        "is_holiday": [0],
        "is_earthquake": [0],
        "oil_price": [50.0],
        "lag_60": [0.0]
    }
    
    df = pl.DataFrame(input_data)
    
    # Preprocessing (Encoding)
    df_pd = df.to_pandas()
    
    cat_cols = ["family", "store_nbr", "city", "state", "type", "cluster"]
    try:
        # Use the pre-trained Encoder to transform features
        df_pd[cat_cols] = encoder.transform(df_pd[cat_cols])
    except Exception as e:
        print(f"Encoding warning: {e}")
        pass
    
    # Execute prediction
    try:
        pred_log = model.predict(df_pd[model.feature_names_in_])
        
        # resume log (expm1)
        pred_value = np.expm1(pred_log)[0]
        
        # deal with negative values
        pred_value = max(0.0, float(pred_value))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    return {
        "prediction": pred_value,
        "input_info": {
            "store": data.store_nbr,
            "family": data.family
        }
    }
    
# --- AWS Lambda Adapter ---
handler = Mangum(app)