# favorita-sales-prediction
## Favorita Store Sales Forecasting: End-to-End MLE Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Polars](https://img.shields.io/badge/Polars-Fast_ETL-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

A production-ready Machine Learning pipeline to forecast daily sales for thousands of product families at Favorita stores in Ecuador.

This project demonstrates a shift from traditional analysis to **Machine Learning Engineering (MLE)** practices, focusing on modular code, high-performance data processing (Polars), and reproducible API deployment (Docker + FastAPI).

## Project Overview

* **Goal**: Minimize RMSLE (Root Mean Squared Logarithmic Error) for 16-day future sales forecasting.
* **Performance**: Achieved **Validation RMSLE: 0.4155** and Top 15% zone on Kaggle.
* **Key Challenge**: Handling complex seasonality, local/national holidays, and exogenous shocks (2016 Earthquake).

## Datasets
[Kaggle: Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

## Tech Stack & Engineering Decisions

* **Data Processing**: Used **Polars** instead of Pandas for 50x faster ETL on 3M+ rows and lazy execution capabilities.
* **Modeling**: **XGBoost Regressor** with Log-Loss objective (`reg:squaredlogerror` proxy) to handle long-tail sales distribution.
* **Feature Engineering**:
    * **Lag-16 Strategy**: Designed features strictly based on the 16-day forecast horizon to prevent data leakage.
    * **Context-Aware Holidays**: logic to map holidays strictly to their specific cities/states.
    * **Rolling Statistics**: Rolling mean/std over 30/60 days to capture long-term trends.
* **MLOps & Deployment**:
    * **FastAPI**: Serving predictions via a high-performance REST API.
    * **Docker**: Containerized environment ensuring reproducibility from local dev to cloud (AWS Lambda ready).
    * **Artifact Serialization**: Managed model/encoder lifecycle using `joblib`.

## Feature Importance

The model identifies **long-term trends (Rolling Means)** and **short-term history (Lags)** as the strongest predictors, aligning with retail business intuition.

![Feature Importance](models/feature_importance.png)

## 📂 Project Structure

```bash
.
├── models/             # Trained artifacts (model.pkl, encoder.pkl)
├── src/                # (Optional) Helper modules
├── app.py              # FastAPI Application (Entry Point)
├── 01_preprocessing.py # ETL: Cleaning & Merging (Polars)
├── 02_feature_engineering.py # Lags, Rolling, Holiday Logic
├── 03_training.py      # XGBoost Training & Validation
├── 04_inference.py     # Batch Inference Script
├── Dockerfile          # Reproducible Environment (API Server)
└── requirements.txt
```

## How to Run

### Method 1: Local Training Pipeline (Reproduce Model)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline step-by-step
python 01_preprocessing.py
python 02_feature_engineering.py
python 03_training.py
python 04_inference.py
```

### Method 2: Dockerized API (Inference Demo)
Build and run the containerized REST API server.

```bash
# Build the Docker Image
docker build -t favorita-api .

# Run the Container
# Maps host port 8080 to container port 8080
docker run -p 8080:8080 favorita-api
```

### Access the API

Once the container is running, access the Interactive Swagger UI to make predictions:

**👉 http://localhost:8080/docs**

1. Click on ```POST /predict```
2. Click **Try it out**
3. Enter feature values (ensure lags/rolling means are not 0)
4. Click **Execute** to see the sales forecast.

## Results

* Validation RMSLE: 0.4155
* Kaggle Score: 0.49017
