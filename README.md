# 🌍 Pearls AQI Predictor — Lahore

**Predict the Air Quality Index (AQI) in Lahore, Pakistan for the next 3 days using a 100% serverless ML stack.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b)

---

## 📋 Project Overview

An end-to-end machine learning pipeline for AQI forecasting featuring:

- **Feature Pipeline** — Automated hourly data collection from AQICN & Open-Meteo APIs
- **Training Pipeline** — 4 ML models (Ridge, Random Forest, XGBoost, LSTM) with daily retraining
- **Model Fallback Chain** — If one model fails, the system automatically cascades to the next
- **Interactive Dashboard** — Streamlit + Flask web app with real-time predictions & alerts
- **Explainability** — SHAP feature importance analysis
- **CI/CD** — GitHub Actions for automated pipeline execution
- **Lahore-Specific Features** — Smog season, crop burning, brick kiln activity detection

## 🏗️ Architecture

```
Data Sources (AQICN + Open-Meteo)
        ↓
Feature Pipeline (fetch → engineer → store)
        ↓
Feature Store (Local Parquet / Hopsworks)
        ↓
Training Pipeline (Ridge, RF, XGBoost, LSTM → evaluate → register)
        ↓
Model Registry (Local / Hopsworks)
        ↓
Inference Engine (Fallback Chain: XGBoost → RF → Ridge → LSTM → Last Known)
        ↓
Web Dashboard (Flask API + Streamlit UI)
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd AQI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API keys
# AQICN_API_TOKEN=your_token_here  (get from https://aqicn.org/data-platform/token/)
```

### 3. Backfill Historical Data

```bash
python -m src.feature_pipeline.backfill
```

This fetches ~2 years of historical weather & air quality data for Lahore from Open-Meteo (free, no API key needed).

### 4. Train Models

```bash
python -m src.training_pipeline.trainer
```

Trains all 4 models and registers the best one.

### 5. Launch Dashboard

```bash
# Terminal 1: Start Flask API
python app/flask_api.py

# Terminal 2: Start Streamlit Dashboard
streamlit run app/streamlit_app.py
```

Visit `http://localhost:8501` for the dashboard.

## 📁 Project Structure

```
AQI/
├── .github/workflows/           # CI/CD pipelines
│   ├── feature_pipeline.yml     # Hourly data collection
│   └── training_pipeline.yml    # Daily model retraining
├── src/
│   ├── config.py                # Centralized configuration
│   ├── feature_pipeline/        # Data fetching & feature engineering
│   │   ├── data_fetcher.py      # AQICN + Open-Meteo API clients
│   │   ├── feature_engineer.py  # Feature computation
│   │   ├── feature_store.py     # Dual-mode: Local/Hopsworks
│   │   └── backfill.py          # Historical data backfill
│   ├── training_pipeline/       # Model training & evaluation
│   │   ├── models/
│   │   │   ├── ridge_regression.py
│   │   │   ├── random_forest.py
│   │   │   ├── xgboost_model.py
│   │   │   └── tensorflow_model.py
│   │   ├── trainer.py           # Training orchestrator
│   │   ├── evaluator.py         # RMSE, MAE, R² metrics
│   │   └── model_registry.py    # Model versioning & storage
│   ├── inference/
│   │   └── predictor.py         # Fallback chain + 3-day forecast
│   └── analytics/
│       ├── eda.py               # Exploratory Data Analysis
│       └── explainability.py    # SHAP feature importance
├── app/
│   ├── flask_api.py             # REST API (6 endpoints)
│   └── streamlit_app.py         # Interactive dashboard
├── data/                        # Local feature store (gitignored)
├── reports/                     # Generated reports & plots
│   └── report_generator.py      # PDF report generator
├── tests/                       # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## 🧠 Models

| Model | Type | Strengths |
|:--|:--|:--|
| Ridge Regression | Statistical baseline | Fast, interpretable |
| Random Forest | Ensemble | Robust, handles non-linearity |
| XGBoost | Gradient boosting | Best tabular performance |
| LSTM (TensorFlow) | Deep learning | Captures temporal sequences |

### Model Fallback Chain

```
XGBoost → Random Forest → Ridge → LSTM → Last Known AQI
```

If a model errors, returns NaN, negative AQI, or >600, the system automatically tries the next model.

## 📊 Features

| Category | Count | Examples |
|:--|:--|:--|
| Weather | 8 | Temperature, humidity, wind, pressure |
| Pollutants | 6 | PM2.5, PM10, NO₂, SO₂, O₃, CO |
| Time-based | 6 | Hour, day, month, weekend, season |
| Lahore-specific | 5 | Smog season, crop burning, brick kilns |
| Derived/Lag | 10 | Rolling stats, lag features, interactions |
| **Total** | **35** | |

## 🌡️ AQI Categories

| Range | Category | Advisory |
|:--|:--|:--|
| 0-50 | 🟢 Good | Enjoy outdoor activities |
| 51-100 | 🟡 Moderate | Sensitive groups limit exertion |
| 101-150 | 🟠 Unhealthy (Sensitive) | Children/elderly stay indoors |
| 151-200 | 🔴 Unhealthy | Reduce outdoor activity |
| 201-300 | 🟣 Very Unhealthy | Avoid all outdoor activity |
| 301-500 | 🟤 Hazardous | EMERGENCY: Stay indoors |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|:--|:--|:--|
| `/api/predict` | GET | 3-day AQI forecast |
| `/api/current` | GET | Current AQI + weather |
| `/api/history?days=30` | GET | Historical AQI data |
| `/api/models` | GET | Model performance metrics |
| `/api/explain` | GET | SHAP feature importances |
| `/api/alerts` | GET | Active hazardous alerts |
| `/api/health` | GET | Health check |

## ⚙️ CI/CD

- **Feature Pipeline**: GitHub Actions cron — every hour
- **Training Pipeline**: GitHub Actions cron — daily at 2 AM UTC (7 AM PKT)

Set these GitHub Secrets:
- `AQICN_API_TOKEN`
- `HOPSWORKS_API_KEY` (when ready)

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

## 📝 Generate Report

```bash
python reports/report_generator.py
```

Generates a comprehensive PDF report in `reports/`.

## 📜 License

This project is for educational purposes.

## 🙏 Data Sources

- [AQICN / WAQI](https://aqicn.org/) — Real-time air quality data
- [Open-Meteo](https://open-meteo.com/) — Weather & historical air quality data (free, no API key)
