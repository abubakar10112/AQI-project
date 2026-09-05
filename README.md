# 🌍 Lahore AQI Predictor

**Predict the Air Quality Index (AQI) in Lahore, Pakistan for the next 3 days using a 100% serverless ML stack.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b)
![Supabase](https://img.shields.io/badge/Supabase-Feature_Store-3ecf8e)

---

## 📋 Project Overview

An end-to-end machine learning pipeline for AQI forecasting featuring:

- **Feature Pipeline** — Automated hourly data collection from AQICN & Open-Meteo APIs
- **Training Pipeline** — 3 ML models (Ridge, Random Forest, XGBoost) with daily retraining
- **Model Fallback Chain** — If one model fails, the system automatically cascades to the next
- **Interactive Dashboard** — Streamlit + Flask web app with real-time predictions & alerts
- **Explainability** — SHAP feature importance analysis
- **CI/CD** — GitHub Actions for automated pipeline execution
- **Lahore-Specific Features** — Smog season, crop burning, brick kiln activity detection
- **Supabase Feature Store** — Managed PostgreSQL for feature storage & retrieval
- **Hopsworks Model Registry** — Versioned model storage & deployment

## 🏗️ Architecture

```
Data Sources (AQICN + Open-Meteo)
        ↓
Feature Pipeline (fetch → engineer → store)
        ↓
Feature Store (Supabase PostgreSQL / Free Serverless DB)
        ↓
Training Pipeline (Ridge, RF, XGBoost → evaluate → register)
        ↓
Model Registry (Hopsworks Cloud)
        ↓
Inference Engine (Fallback Chain: XGBoost → RF → Ridge → Last Known)
        ↓
Web Dashboard (Flask API + Streamlit UI)
```

### 💡 Architectural Rationale: Why Supabase for Feature Store?

1. **Hopsworks Free Tier Limits & Paywalls**: Hopsworks Cloud enforces strict storage and credit limits on free accounts. Continuous 24/7 hourly ingestion runs (720 writes/month) quickly exhaust free quotas, resulting in `402 Payment Required` or suspended feature ingestion unless upgraded to expensive enterprise tiers.
2. **Hibernation Cold Starts & 504 Timeouts**: Free Hopsworks shared clusters hibernate during inactive periods. Waking up requires 45–90 seconds, causing serverless runners (Streamlit Cloud and GitHub Actions) to fail with `504 Gateway Timeouts`.
3. **Heavy PySpark / JVM Bloat**: The Hopsworks Feature Store client (`hsfs`) relies on complex PySpark and Java dependencies that frequently cause binary conflicts and out-of-memory errors on lightweight serverless containers.
4. **The Supabase Advantage**: Supabase provides a 100% free serverless PostgreSQL database with 500 MB storage, instant sub-80ms PostgREST HTTP queries, atomic primary key `(timestamp, city)` deduplication, and zero JVM/Java requirements.
5. **Separation of Responsibilities**: Hopsworks is retained exclusively for its greatest strength without paywall risk—the **Model Registry** (`hsml`) for versioning models and metrics—while Supabase provides rock-solid, high-frequency feature storage.

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

# Edit .env with your credentials:
# AQICN_API_TOKEN=your_token_here       (get from https://aqicn.org/data-platform/token/)
# SUPABASE_URL=https://xxx.supabase.co  (get from https://supabase.com/dashboard)
# SUPABASE_KEY=your_service_role_key
# HOPSWORKS_API_KEY=your_hopsworks_key  (get from https://app.hopsworks.ai)
# HOPSWORKS_PROJECT_NAME=your_project
```

### 3. Create Supabase Table

Run the SQL in `create_supabase_table.sql` in your Supabase SQL Editor.

### 4. Backfill Historical Data

```bash
python -m src.feature_pipeline.backfill
```

This fetches ~2 years of historical weather & air quality data for Lahore from Open-Meteo (free, no API key needed) and stores it in Supabase.

### 5. Train Models

```bash
python -m src.training_pipeline.trainer
```

Trains all 3 models and registers the best one in Hopsworks.

### 6. Launch Dashboard

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
│   │   ├── feature_store.py     # Supabase feature store adapter
│   │   └── backfill.py          # Historical data backfill
│   ├── training_pipeline/       # Model training & evaluation
│   │   ├── models/
│   │   │   ├── ridge_regression.py
│   │   │   ├── random_forest.py
│   │   │   └── xgboost_model.py
│   │   ├── trainer.py           # Training orchestrator
│   │   ├── evaluator.py         # RMSE, MAE, R² metrics
│   │   └── model_registry.py    # Hopsworks model versioning
│   ├── inference/
│   │   └── predictor.py         # Fallback chain + 3-day forecast
│   └── analytics/
│       ├── eda.py               # Exploratory Data Analysis
│       └── explainability.py    # SHAP feature importance
├── app/
│   ├── flask_api.py             # REST API (7 endpoints)
│   └── streamlit_app.py         # Interactive dashboard
├── data/                        # Deprecated local artifacts; not used at runtime
├── reports/                     # Project documentation & evaluation results
│   └── FINAL_REPORT.md          # Comprehensive technical report
├── tests/                       # Unit tests
├── create_supabase_table.sql    # Supabase table DDL
├── FINAL_REPORT.pdf             # Compiled final technical project report
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

### Model Fallback Chain

```
XGBoost → Random Forest → Ridge → Last Known AQI
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
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_HOST`
- `HOPSWORKS_PROJECT_NAME`

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

## 📜 License

This project is for educational purposes.

## 🙏 Data Sources

- [AQICN / WAQI](https://aqicn.org/) — Real-time air quality data
- [Open-Meteo](https://open-meteo.com/) — Weather & historical air quality data (free, no API key)
