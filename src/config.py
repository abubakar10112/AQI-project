"""
Pearls AQI Predictor — Configuration Module

Centralized configuration for the AQI prediction system targeting Lahore, Pakistan.
All API endpoints, feature definitions, model parameters, and AQI thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for d in [FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API Configuration
# =============================================================================
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN", "")
AQICN_BASE_URL = "https://api.waqi.info"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_WEATHER_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# =============================================================================
# Target City Configuration — Lahore, Pakistan
# =============================================================================
CITY_NAME = "Lahore"
CITY_COUNTRY = "Pakistan"
CITY_LATITUDE = 31.5497
CITY_LONGITUDE = 74.3436
CITY_TIMEZONE = "Asia/Karachi"

# AQICN station identifiers for Lahore
AQICN_CITY_FEED = "lahore"
AQICN_STATIONS = [
    {"name": "US Consulate Lahore", "id": "@8539"},
    {"name": "Lahore", "id": "lahore"},
]

# =============================================================================
# Feature Store Configuration
# =============================================================================
FEATURE_STORE_BACKEND = os.getenv("FEATURE_STORE_BACKEND", "local")  # "local" or "hopsworks"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi_predictor")

# =============================================================================
# Feature Definitions
# =============================================================================

# Weather features from Open-Meteo
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

# Air quality / pollutant features
POLLUTANT_FEATURES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "carbon_monoxide",
]

# Time-based features (computed, not fetched)
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "season",
]

# Lahore-specific features (computed)
LAHORE_FEATURES = [
    "is_smog_season",        # Oct-Feb
    "is_crop_burning_season", # Oct-Nov
    "is_brick_kiln_active",   # Nov-Mar
    "wind_from_east",         # Cross-border pollution indicator
    "days_since_rain",        # Rain cleans the air
]

# Derived / lag features (computed from historical data)
DERIVED_FEATURES = [
    "aqi_change_rate",       # AQI change over last 3 hours
    "aqi_rolling_mean_24h",  # 24-hour rolling average
    "aqi_rolling_std_24h",   # 24-hour rolling std dev
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "temp_x_humidity",       # Interaction feature
    "wind_x_pm25",           # Interaction feature
]

# All model input features
ALL_FEATURES = (
    WEATHER_FEATURES
    + POLLUTANT_FEATURES
    + TIME_FEATURES
    + LAHORE_FEATURES
    + DERIVED_FEATURES
)

# Target variable
TARGET = "us_aqi"

# =============================================================================
# Model Configuration
# =============================================================================
FORECAST_HOURS = 72  # 3 days
LOOKBACK_HOURS = 48  # Input window for sequence models

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

RIDGE_PARAMS = {
    "alpha": 1.0,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

TENSORFLOW_PARAMS = {
    "lstm_units": [64, 64],
    "dense_units": 128,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10,  # Early stopping
}

# Model fallback chain order (primary → fallback → emergency)
MODEL_FALLBACK_CHAIN = [
    "xgboost",
    "random_forest",
    "ridge",
]

# =============================================================================
# AQI Categories — US EPA Standard (used in Lahore / Pakistan)
# =============================================================================
AQI_CATEGORIES = [
    {"min": 0, "max": 50, "label": "Good", "color": "#00e400", "emoji": "🟢"},
    {"min": 51, "max": 100, "label": "Moderate", "color": "#ffff00", "emoji": "🟡"},
    {"min": 101, "max": 150, "label": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "emoji": "🟠"},
    {"min": 151, "max": 200, "label": "Unhealthy", "color": "#ff0000", "emoji": "🔴"},
    {"min": 201, "max": 300, "label": "Very Unhealthy", "color": "#8f3f97", "emoji": "🟣"},
    {"min": 301, "max": 500, "label": "Hazardous", "color": "#7e0023", "emoji": "🟤"},
]

AQI_HEALTH_ADVISORIES = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
    "Moderate": "Acceptable air quality. Sensitive individuals should limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, elderly, and those with respiratory conditions should stay indoors.",
    "Unhealthy": "Everyone should reduce outdoor activity. Wear masks if going outside.",
    "Very Unhealthy": "Health alert! Avoid all outdoor activity. Keep windows closed.",
    "Hazardous": "EMERGENCY: Stay indoors. Use air purifiers. Seek medical attention if experiencing symptoms.",
}

# Alert threshold — AQI above this triggers a hazardous alert
ALERT_THRESHOLD = 200

# =============================================================================
# Backfill Configuration
# =============================================================================
BACKFILL_START_DATE = "2023-01-01"
BACKFILL_END_DATE = "2025-08-26"  # Yesterday
BACKFILL_CHUNK_DAYS = 30  # Process in monthly chunks to avoid API limits

# =============================================================================
# Lahore Seasonal Dates (approximate)
# =============================================================================
SMOG_SEASON_MONTHS = [10, 11, 12, 1, 2]          # October through February
CROP_BURNING_MONTHS = [10, 11]                     # October–November
BRICK_KILN_MONTHS = [11, 12, 1, 2, 3]             # November through March


def get_aqi_category(aqi_value: float) -> dict:
    """Return the AQI category dict for a given AQI value."""
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_value <= cat["max"]:
            return cat
    # Above 500 — still hazardous
    return AQI_CATEGORIES[-1]


def get_health_advisory(aqi_value: float) -> str:
    """Return the health advisory string for a given AQI value."""
    category = get_aqi_category(aqi_value)
    return AQI_HEALTH_ADVISORIES.get(category["label"], "")
