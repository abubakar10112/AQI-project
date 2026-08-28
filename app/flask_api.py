import time
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import Predictor
from src.feature_pipeline.data_fetcher import AQICNClient, OpenMeteoWeatherClient, OpenMeteoAirQualityClient
from src.feature_pipeline.feature_store import get_feature_store
from src.training_pipeline.model_registry import get_model_registry
import src.config as config

app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)

# Simple Dict-based Cache
CACHE = {}
CACHE_TTL = 1800  # 30 minutes in seconds

def get_from_cache(key: str):
    if key in CACHE:
        data, timestamp = CACHE[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None

def set_in_cache(key: str, data):
    CACHE[key] = (data, time.time())

# Error Handlers
@app.errorhandler(404)
def not_found_error(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

# Endpoints
@app.route('/', methods=['GET'])
def index():
    """Project root endpoint for quick health checks and routing."""
    return jsonify({
        "name": "Pearls AQI Predictor",
        "status": "ok",
        "forecast_horizon_days": 3,
        "api_endpoints": [
            "/api/health",
            "/api/predict",
            "/api/current",
            "/api/history",
            "/api/models",
            "/api/explain",
            "/api/alerts",
        ],
    })


@app.route('/api/predict', methods=['GET'])
def get_predict():
    """Returns 3-day AQI forecast from Predictor."""
    cached_data = get_from_cache('predict')
    if cached_data:
        return jsonify(cached_data)
    
    try:
        predictor = Predictor()
        data = predictor.predict_next_3_days()
        set_in_cache('predict', data)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/current', methods=['GET'])
def get_current():
    """Returns current AQI + weather and pollutant levels."""
    cached_data = get_from_cache('current')
    if cached_data:
        return jsonify(cached_data)
    
    try:
        from src.feature_pipeline.data_fetcher import fetch_all_current
        current_df = fetch_all_current()
        
        if current_df is None or current_df.empty:
            return jsonify({"error": "No current data available"}), 503
        
        # Filter for latest valid real-time AQI row
        valid_df = current_df[current_df['us_aqi'].notna()]
        if not valid_df.empty:
            current_data = valid_df.iloc[-1].to_dict()
        else:
            current_data = current_df.iloc[0].to_dict()
        
        # Use LocalFeatureStore for instant 0.01s backfill
        from src.feature_pipeline.feature_store import LocalFeatureStore
        recent_df = LocalFeatureStore().get_latest_features(n_hours=48)
        
        if recent_df is not None and not recent_df.empty:
            for col in current_data:
                if col in recent_df.columns and (pd.isna(current_data[col]) or current_data[col] is None):
                    valid_vals = recent_df[col].dropna()
                    if not valid_vals.empty:
                        current_data[col] = valid_vals.iloc[-1]
                        
        aqi = current_data.get('us_aqi')
        if pd.isna(aqi) or aqi is None:
            aqi = 75.0  # Safe default if entirely unavailable
            current_data['us_aqi'] = aqi
            
        category = config.get_aqi_category(float(aqi))
        
        current_data['category'] = category['label']
        current_data['color'] = category['color']
        current_data['emoji'] = category['emoji']
        current_data['health_advisory'] = config.get_health_advisory(float(aqi))
        
        # Convert any numpy/pandas/timestamp types to native Python for JSON
        cleaned_data = {}
        for k, v in current_data.items():
            if pd.isna(v):
                cleaned_data[k] = 0.0
            elif isinstance(v, (np.floating, float)):
                cleaned_data[k] = round(float(v), 2)
            elif isinstance(v, (np.integer, int)):
                cleaned_data[k] = int(v)
            elif isinstance(v, (pd.Timestamp, datetime)):
                cleaned_data[k] = v.isoformat()
            else:
                cleaned_data[k] = v
        
        set_in_cache('current', cleaned_data)
        return jsonify(cleaned_data)
    except Exception as e:
        logger.error(f"Fetching current data failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns historical AQI from feature store."""
    days = request.args.get('days', default=30, type=int)
    cache_key = f'history_{days}'
    
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
        
    try:
        store = get_feature_store()
        df = store.get_latest_features(n_hours=days * 24)
        if df is None or df.empty:
            return jsonify({"data": [], "message": "No historical data available"})
        # Convert to dictionary safely handling NaNs
        df_clean = df.fillna(0).copy()
        df_clean['timestamp'] = df_clean.index.strftime('%Y-%m-%d %H:%M:%S')
        data = df_clean.to_dict(orient='records')
        set_in_cache(cache_key, data)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Fetching history failed: {e}")
        return jsonify({"data": [], "message": "Could not fetch history"})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Returns model performance metrics from model registry."""
    try:
        registry = get_model_registry()
        models = registry.list_models()
        
        # Ensure all expected models from config are represented
        model_names_in_registry = {m.get('model_name') for m in models}
        for model_name in config.MODEL_FALLBACK_CHAIN:
            if model_name not in model_names_in_registry:
                models.append({
                    'model_name': model_name,
                    'version': 0,
                    'metrics': {'rmse': None, 'mae': None, 'r2': None},
                    'timestamp': None,
                })
        
        return jsonify(models)
    except Exception as e:
        logger.error(f"Fetching models metrics failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/explain', methods=['GET'])
def get_explain():
    """Returns SHAP feature importances from saved results."""
    try:
        shap_file = config.REPORTS_DIR / 'shap_summary.json'
        if shap_file.exists():
            with open(shap_file, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({"error": "Explainability data not found"}), 404
    except Exception as e:
        logger.error(f"Fetching explainability failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Returns active hazardous AQI alerts from latest prediction."""
    cached_predict = get_from_cache('predict')
    if not cached_predict:
        try:
            predictor = Predictor()
            cached_predict = predictor.predict_next_3_days()
            set_in_cache('predict', cached_predict)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    alerts = cached_predict.get('alerts', [])
    return jsonify(alerts)

@app.route('/api/health', methods=['GET'])
def get_health():
    """Health check endpoint with feature-store and model-registry status."""
    try:
        store = get_feature_store()
        store_status = store.get_feature_store_status() if hasattr(store, 'get_feature_store_status') else {
            "backend": config.FEATURE_STORE_BACKEND,
            "available": False,
            "row_count": 0,
            "latest_timestamp": None,
        }

        registry = get_model_registry()
        model_count = len(registry.list_models()) if hasattr(registry, 'list_models') else 0

        status = {
            "status": "healthy",
            "forecast_horizon_days": 3,
            "feature_store": store_status,
            "model_registry": {
                "backend": type(registry).__name__,
                "model_count": model_count,
                "available": model_count > 0,
            },
        }

        warnings = []
        if not store_status.get("available", False):
            warnings.append("Feature store has no recent data yet.")
        if model_count == 0:
            warnings.append("No trained models are registered yet.")
        if warnings:
            status["warnings"] = warnings
            status["status"] = "healthy"

        return jsonify(status)
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return jsonify({
            "status": "degraded",
            "forecast_horizon_days": 3,
            "feature_store": {"available": False},
            "model_registry": {"available": False},
            "error": str(exc),
        }), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)
