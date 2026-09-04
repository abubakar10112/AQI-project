import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src import config
from src.feature_pipeline.feature_store import get_feature_store
from src.training_pipeline.model_registry import get_model_registry

logger = logging.getLogger(__name__)

class ModelFallbackChain:
    """
    Handles predicting AQI using a chain of fallback models.
    Tries primary model, then falls back to secondary if it fails.
    """
    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or config.MODEL_FALLBACK_CHAIN
        self.registry = get_model_registry()
        self.models = {}
        
        # Load all models
        for name in self.model_names:
            try:
                model, metadata = self.registry.load_model(name)
                self.models[name] = model
                logger.info(f"Loaded model: {name} (metrics: {metadata.get('metrics', {})})")
            except Exception as e:
                logger.warning(f"Could not load model {name}: {e}")

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        # Fallback base value: last known AQI
        last_known_aqi = features_df[config.TARGET].iloc[-1] if config.TARGET in features_df.columns else 100.0
        
        # Filter to feature columns only
        features_to_use = [f for f in config.ALL_FEATURES if f in features_df.columns]
        
        # Fetch future weather forecast if available for the 72h horizon
        weather_df = None
        try:
            from src.feature_pipeline.data_fetcher import OpenMeteoWeatherClient
            weather_df = OpenMeteoWeatherClient().fetch_current()
        except Exception as exc:
            logger.debug(f"Could not fetch future weather: {exc}")

        last_time = pd.to_datetime(features_df.index.max()) if isinstance(features_df.index, pd.DatetimeIndex) else pd.Timestamp.now()

        for name in self.model_names:
            model = self.models.get(name)
            if model is None:
                continue
            
            try:
                preds = []
                current_features = features_df[features_to_use].iloc[-1:].astype(float).copy()
                
                # History buffer for lag and rolling calculations
                if config.TARGET in features_df.columns:
                    aqi_history = [float(v) for v in features_df[config.TARGET].tail(48).dropna().values]
                elif 'aqi_lag_1h' in features_df.columns:
                    aqi_history = [float(v) for v in features_df['aqi_lag_1h'].tail(48).dropna().values]
                else:
                    aqi_history = []
                if not aqi_history:
                    aqi_history = [float(last_known_aqi)] * 48

                for step in range(1, config.FORECAST_HOURS + 1):
                    target_time = last_time + pd.Timedelta(hours=step)

                    # 1. Update future time features
                    if 'hour' in current_features.columns:
                        current_features.at[current_features.index[0], 'hour'] = target_time.hour
                    if 'day_of_week' in current_features.columns:
                        current_features.at[current_features.index[0], 'day_of_week'] = target_time.weekday()
                    if 'is_weekend' in current_features.columns:
                        current_features.at[current_features.index[0], 'is_weekend'] = 1.0 if target_time.weekday() >= 5 else 0.0
                    if 'day_of_month' in current_features.columns:
                        current_features.at[current_features.index[0], 'day_of_month'] = target_time.day
                    if 'month' in current_features.columns:
                        current_features.at[current_features.index[0], 'month'] = target_time.month
                    if 'season' in current_features.columns:
                        m = target_time.month
                        current_features.at[current_features.index[0], 'season'] = 1.0 if m in [12, 1, 2] else (2.0 if m in [3, 4, 5] else (3.0 if m in [6, 7, 8] else 4.0))

                    # 2. Update future weather features if available
                    if weather_df is not None and target_time in weather_df.index:
                        w_row = weather_df.loc[target_time]
                        for w_col in config.WEATHER_FEATURES:
                            if w_col in current_features.columns and w_col in w_row:
                                current_features.at[current_features.index[0], w_col] = float(w_row[w_col])

                    # 3. Update interaction feature
                    if 'temp_x_humidity' in current_features.columns:
                        t_val = current_features.at[current_features.index[0], 'temperature_2m'] if 'temperature_2m' in current_features.columns else 25.0
                        h_val = current_features.at[current_features.index[0], 'relative_humidity_2m'] if 'relative_humidity_2m' in current_features.columns else 60.0
                        current_features.at[current_features.index[0], 'temp_x_humidity'] = t_val * h_val

                    # 4. Update lag features from evolving AQI history
                    if len(aqi_history) >= 1 and 'aqi_lag_1h' in current_features.columns:
                        current_features.at[current_features.index[0], 'aqi_lag_1h'] = aqi_history[-1]
                    if len(aqi_history) >= 3 and 'aqi_lag_3h' in current_features.columns:
                        current_features.at[current_features.index[0], 'aqi_lag_3h'] = aqi_history[-3]
                    if len(aqi_history) >= 6 and 'aqi_lag_6h' in current_features.columns:
                        current_features.at[current_features.index[0], 'aqi_lag_6h'] = aqi_history[-6]
                    if len(aqi_history) >= 12 and 'aqi_lag_12h' in current_features.columns:
                        current_features.at[current_features.index[0], 'aqi_lag_12h'] = aqi_history[-12]
                    if len(aqi_history) >= 24 and 'aqi_lag_24h' in current_features.columns:
                        current_features.at[current_features.index[0], 'aqi_lag_24h'] = aqi_history[-24]

                    # 5. Update rolling features
                    if len(aqi_history) >= 24:
                        if 'aqi_rolling_mean_24h' in current_features.columns:
                            current_features.at[current_features.index[0], 'aqi_rolling_mean_24h'] = float(np.mean(aqi_history[-24:]))
                        if 'aqi_rolling_std_24h' in current_features.columns:
                            current_features.at[current_features.index[0], 'aqi_rolling_std_24h'] = float(np.std(aqi_history[-24:]))
                    if len(aqi_history) >= 4 and 'aqi_change_rate' in current_features.columns:
                        current_features.at[current_features.index[0], 'aqi_change_rate'] = float(aqi_history[-1] - aqi_history[-4]) / 3.0

                    pred = float(model.predict(current_features.values)[0])
                    preds.append(pred)
                    aqi_history.append(pred)
                
                preds = np.array(preds).flatten()
                
                # Check for invalid outputs
                if np.isnan(preds).any() or (preds < 0).any() or (preds > 600).any():
                    logger.warning(f"Model {name} returned invalid predictions. Attempting next model.")
                    continue
                
                logger.info(f"Successfully used model {name} for predictions.")
                return {
                    'predictions': preds,
                    'model_used': name,
                    'fallback_used': False
                }
            except Exception as e:
                logger.error(f"Model {name} prediction failed: {e}")
        
        logger.error("All models failed. Using last known AQI values as fallback.")
        fallback_preds = np.full(config.FORECAST_HOURS, last_known_aqi)
        
        return {
            'predictions': fallback_preds,
            'model_used': 'fallback_last_known',
            'fallback_used': True
        }


class Predictor:
    """
    Main predictor interface for the AQI Predictor project.
    """
    def __init__(self, model_names: List[str] = None):
        self.fallback_chain = ModelFallbackChain(model_names=model_names)
        self.feature_store = get_feature_store()
        
    def predict_next_3_days(self) -> Dict[str, Any]:
        """
        Generate predictions for the next 72 hours.
        """
        try:
            recent_data = self.feature_store.get_latest_features(n_hours=config.LOOKBACK_HOURS)
        except Exception as e:
            logger.warning(f"Failed to fetch features from Supabase: {e}")
            recent_data = None
            
        if recent_data is None or recent_data.empty:
            raise ValueError("No recent data available for prediction.")
            
        # 2 & 3. Make predictions using the fallback chain
        preds_result = self.fallback_chain.predict(recent_data)
        preds = preds_result['predictions']
        model_used = preds_result['model_used']
        
        hourly_predictions = []
        alerts = []
        daily_stats = {}
        
        base_time = pd.Timestamp.now(tz=config.CITY_TIMEZONE)
        
        # 4 & 5. Process predictions
        for i in range(len(preds)):
            pred_time = base_time + pd.Timedelta(hours=i+1)
            aqi_val = float(preds[i])
            category = config.get_aqi_category(aqi_val)
            
            hourly_predictions.append({
                'timestamp': pred_time.isoformat(),
                'aqi': round(aqi_val, 2),
                'category': category['label'],
                'color': category['color'],
                'emoji': category['emoji']
            })
            
            # Check for hazardous alerts
            if aqi_val > config.ALERT_THRESHOLD:
                alerts.append({
                    'timestamp': pred_time.isoformat(),
                    'aqi': round(aqi_val, 2),
                    'message': config.get_health_advisory(aqi_val)
                })
                
            # Aggregate for daily summary
            date_str = pred_time.strftime('%Y-%m-%d')
            if date_str not in daily_stats:
                daily_stats[date_str] = []
            daily_stats[date_str].append(aqi_val)
            
        # Compile daily summary
        daily_summary = []
        for date_str, vals in daily_stats.items():
            avg_aqi = np.mean(vals)
            max_aqi = np.max(vals)
            min_aqi = np.min(vals)
            cat = config.get_aqi_category(avg_aqi)['label']
            
            daily_summary.append({
                'date': date_str,
                'avg_aqi': round(avg_aqi, 2),
                'max_aqi': round(max_aqi, 2),
                'min_aqi': round(min_aqi, 2),
                'category': cat
            })
            
        # 6. Return response payload
        return {
            'hourly_predictions': hourly_predictions,
            'daily_summary': daily_summary,
            'alerts': alerts,
            'model_used': model_used,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = Predictor()
    try:
        results = predictor.predict_next_3_days()
        print("Predictions generated successfully.")
        print(f"Model used: {results['model_used']}")
        print(f"Generated at: {results['generated_at']}")
        print(f"Total alerts: {len(results['alerts'])}")
    except Exception as e:
        print(f"Error during prediction: {e}")
