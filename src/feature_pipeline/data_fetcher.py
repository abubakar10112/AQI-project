import requests
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import src.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQICNClient:
    """Client for AQICN Real-time API."""
    def __init__(self):
        self.token = config.AQICN_API_TOKEN
        self.base_url = config.AQICN_BASE_URL
        self.stations = config.AQICN_STATIONS
        self.max_retries = 3

    def fetch_current(self) -> Optional[pd.DataFrame]:
        for attempt in range(self.max_retries):
            try:
                for station in self.stations:
                    city = station["id"]
                    url = f"{self.base_url}/feed/{city}/?token={self.token}"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get("status") == "ok":
                        iaqi = data["data"].get("iaqi", {})
                        time_obj = data["data"].get("time", {})
                        time_val = time_obj.get("s") or time_obj.get("iso") or time_obj.get("v") or datetime.now().isoformat()
                        
                        def _to_float(val):
                            if val is None or val == "-" or val == "":
                                return None
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return None
                        
                        record = {
                            "timestamp": pd.to_datetime(time_val),
                            "us_aqi": _to_float(data["data"].get("aqi")),
                            "pm2_5": _to_float(iaqi.get("pm25", {}).get("v")),
                            "pm10": _to_float(iaqi.get("pm10", {}).get("v")),
                            "nitrogen_dioxide": _to_float(iaqi.get("no2", {}).get("v")),
                            "sulphur_dioxide": _to_float(iaqi.get("so2", {}).get("v")),
                            "ozone": _to_float(iaqi.get("o3", {}).get("v")),
                            "carbon_monoxide": _to_float(iaqi.get("co", {}).get("v")),
                            "temperature_2m": _to_float(iaqi.get("t", {}).get("v")),
                            "relative_humidity_2m": _to_float(iaqi.get("h", {}).get("v")),
                            "wind_speed_10m": _to_float(iaqi.get("w", {}).get("v")),
                            "surface_pressure": _to_float(iaqi.get("p", {}).get("v")),
                        }
                        df = pd.DataFrame([record]).set_index("timestamp")
                        return df
                return None
            except Exception as e:
                logger.warning(f"AQICN fetch failed (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(2 ** attempt)
        return None

class OpenMeteoWeatherClient:
    """Client for Open-Meteo Weather API."""
    def __init__(self):
        self.forecast_url = config.OPEN_METEO_WEATHER_URL
        self.history_url = config.OPEN_METEO_WEATHER_HISTORICAL_URL
        self.lat = config.CITY_LATITUDE
        self.lon = config.CITY_LONGITUDE
        self.params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        }

    def fetch_current(self) -> Optional[pd.DataFrame]:
        try:
            response = requests.get(self.forecast_url, params=self.params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "hourly" in data:
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
                df.index.name = "timestamp"
                return df
        except Exception as e:
            logger.error(f"OpenMeteo Weather fetch current failed: {e}")
        return None

    def fetch_historical(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            params = self.params.copy()
            params["start_date"] = start_date
            params["end_date"] = end_date
            response = requests.get(self.history_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "hourly" in data:
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
                df.index.name = "timestamp"
                return df
        except Exception as e:
            logger.error(f"OpenMeteo Weather fetch historical failed: {e}")
        return None

class OpenMeteoAirQualityClient:
    """Client for Open-Meteo Air Quality API."""
    def __init__(self):
        self.url = config.OPEN_METEO_AIR_QUALITY_URL
        self.lat = config.CITY_LATITUDE
        self.lon = config.CITY_LONGITUDE
        self.params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide,us_aqi"
        }

    def fetch_current(self) -> Optional[pd.DataFrame]:
        try:
            response = requests.get(self.url, params=self.params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "hourly" in data:
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
                df.index.name = "timestamp"
                return df
        except Exception as e:
            logger.error(f"OpenMeteo AQ fetch current failed: {e}")
        return None

    def fetch_historical(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            params = self.params.copy()
            params["start_date"] = start_date
            params["end_date"] = end_date
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "hourly" in data:
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
                df.index.name = "timestamp"
                return df
        except Exception as e:
            logger.error(f"OpenMeteo AQ fetch historical failed: {e}")
        return None

def fetch_all_current() -> Optional[pd.DataFrame]:
    """Fetches current data from all sources and merges them."""
    aqicn_client = AQICNClient()
    weather_client = OpenMeteoWeatherClient()
    aq_client = OpenMeteoAirQualityClient()

    weather_df = weather_client.fetch_current()
    aq_df = aq_client.fetch_current()
    
    if weather_df is None or aq_df is None:
        logger.error("Failed to fetch current data from OpenMeteo.")
        return None

    # Merge OpenMeteo data
    df = weather_df.join(aq_df, how='outer')
    
    # Try to augment/override with AQICN data for real-time accuracy
    aqicn_df = aqicn_client.fetch_current()
    if aqicn_df is not None and not aqicn_df.empty:
        # Update current hour with AQICN data if available
        last_idx = aqicn_df.index[-1]
        closest_idx = df.index.get_indexer([last_idx], method='nearest')[0]
        if closest_idx != -1:
            idx_name = df.index[closest_idx]
            for col in aqicn_df.columns:
                if col in df.columns and pd.notna(aqicn_df.iloc[-1][col]):
                    val = aqicn_df.iloc[-1][col]
                    try:
                        f_val = float(val)
                        df.at[idx_name, col] = f_val
                    except (ValueError, TypeError):
                        pass
                    
    return df

if __name__ == "__main__":
    df = fetch_all_current()
    if df is not None:
        print(f"Fetched current data. Shape: {df.shape}")
        print(df.head())
