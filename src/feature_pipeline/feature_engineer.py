import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import src.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Computes features for the AQI Predictor model."""
    
    def __init__(self):
        pass

    def compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Season (1=winter, 2=spring, 3=summer, 4=autumn)
        # Assuming Winter: 12,1,2; Spring: 3,4,5; Summer: 6,7,8; Autumn: 9,10,11
        season_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
        df["season"] = df["month"].map(season_map)
        
        return df

    def compute_lahore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_smog_season"] = df["month"].isin(config.SMOG_SEASON_MONTHS).astype(int)
        df["is_crop_burning_season"] = df["month"].isin(config.CROP_BURNING_MONTHS).astype(int)
        df["is_brick_kiln_active"] = df["month"].isin(config.BRICK_KILN_MONTHS).astype(int)
        
        # Wind from east (45-135 degrees)
        df["wind_from_east"] = ((df["wind_direction_10m"] >= 45) & (df["wind_direction_10m"] <= 135)).astype(int)
        
        # Days since rain
        # Identify rain events
        rain_events = df["precipitation"] > 0.1
        # Calculate days since last rain event
        # Create a series with dates of rain events
        rain_dates = df.index.to_series()[rain_events]
        if not rain_dates.empty:
            # Broadcast the last rain date to each row using reindex and ffill
            last_rain = rain_dates.reindex(df.index, method='ffill')
            # Calculate difference in days
            days_since = (df.index - last_rain).dt.total_seconds() / (24 * 3600)
            df["days_since_rain"] = days_since.fillna(30) # Default to 30 days if no prior rain
        else:
            df["days_since_rain"] = 30
            
        return df

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "us_aqi" not in df.columns:
            return df
            
        # Change rate over 3 hours
        df["aqi_change_rate"] = (df["us_aqi"] - df["us_aqi"].shift(3)) / 3
        
        # Rolling stats
        df["aqi_rolling_mean_24h"] = df["us_aqi"].rolling(window=24, min_periods=1).mean()
        df["aqi_rolling_std_24h"] = df["us_aqi"].rolling(window=24, min_periods=1).std()
        
        # Lags
        lags = [1, 3, 6, 12, 24]
        for lag in lags:
            df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)
            
        # Interactions
        if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
            df["temp_x_humidity"] = (df["temperature_2m"] * df["relative_humidity_2m"]) / 100
            
        if "wind_speed_10m" in df.columns and "pm2_5" in df.columns:
            df["wind_x_pm25"] = df["wind_speed_10m"] * df["pm2_5"]
            
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to compute all features."""
        logger.info("Engineering features...")
        df_feat = df.copy()
        
        # Ensure index is sorted datetime
        df_feat = df_feat.sort_index()
        
        # Forward fill missing values then backward fill
        df_feat = df_feat.ffill().bfill()
        
        df_feat = self.compute_time_features(df_feat)
        df_feat = self.compute_lahore_features(df_feat)
        df_feat = self.compute_derived_features(df_feat)
        
        # Drop any remaining NaNs
        df_feat = df_feat.dropna()
        
        return df_feat

if __name__ == "__main__":
    # Test feature engineering
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df_test = pd.DataFrame({
        "us_aqi": np.random.randint(50, 300, 100),
        "temperature_2m": np.random.uniform(10, 40, 100),
        "relative_humidity_2m": np.random.uniform(20, 90, 100),
        "wind_speed_10m": np.random.uniform(0, 15, 100),
        "wind_direction_10m": np.random.uniform(0, 360, 100),
        "precipitation": np.random.choice([0, 0.5, 2.0], 100, p=[0.9, 0.05, 0.05]),
        "pm2_5": np.random.uniform(10, 150, 100)
    }, index=dates)
    
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_test)
    print(f"Engineered features shape: {df_features.shape}")
    print(df_features.columns)
