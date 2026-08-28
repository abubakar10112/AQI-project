"""
Tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_raw_data():
    """Create a sample raw data DataFrame mimicking API output."""
    hours = 72  # 3 days of data
    timestamps = pd.date_range(
        start="2025-01-15 00:00:00",
        periods=hours,
        freq="h",
    )

    np.random.seed(42)
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": np.random.uniform(5, 15, hours),
            "relative_humidity_2m": np.random.uniform(50, 90, hours),
            "surface_pressure": np.random.uniform(1005, 1015, hours),
            "precipitation": np.random.choice([0.0, 0.0, 0.0, 0.1, 0.5], hours),
            "cloud_cover": np.random.uniform(10, 100, hours),
            "wind_speed_10m": np.random.uniform(1, 10, hours),
            "wind_direction_10m": np.random.uniform(0, 360, hours),
            "wind_gusts_10m": np.random.uniform(5, 20, hours),
            "pm2_5": np.random.uniform(50, 200, hours),
            "pm10": np.random.uniform(80, 300, hours),
            "nitrogen_dioxide": np.random.uniform(10, 60, hours),
            "sulphur_dioxide": np.random.uniform(5, 30, hours),
            "ozone": np.random.uniform(20, 80, hours),
            "carbon_monoxide": np.random.uniform(200, 800, hours),
            "us_aqi": np.random.uniform(80, 350, hours),
        }
    )
    data.set_index("timestamp", inplace=True)
    return data


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_time_features_computed(self, sample_raw_data):
        """Test that time-based features are correctly computed."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
        assert "season" in result.columns

        # Check value ranges
        assert result["hour"].between(0, 23).all()
        assert result["day_of_week"].between(0, 6).all()
        assert result["month"].between(1, 12).all()
        assert result["is_weekend"].isin([0, 1]).all()
        assert result["season"].isin([1, 2, 3, 4]).all()

    def test_lahore_features_computed(self, sample_raw_data):
        """Test Lahore-specific features for January data (smog season)."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        # January should be smog season
        assert "is_smog_season" in result.columns
        assert result["is_smog_season"].iloc[0] == 1

        # January should NOT be crop burning season (Oct-Nov only)
        assert "is_crop_burning_season" in result.columns
        assert result["is_crop_burning_season"].iloc[0] == 0

        # January should be brick kiln active
        assert "is_brick_kiln_active" in result.columns
        assert result["is_brick_kiln_active"].iloc[0] == 1

    def test_lag_features_computed(self, sample_raw_data):
        """Test that lag features are present and correct."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        assert "aqi_lag_1h" in result.columns
        assert "aqi_lag_3h" in result.columns
        assert "aqi_lag_24h" in result.columns

    def test_rolling_features_computed(self, sample_raw_data):
        """Test rolling mean and std features."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        assert "aqi_rolling_mean_24h" in result.columns
        assert "aqi_rolling_std_24h" in result.columns

    def test_interaction_features_computed(self, sample_raw_data):
        """Test interaction features."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        assert "temp_x_humidity" in result.columns
        assert "wind_x_pm25" in result.columns

    def test_no_nan_in_output(self, sample_raw_data):
        """Test that output has no NaN values after engineering."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        # After dropping NaN rows, there should be no NaN
        assert not result.isnull().any().any(), f"NaN found in columns: {result.columns[result.isnull().any()].tolist()}"

    def test_output_has_fewer_rows_than_input(self, sample_raw_data):
        """Test that some rows are dropped due to lag/rolling feature computation."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_raw_data)

        # We expect rows to be dropped because of 24h lag and rolling features
        assert len(result) < len(sample_raw_data)
        # But we should still have a reasonable number of rows
        assert len(result) > 0

    def test_summer_season_detection(self):
        """Test season detection for summer months."""
        from src.feature_pipeline.feature_engineer import FeatureEngineer

        # Create data for July (summer)
        hours = 48
        timestamps = pd.date_range(start="2025-07-15 00:00:00", periods=hours, freq="h")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_2m": np.random.uniform(35, 45, hours),
                "relative_humidity_2m": np.random.uniform(30, 50, hours),
                "surface_pressure": np.random.uniform(998, 1005, hours),
                "precipitation": np.zeros(hours),
                "cloud_cover": np.random.uniform(5, 30, hours),
                "wind_speed_10m": np.random.uniform(2, 8, hours),
                "wind_direction_10m": np.random.uniform(0, 360, hours),
                "wind_gusts_10m": np.random.uniform(8, 25, hours),
                "pm2_5": np.random.uniform(20, 80, hours),
                "pm10": np.random.uniform(40, 120, hours),
                "nitrogen_dioxide": np.random.uniform(5, 30, hours),
                "sulphur_dioxide": np.random.uniform(3, 15, hours),
                "ozone": np.random.uniform(40, 100, hours),
                "carbon_monoxide": np.random.uniform(100, 400, hours),
                "us_aqi": np.random.uniform(40, 150, hours),
            }
        )
        data.set_index("timestamp", inplace=True)

        engineer = FeatureEngineer()
        result = engineer.engineer_features(data)

        # July is NOT smog season
        assert result["is_smog_season"].iloc[0] == 0
        # July is summer (season 3)
        assert result["season"].iloc[0] == 3
