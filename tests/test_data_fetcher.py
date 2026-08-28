"""
Tests for the data fetcher module.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from datetime import datetime


class TestAQICNClient:
    """Tests for the AQICN API client."""

    @pytest.fixture
    def mock_aqicn_response(self):
        """Sample AQICN API response for Lahore."""
        return {
            "status": "ok",
            "data": {
                "aqi": 156,
                "idx": 8539,
                "time": {"iso": "2025-08-26T12:00:00+05:00"},
                "city": {"name": "Lahore US Consulate"},
                "iaqi": {
                    "pm25": {"v": 78.5},
                    "pm10": {"v": 120.3},
                    "no2": {"v": 32.1},
                    "so2": {"v": 8.5},
                    "o3": {"v": 45.2},
                    "co": {"v": 12.3},
                    "t": {"v": 35.0},
                    "h": {"v": 65.0},
                    "w": {"v": 3.5},
                    "p": {"v": 1008.0},
                },
            },
        }

    @patch("src.feature_pipeline.data_fetcher.requests.get")
    def test_fetch_current_aqi_success(self, mock_get, mock_aqicn_response):
        """Test successful AQI fetch from AQICN."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aqicn_response
        mock_get.return_value = mock_response

        from src.feature_pipeline.data_fetcher import AQICNClient

        client = AQICNClient()
        result = client.fetch_current()

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "us_aqi" in result.columns
        assert result["us_aqi"].iloc[0] == 156

    @patch("src.feature_pipeline.data_fetcher.requests.get")
    def test_fetch_current_aqi_failure(self, mock_get):
        """Test graceful handling of API failure."""
        mock_get.side_effect = Exception("Connection timeout")

        from src.feature_pipeline.data_fetcher import AQICNClient

        client = AQICNClient()
        result = client.fetch_current()

        # Should return None on failure, not crash
        assert result is None

    @patch("src.feature_pipeline.data_fetcher.requests.get")
    def test_fetch_handles_invalid_response(self, mock_get):
        """Test handling of invalid API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "error", "data": "Invalid key"}
        mock_get.return_value = mock_response

        from src.feature_pipeline.data_fetcher import AQICNClient

        client = AQICNClient()
        result = client.fetch_current()

        assert result is None


class TestOpenMeteoWeatherClient:
    """Tests for the Open-Meteo weather API client."""

    @pytest.fixture
    def mock_weather_response(self):
        """Sample Open-Meteo weather response."""
        return {
            "hourly": {
                "time": ["2025-08-26T00:00", "2025-08-26T01:00", "2025-08-26T02:00"],
                "temperature_2m": [30.5, 29.8, 29.2],
                "relative_humidity_2m": [65, 68, 70],
                "surface_pressure": [1008.0, 1007.5, 1007.2],
                "precipitation": [0.0, 0.0, 0.1],
                "cloud_cover": [25, 30, 35],
                "wind_speed_10m": [3.5, 3.2, 2.8],
                "wind_direction_10m": [180, 185, 190],
                "wind_gusts_10m": [8.5, 7.8, 7.2],
            }
        }

    @patch("src.feature_pipeline.data_fetcher.requests.get")
    def test_fetch_weather_success(self, mock_get, mock_weather_response):
        """Test successful weather data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_weather_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.feature_pipeline.data_fetcher import OpenMeteoWeatherClient

        client = OpenMeteoWeatherClient()
        df = client.fetch_current()

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "temperature_2m" in df.columns

    @patch("src.feature_pipeline.data_fetcher.requests.get")
    def test_fetch_historical_weather(self, mock_get, mock_weather_response):
        """Test historical weather data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_weather_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.feature_pipeline.data_fetcher import OpenMeteoWeatherClient

        client = OpenMeteoWeatherClient()
        df = client.fetch_historical("2025-01-01", "2025-01-31")

        assert df is not None
        assert isinstance(df, pd.DataFrame)


class TestOpenMeteoAirQualityClient:
    """Tests for the Open-Meteo air quality API client."""

    @pytest.fixture
    def mock_aq_response(self):
        """Sample Open-Meteo air quality response."""
        return {
            "hourly": {
                "time": ["2025-08-26T00:00", "2025-08-26T01:00"],
                "pm2_5": [45.2, 48.1],
                "pm10": [85.3, 90.1],
                "nitrogen_dioxide": [25.3, 27.1],
                "sulphur_dioxide": [8.5, 9.2],
                "ozone": [42.0, 40.5],
                "carbon_monoxide": [550.0, 580.0],
                "us_aqi": [125, 132],
            }
        }

    @patch("src.feature_pipeline.data_fetcher.requests.get")
    def test_fetch_air_quality_success(self, mock_get, mock_aq_response):
        """Test successful air quality data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aq_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.feature_pipeline.data_fetcher import OpenMeteoAirQualityClient

        client = OpenMeteoAirQualityClient()
        df = client.fetch_current()

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert "us_aqi" in df.columns
        assert "pm2_5" in df.columns
