"""Supabase feature-store adapter.

The application uses Supabase as the single source of truth for storing and
retrieving engineered AQI features. The Hopsworks platform is used exclusively
for the Model Registry.
"""

import logging
from datetime import timedelta
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class FeatureStoreUnavailable(RuntimeError):
    """Raised when the configured remote feature store cannot be used."""


class SupabaseFeatureStore:
    """Read and write AQI features through Supabase."""

    def __init__(self):
        self.table_name = config.SUPABASE_TABLE_NAME
        self._last_error: Optional[str] = None

        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise FeatureStoreUnavailable(
                "SUPABASE_URL and SUPABASE_KEY must be configured."
            )

        try:
            from supabase import create_client
        except ImportError as exc:
            raise FeatureStoreUnavailable(
                "The Supabase client is not installed. Run `pip install supabase`."
            ) from exc

        try:
            url = config.SUPABASE_URL.rstrip("/")
            if url.endswith("/rest/v1"):
                url = url[:-len("/rest/v1")].rstrip("/")
            self._client = create_client(url, config.SUPABASE_KEY)
            logger.info("Connected to Supabase project.")
        except Exception as exc:
            self._last_error = str(exc)
            raise FeatureStoreUnavailable(
                f"Could not connect to Supabase: {exc}"
            ) from exc

    @staticmethod
    def validate_features(df: pd.DataFrame) -> None:
        """Validate the exact schema before sending a batch to Supabase."""
        if df is None or df.empty:
            raise ValueError("Feature batch is empty.")
        if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" not in df.columns:
            raise ValueError("Features require a DatetimeIndex or a timestamp column.")

        required = {config.TARGET, *config.ALL_FEATURES}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Feature batch is missing columns: {missing}")

        numeric_columns = [config.TARGET, *config.ALL_FEATURES]
        invalid = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(df[column])]
        if invalid:
            raise ValueError(f"Feature columns must be numeric: {invalid}")
        if df[numeric_columns].isna().any().any():
            raise ValueError("Feature batch contains null feature values.")

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise a DataFrame into a flat table with a timestamp column."""
        result = df.copy()
        if "timestamp" not in result.columns:
            result = result.reset_index()
            result = result.rename(columns={result.columns[0]: "timestamp"})
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
        if result["timestamp"].isna().any():
            raise ValueError("Feature timestamps cannot be null.")
        return (
            result.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def get_training_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch features within a date range for model training."""
        try:
            response = (
                self._client.table(self.table_name)
                .select("*")
                .gte("timestamp", start_date)
                .lte("timestamp", end_date)
                .order("timestamp")
                .execute()
            )
            if not response.data:
                return pd.DataFrame()
            df = pd.DataFrame(response.data)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            return df.set_index("timestamp").sort_index()
        except Exception as exc:
            self._last_error = str(exc)
            raise FeatureStoreUnavailable(
                f"Supabase training-data read failed: {exc}"
            ) from exc

    def get_latest_features(self, n_hours: int = 48) -> pd.DataFrame:
        """Fetch the most recent N hours of features."""
        try:
            cutoff = (pd.Timestamp.now() - timedelta(hours=n_hours)).isoformat()
            response = (
                self._client.table(self.table_name)
                .select("*")
                .gte("timestamp", cutoff)
                .order("timestamp")
                .execute()
            )
            if not response.data:
                return pd.DataFrame()
            df = pd.DataFrame(response.data)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            return df.set_index("timestamp").sort_index()
        except Exception as exc:
            self._last_error = str(exc)
            raise FeatureStoreUnavailable(
                f"Supabase feature read failed: {exc}"
            ) from exc

    def save_features(self, df: pd.DataFrame) -> None:
        """Upsert a validated batch into the Supabase table."""
        self.validate_features(df)
        frame = self._prepare_frame(df)

        # Convert timestamps to ISO strings for JSON serialization
        records = frame.copy()
        records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Replace any remaining NaN/inf with None for JSON
        records = records.where(pd.notnull(records), None)
        rows = records.to_dict(orient="records")

        try:
            # Upsert in batches of 500 to avoid payload limits
            batch_size = 500
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                self._client.table(self.table_name).upsert(
                    batch, on_conflict="timestamp"
                ).execute()
            logger.info("Upserted %s rows to Supabase.", len(rows))
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Supabase feature insert failed")
            raise FeatureStoreUnavailable(
                "Supabase insert failed. Check the table schema, API key, "
                f"and RLS policies. Original error: {exc}"
            ) from exc

    def get_feature_store_status(self) -> dict:
        """Return status information about the feature store."""
        status = {
            "backend": "supabase",
            "available": False,
            "row_count": 0,
            "latest_timestamp": None,
            "missing_columns": [],
            "forecast_horizon_days": 3,
            "error": self._last_error,
        }
        try:
            # Get the latest row to check availability
            response = (
                self._client.table(self.table_name)
                .select("timestamp")
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                status["available"] = True
                status["latest_timestamp"] = response.data[0]["timestamp"]

            # Get approximate row count
            count_response = (
                self._client.table(self.table_name)
                .select("timestamp", count="exact")
                .execute()
            )
            status["row_count"] = count_response.count or 0
            return status
        except Exception as exc:
            status["error"] = str(exc)
            return status


def get_feature_store() -> SupabaseFeatureStore:
    """Return the Supabase feature store."""
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        raise FeatureStoreUnavailable(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY."
        )
    return SupabaseFeatureStore()
