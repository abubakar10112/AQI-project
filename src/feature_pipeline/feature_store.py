import pandas as pd
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import src.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalFeatureStore:
    """Local parquet-based feature store."""
    def __init__(self):
        self.features_dir = config.FEATURES_DIR
        self.combined_file = self.features_dir / "aqi_features_all.parquet"

    @staticmethod
    def validate_features(df: pd.DataFrame) -> bool:
        """Validate that the feature dataset has the expected shape and schema."""
        if df is None or df.empty:
            return False
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception:
                return False
        required_columns = {config.TARGET, *config.ALL_FEATURES}
        if not required_columns.issubset(set(df.columns)):
            missing = sorted(required_columns - set(df.columns))
            logger.warning(f"Feature store validation failed. Missing columns: {missing}")
            return False
        return True

    def get_feature_store_status(self) -> dict:
        """Return basic metadata about the local feature store for health checks."""
        status = {
            "backend": "local",
            "available": False,
            "row_count": 0,
            "latest_timestamp": None,
            "missing_columns": [],
            "forecast_horizon_days": 3,
        }

        if not self.combined_file.exists():
            return status

        try:
            df = pd.read_parquet(self.combined_file)
            status["available"] = not df.empty
            status["row_count"] = int(len(df))
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                status["latest_timestamp"] = df.index.max().isoformat()
            if not df.empty:
                required_columns = {config.TARGET, *config.ALL_FEATURES}
                missing = sorted(required_columns - set(df.columns))
                status["missing_columns"] = missing
            return status
        except Exception as exc:
            logger.warning(f"Could not read feature store status: {exc}")
            return status

    def get_training_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if not self.combined_file.exists():
            logger.warning("No combined local feature store found.")
            return None
        
        df = pd.read_parquet(self.combined_file)
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return df.loc[mask]

    def get_latest_features(self, n_hours: int = 48) -> Optional[pd.DataFrame]:
        if not self.combined_file.exists():
            return None
            
        df = pd.read_parquet(self.combined_file)
        if df.empty:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        cutoff = df.index.max() - timedelta(hours=n_hours)
        return df.loc[df.index >= cutoff]

    def save_features(self, df: pd.DataFrame) -> None:
        if not self.validate_features(df):
            logger.warning("Feature validation failed before writing to store.")
            return

        # Save daily chunk
        date_str = df.index.max().strftime('%Y-%m-%d')
        chunk_file = self.features_dir / f"aqi_features_{date_str}.parquet"
        df.to_parquet(chunk_file)
        
        # Append to combined file
        if self.combined_file.exists():
            existing_df = pd.read_parquet(self.combined_file)
            # Concat and deduplicate by index, keeping latest
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            combined_df.to_parquet(self.combined_file)
        else:
            df.to_parquet(self.combined_file)
            
        logger.info(f"Saved {len(df)} rows to local feature store.")

class HopsworksFeatureStore:
    """Hopsworks feature store backend with automatic fallback to local."""
    def __init__(self):
        self.project_name = config.HOPSWORKS_PROJECT_NAME
        self.api_key = config.HOPSWORKS_API_KEY
        self.fg_name = 'aqi_features'
        self.fg_version = 1
        self._fs = None
        self._fg = None
        self._fallback = LocalFeatureStore()
        self._init_connection()

    def _init_connection(self):
        """Initialize Hopsworks connection if credentials are available."""
        if not self.api_key:
            logger.warning("HOPSWORKS_API_KEY not set. Feature store will use local fallback.")
            return
        try:
            import hopsworks
            project = hopsworks.login(
                host=config.HOPSWORKS_HOST,
                api_key_value=self.api_key,
                project=self.project_name
            )
            self._fs = project.get_feature_store()
            logger.info(f"Connected to Hopsworks: {self.project_name}")
        except Exception as e:
            logger.warning(f"Hopsworks connection failed: {e}. Using local fallback.")
            self._fs = None

    def _get_or_create_fg(self):
        """Get or create the feature group."""
        if self._fs is None or self._fg is not None:
            return self._fg
        try:
            self._fg = self._fs.get_or_create_feature_group(
                name=self.fg_name,
                version=self.fg_version,
                description="AQI features for Lahore forecasting",
                primary_key=["timestamp"],
                event_time="timestamp",
                online_enabled=True,
                stream=False,
            )
            return self._fg
        except Exception as e:
            logger.error(f"Failed to get/create feature group: {e}")
            self._fg = None
            return None

    def get_training_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        fg = self._get_or_create_fg()
        if fg is None:
            return self._fallback.get_training_data(start_date, end_date)
        try:
            df = self._to_timestamp_index(fg.read())
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            remote_data = df.loc[mask] if not df.empty else None
            if remote_data is not None and not remote_data.empty:
                return remote_data
            logger.warning("Hopsworks feature group is empty; falling back to local training data.")
            return self._fallback.get_training_data(start_date, end_date)
        except Exception as e:
            logger.warning(f"Hopsworks read failed: {e}. Falling back to local.")
            return self._fallback.get_training_data(start_date, end_date)

    def get_latest_features(self, n_hours: int = 48) -> Optional[pd.DataFrame]:
        fg = self._get_or_create_fg()
        if fg is None:
            return self._fallback.get_latest_features(n_hours)
        try:
            df = self._to_timestamp_index(fg.read())
            cutoff = df.index.max() - timedelta(hours=n_hours) if not df.empty else pd.Timestamp.now()
            remote_data = df.loc[df.index >= cutoff] if not df.empty else None
            if remote_data is not None and not remote_data.empty:
                return remote_data
            logger.warning("Hopsworks feature group is empty; falling back to local recent features.")
            return self._fallback.get_latest_features(n_hours)
        except Exception as e:
            logger.warning(f"Hopsworks read failed: {e}. Falling back to local.")
            return self._fallback.get_latest_features(n_hours)

    def get_feature_store_status(self) -> dict:
        """Return Hopsworks feature store status for health checks."""
        status = {
            "backend": "hopsworks",
            "available": False,
            "row_count": 0,
            "latest_timestamp": None,
            "missing_columns": [],
            "forecast_horizon_days": 3,
        }
        
        fg = self._get_or_create_fg()
        if fg is None:
            return status
            
        try:
            df = self._to_timestamp_index(fg.read())
            status["available"] = not df.empty
            status["row_count"] = int(len(df))
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                status["latest_timestamp"] = df.index.max().isoformat()
                required_columns = {config.TARGET, *config.ALL_FEATURES}
                status["missing_columns"] = sorted(required_columns - set(df.columns))
            return status
        except Exception as e:
            logger.warning(f"Could not read Hopsworks status: {e}")
            return status

    def save_features(self, df: pd.DataFrame) -> bool:
        """Persist features locally and return whether the Hopsworks upsert succeeded."""
        if not LocalFeatureStore.validate_features(df):
            logger.warning("Feature validation failed before writing to Hopsworks.")
            return False

        # Always save locally
        try:
            self._fallback.save_features(df)
        except Exception as e:
            logger.error(f"Local save failed: {e}")
        
        # Try to save to Hopsworks
        fg = self._get_or_create_fg()
        if fg is None:
            return False
        
        try:
            df_hs = self._to_hopsworks_frame(df)
            # Hudi's default operation is upsert. Waiting for both jobs makes a
            # successful return mean the data is available in the feature store.
            fg.insert(
                df_hs,
                operation="upsert",
                write_options={
                    "wait_for_job": True,
                    "wait_for_online_ingestion": True,
                },
            )
            logger.info(f"Stored {len(df_hs)} feature rows in Hopsworks.")
            return True
        except Exception as e:
            logger.warning(f"Hopsworks save failed: {e}")
            return False

    @staticmethod
    def _to_hopsworks_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Return a deterministic, Hopsworks-compatible feature dataframe."""
        df_hs = df.copy()
        if "timestamp" not in df_hs.columns:
            df_hs = df_hs.reset_index()
            index_column = df_hs.columns[0]
            if index_column != "timestamp":
                df_hs = df_hs.rename(columns={index_column: "timestamp"})

        if "timestamp" not in df_hs.columns:
            raise ValueError("Features must have a timestamp index or column.")

        df_hs["timestamp"] = pd.to_datetime(df_hs["timestamp"], errors="raise")
        if df_hs["timestamp"].isna().any():
            raise ValueError("Feature timestamps cannot be null.")

        # A Hopsworks feature group's primary key must be unique within a batch.
        return (
            df_hs.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    @staticmethod
    def _to_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature-group reads to the local store's timestamp index."""
        if "timestamp" not in df.columns:
            return df
        result = df.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
        return result.set_index("timestamp").sort_index()

def get_feature_store():
    """Factory function to get the configured feature store backend."""
    if config.FEATURE_STORE_BACKEND.lower() == "hopsworks":
        try:
            import hopsworks
            return HopsworksFeatureStore()
        except ImportError:
            logger.warning("hopsworks library not found. Falling back to local feature store.")
            return LocalFeatureStore()
    else:
        return LocalFeatureStore()

if __name__ == "__main__":
    fs = get_feature_store()
    print(f"Initialized {type(fs).__name__}")
