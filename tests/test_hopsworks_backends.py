"""Unit tests for Supabase feature-store and Hopsworks model-registry adapters."""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_pipeline import data_fetcher, feature_store
from src.feature_pipeline.feature_store import FeatureStoreUnavailable, SupabaseFeatureStore
from src.training_pipeline.model_registry import HopsworksModelRegistry


def _feature_frame():
    index = pd.date_range("2025-01-01", periods=2, freq="h", name="timestamp")
    values = {column: [1.0, 2.0] for column in [config.TARGET, *config.ALL_FEATURES]}
    return pd.DataFrame(values, index=index)


def test_supabase_store_upserts_features():
    """Verify that save_features calls upsert on the Supabase table."""
    mock_table = Mock()
    mock_table.upsert.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])

    mock_client = Mock()
    mock_client.table.return_value = mock_table

    store = SupabaseFeatureStore.__new__(SupabaseFeatureStore)
    store.table_name = config.SUPABASE_TABLE_NAME or "aqi_features"
    store._client = mock_client
    store._last_error = None

    store.save_features(_feature_frame())

    mock_client.table.assert_called_with(store.table_name)
    mock_table.upsert.assert_called_once()
    rows = mock_table.upsert.call_args.args[0]
    assert len(rows) == 2
    assert "timestamp" in rows[0]


def test_supabase_insert_failure_is_not_hidden():
    """Verify that Supabase insert errors propagate as FeatureStoreUnavailable."""
    mock_table = Mock()
    mock_table.upsert.return_value = mock_table
    mock_table.execute.side_effect = RuntimeError("Connection refused")

    mock_client = Mock()
    mock_client.table.return_value = mock_table

    store = SupabaseFeatureStore.__new__(SupabaseFeatureStore)
    store.table_name = "aqi_features"
    store._client = mock_client
    store._last_error = None

    with pytest.raises(FeatureStoreUnavailable, match="Connection refused"):
        store.save_features(_feature_frame())


def test_hopsworks_registry_uploads_without_persistent_local_copy():
    remote_model = Mock(version=17)
    model_registry = Mock()
    model_registry.python.create_model.return_value = remote_model

    registry = HopsworksModelRegistry.__new__(HopsworksModelRegistry)
    registry.mr = model_registry

    version = registry.save_model({"fitted": True}, "ridge", {"rmse": 12.3}, ["pm2_5"])

    assert version == 17
    remote_model.save.assert_called_once()
    uploaded_path = remote_model.save.call_args.args[0]
    assert "aqi-model-" in uploaded_path
    model_registry.python.create_model.assert_called_once()


def test_hourly_feature_pipeline_returns_failure_when_store_rejects_batch(monkeypatch):
    timestamps = pd.date_range(
        pd.Timestamp.now(tz=config.CITY_TIMEZONE).tz_localize(None).floor("h") - pd.Timedelta(hours=71),
        periods=72,
        freq="h",
        name="timestamp",
    )
    raw_columns = [config.TARGET, *config.WEATHER_FEATURES, *config.POLLUTANT_FEATURES]
    raw_data = pd.DataFrame(
        {column: np.linspace(1, 72, 72) for column in raw_columns}, index=timestamps
    )
    store = Mock()
    store.get_latest_features.return_value = None
    store.save_features.side_effect = FeatureStoreUnavailable("insert failed")

    monkeypatch.setattr(data_fetcher, "fetch_all_current", lambda: raw_data)
    monkeypatch.setattr(feature_store, "get_feature_store", lambda: store)

    assert data_fetcher.run_feature_pipeline() is False
