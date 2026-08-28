"""Unit tests for the Hopsworks persistence adapters (no live project needed)."""

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src import config
from src.feature_pipeline import data_fetcher
from src.feature_pipeline import feature_store
from src.feature_pipeline.feature_store import HopsworksFeatureStore, LocalFeatureStore
from src.training_pipeline.model_registry import HopsworksModelRegistry, LocalModelRegistry


def _feature_frame():
    index = pd.date_range("2025-01-01", periods=2, freq="h", name="timestamp")
    values = {column: [1.0, 2.0] for column in [config.TARGET, *config.ALL_FEATURES]}
    return pd.DataFrame(values, index=index)


def test_hopsworks_store_normalizes_and_waits_for_upsert(monkeypatch):
    feature_group = Mock()
    store = HopsworksFeatureStore.__new__(HopsworksFeatureStore)
    store._fg = feature_group
    store._fallback = Mock(spec=LocalFeatureStore)
    store._get_or_create_fg = Mock(return_value=feature_group)

    assert store.save_features(_feature_frame()) is True

    store._fallback.save_features.assert_called_once()
    inserted = feature_group.insert.call_args.args[0]
    assert "timestamp" in inserted.columns
    assert inserted["timestamp"].is_monotonic_increasing
    assert feature_group.insert.call_args.kwargs == {
        "operation": "upsert",
        "write_options": {
            "wait_for_job": True,
            "wait_for_online_ingestion": True,
        },
    }


def test_hopsworks_registry_uploads_local_artifact_and_records_remote_version(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    remote_model = Mock(version=17)
    model_registry = Mock()
    model_registry.python.create_model.return_value = remote_model

    registry = HopsworksModelRegistry.__new__(HopsworksModelRegistry)
    LocalModelRegistry.__init__(registry)
    registry.hopsworks_available = True
    registry.mr = model_registry

    version = registry.save_model({"fitted": True}, "ridge", {"rmse": 12.3}, ["pm2_5"])

    artifact_dir = tmp_path / "hopsworks_artifacts" / "ridge_v1"
    assert version == 1
    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()
    remote_model.save.assert_called_once_with(str(artifact_dir))

    metadata = json.loads((tmp_path / "ridge_v1_meta.json").read_text())
    assert metadata["hopsworks"]["version"] == 17
    model_registry.python.create_model.assert_called_once()


def test_hourly_feature_pipeline_engineers_and_persists_new_rows(monkeypatch):
    timestamps = pd.date_range("2025-01-01", periods=72, freq="h", name="timestamp")
    raw_columns = [config.TARGET, *config.WEATHER_FEATURES, *config.POLLUTANT_FEATURES]
    raw_data = pd.DataFrame(
        {column: np.linspace(1, 72, 72) for column in raw_columns},
        index=timestamps,
    )
    store = Mock()
    store.get_latest_features.return_value = None

    monkeypatch.setattr(data_fetcher, "fetch_all_current", lambda: raw_data)
    monkeypatch.setattr(feature_store, "get_feature_store", lambda: store)

    assert data_fetcher.run_feature_pipeline() is True
    saved_features = store.save_features.call_args.args[0]
    assert not saved_features.empty
    assert {config.TARGET, *config.ALL_FEATURES}.issubset(saved_features.columns)
