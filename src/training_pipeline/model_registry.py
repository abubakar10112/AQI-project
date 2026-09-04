"""Hopsworks-backed model registry.

Only short-lived temporary files are used while serialising an artifact for
upload or downloading it for inference. Model versions and metadata live in
the remote Hopsworks Model Registry.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib

from src import config

logger = logging.getLogger(__name__)


class ModelRegistryUnavailable(RuntimeError):
    """Raised when the remote model registry cannot be used."""


class HopsworksModelRegistry:
    """Versioned remote registry for all trained AQI models."""

    def __init__(self):
        if not config.HOPSWORKS_API_KEY:
            raise ModelRegistryUnavailable("HOPSWORKS_API_KEY is not configured.")
        try:
            import hopsworks
        except ImportError as exc:
            raise ModelRegistryUnavailable(
                "The Hopsworks client is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        try:
            self.project = hopsworks.login(
                host=config.HOPSWORKS_HOST,
                api_key_value=config.HOPSWORKS_API_KEY,
                project=config.HOPSWORKS_PROJECT_NAME,
            )
            self.mr = self.project.get_model_registry()
        except Exception as exc:
            raise ModelRegistryUnavailable(f"Hopsworks model-registry connection failed: {exc}") from exc

    def save_model(self, model, model_name: str, metrics: dict, feature_list: Optional[list] = None) -> int:
        """Upload one model artifact and return its remote Hopsworks version."""
        metadata = {
            "model_name": model_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "feature_list": feature_list or [],
        }
        try:
            with tempfile.TemporaryDirectory(prefix="aqi-model-") as temp_dir:
                artifact_dir = Path(temp_dir)
                joblib.dump(model, artifact_dir / "model.joblib")
                metadata["artifact_file"] = "model.joblib"
                (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                remote_model = self.mr.python.create_model(
                    name=model_name,
                    metrics=metrics,
                    description="AQI forecaster for Lahore, trained from Hopsworks features.",
                )
                remote_model.save(str(artifact_dir))
                version = getattr(remote_model, "version", None)
                if version is None:
                    raise ModelRegistryUnavailable("Hopsworks did not return a registered model version.")
                logger.info("Registered %s v%s in Hopsworks.", model_name, version)
                return int(version)
        except ModelRegistryUnavailable:
            raise
        except Exception as exc:
            logger.exception("Hopsworks model upload failed")
            raise ModelRegistryUnavailable(
                f"Could not register model '{model_name}' in Hopsworks: {exc}"
            ) from exc

    def load_model(self, model_name: str, version: str | int = "latest"):
        """Download a registered artifact and return its model plus metadata."""
        try:
            requested_version = None if version == "latest" else int(version)
            remote_model = self.mr.get_model(model_name, version=requested_version)
            artifact_dir = Path(remote_model.download())
            metadata_path = artifact_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                artifact_file = metadata.get("artifact_file", "model.joblib")
                model_path = artifact_dir / artifact_file
                if not model_path.exists():
                    joblib_files = list(artifact_dir.glob("*.joblib"))
                    model_path = joblib_files[0] if joblib_files else model_path
            else:
                joblib_files = list(artifact_dir.glob("*.joblib"))
                if not joblib_files:
                    raise FileNotFoundError(f"No joblib model artifact found in {artifact_dir}")
                model_path = joblib_files[0]
                metadata = {"model_name": model_name, "artifact_file": model_path.name}

            model = joblib.load(model_path)
            metadata["version"] = getattr(remote_model, "version", requested_version)
            return model, metadata
        except Exception as exc:
            raise ModelRegistryUnavailable(
                f"Could not load model '{model_name}' from Hopsworks: {exc}"
            ) from exc

    def list_models(self) -> list:
        """Return the latest metadata entry for each remote model name."""
        try:
            entries = {}
            for model_name in config.MODEL_FALLBACK_CHAIN:
                try:
                    models = self.mr.get_models(model_name)
                except Exception:
                    continue
                for model in models:
                    name = getattr(model, "name", None)
                    version = getattr(model, "version", None)
                    if not name or version is None:
                        continue
                    if name not in entries or int(version) > int(entries[name]["version"]):
                        entries[name] = {
                            "model_name": name,
                            "version": int(version),
                            "metrics": getattr(model, "metrics", {}) or {},
                            "timestamp": str(getattr(model, "created", "")) or None,
                        }
            return list(entries.values())
        except Exception as exc:
            raise ModelRegistryUnavailable(f"Could not list Hopsworks models: {exc}") from exc

    def get_best_model(self):
        models = self.list_models()
        if not models:
            return None
        best = min(models, key=lambda item: item["metrics"].get("rmse", float("inf")))
        return self.load_model(best["model_name"], best["version"])


def get_model_registry() -> HopsworksModelRegistry:
    """Return the Hopsworks model registry."""
    return HopsworksModelRegistry()
