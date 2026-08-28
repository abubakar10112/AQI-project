import os
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from src import config

logger = logging.getLogger(__name__)

class LocalModelRegistry:
    """Local file-based model registry."""
    
    def __init__(self):
        self.models_dir = config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, model, model_name: str, metrics: dict, feature_list: list = None):
        """Save model and metadata locally."""
        version = self._get_next_version(model_name)
        timestamp = datetime.now().isoformat()
        
        # Save model
        if model_name == 'tensorflow':
            model_path = self.models_dir / f"{model_name}_v{version}.keras"
            model.save_model(str(model_path))
        else:
            model_path = self.models_dir / f"{model_name}_v{version}.joblib"
            joblib.dump(model, model_path)
            
        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'metrics': metrics,
            'timestamp': timestamp,
            'feature_list': feature_list or []
        }
        meta_path = self.models_dir / f"{model_name}_v{version}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved model {model_name} v{version} to {model_path}")
        return version
        
    def load_model(self, model_name: str, version: str = 'latest'):
        """Load model and metadata."""
        if version == 'latest':
            version = str(self._get_latest_version(model_name))
            if not version:
                raise ValueError(f"No models found for {model_name}")
                
        meta_path = self.models_dir / f"{model_name}_v{version}_meta.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        if model_name == 'tensorflow':
            from src.training_pipeline.models.tensorflow_model import TensorFlowModel
            model_path = self.models_dir / f"{model_name}_v{version}.keras"
            model = TensorFlowModel().load_model(str(model_path))
        else:
            model_path = self.models_dir / f"{model_name}_v{version}.joblib"
            model = joblib.load(model_path)
            
        return model, metadata
        
    def list_models(self) -> list:
        """List only the latest version of each model."""
        models_dict = {}
        for file in self.models_dir.glob("*_meta.json"):
            with open(file, 'r') as f:
                model_meta = json.load(f)
                model_name = model_meta.get('model_name')
                version = model_meta.get('version', 0)
                # Keep only if this is the highest version seen so far
                if model_name not in models_dict or version > models_dict[model_name].get('version', 0):
                    models_dict[model_name] = model_meta
        return list(models_dict.values())
        
    def get_best_model(self):
        """Return the model with the lowest RMSE from metadata."""
        models = self.list_models()
        if not models:
            return None
            
        best = min(models, key=lambda x: x.get('metrics', {}).get('rmse', float('inf')))
        return self.load_model(best['model_name'], str(best['version']))
        
    def _get_next_version(self, model_name: str) -> int:
        versions = self._get_all_versions(model_name)
        return max(versions) + 1 if versions else 1
        
    def _get_latest_version(self, model_name: str) -> int:
        versions = self._get_all_versions(model_name)
        return max(versions) if versions else None
        
    def _get_all_versions(self, model_name: str) -> list:
        versions = []
        for file in self.models_dir.glob(f"{model_name}_v*_meta.json"):
            name = file.stem
            try:
                v = int(name.split('_v')[1].split('_meta')[0])
                versions.append(v)
            except (IndexError, ValueError):
                continue
        return versions

class HopsworksModelRegistry(LocalModelRegistry):
    """Hopsworks model registry (fallback to local)."""
    
    def __init__(self):
        super().__init__()
        try:
            import hopsworks
            self.project = hopsworks.login(host=config.HOPSWORKS_HOST, api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT_NAME)
            self.mr = self.project.get_model_registry()
            self.hopsworks_available = True
        except Exception as e:
            logger.warning(f"Failed to connect to Hopsworks: {e}. Falling back to LocalModelRegistry.")
            self.hopsworks_available = False
            
    def save_model(self, model, model_name: str, metrics: dict, feature_list: list = None):
        version = super().save_model(model, model_name, metrics, feature_list)
        if not self.hopsworks_available:
            return version
            
        try:
            # Stub for Hopsworks API interaction
            logger.info(f"Would save {model_name} v{version} to Hopsworks.")
        except Exception as e:
            logger.error(f"Error saving to Hopsworks: {e}")
        return version

def get_model_registry():
    """Factory function for model registry."""
    if config.FEATURE_STORE_BACKEND == "hopsworks":
        return HopsworksModelRegistry()
    return LocalModelRegistry()

if __name__ == "__main__":
    registry = get_model_registry()
    print(registry.list_models())
