"""
Explainability Module using SHAP and Feature Importance
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src import config
from src.training_pipeline.model_registry import get_model_registry
from src.feature_pipeline.feature_store import get_feature_store

logger = logging.getLogger(__name__)


def compute_shap_values(model_wrapper, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Compute SHAP values using the appropriate explainer.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP library is not available.")
        return None

    try:
        # Extract underlying estimator and scaler if present
        inner_model = getattr(model_wrapper, 'model', model_wrapper)
        scaler = getattr(model_wrapper, 'scaler', None)

        if scaler is not None:
            X_train_scaled = pd.DataFrame(scaler.transform(X_train.values), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        model_type = type(inner_model).__name__.lower()

        if 'xgb' in model_type or 'forest' in model_type or 'tree' in model_type:
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer(X_test_scaled)
        elif 'ridge' in model_type or 'linear' in model_type:
            explainer = shap.LinearExplainer(inner_model, X_train_scaled)
            shap_values = explainer(X_test_scaled)
        else:
            sample_size = min(50, len(X_train_scaled))
            background = shap.sample(X_train_scaled, sample_size)
            explainer = shap.KernelExplainer(inner_model.predict, background)
            X_test_sample = shap.sample(X_test_scaled, min(30, len(X_test_scaled)))
            shap_values = explainer.shap_values(X_test_sample)

        return shap_values
    except Exception as e:
        logger.warning(f"Failed to compute SHAP values: {e}")
        return None


def plot_summary(shap_values, X_test: pd.DataFrame, save_path: str = None):
    """SHAP summary beeswarm plot."""
    if not SHAP_AVAILABLE or shap_values is None:
        return
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Summary plot failed: {e}")
        plt.close()


def plot_feature_importance(shap_values, X_test: pd.DataFrame, save_path: str = None):
    """Bar chart of mean |SHAP value| per feature."""
    if not SHAP_AVAILABLE or shap_values is None:
        return
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Feature importance plot failed: {e}")
        plt.close()


def get_top_features(shap_values, X_test: pd.DataFrame, model_wrapper=None, n=15) -> list:
    """Returns list of (feature_name, importance_score)."""
    # 1. Try from SHAP values
    if shap_values is not None:
        try:
            if hasattr(shap_values, 'values'):
                vals = np.abs(shap_values.values).mean(0)
            else:
                vals = np.abs(np.array(shap_values)).mean(0)

            feature_names = X_test.columns.tolist()
            feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['Feature', 'Importance'])
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(n)
            return feature_importance.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Failed to extract top features from SHAP: {e}")

    # 2. Fallback: extract directly from model if available
    if model_wrapper is not None:
        try:
            inner_model = getattr(model_wrapper, 'model', model_wrapper)
            feature_names = X_test.columns.tolist()
            if hasattr(inner_model, 'feature_importances_'):
                importances = inner_model.feature_importances_
                df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                df_imp = df_imp.sort_values(by='Importance', ascending=False).head(n)
                return df_imp.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Failed fallback feature importance: {e}")

    return []


def save_explanations(shap_values, X_test: pd.DataFrame, model_name: str, model_wrapper, path: str):
    """Save SHAP values and plots."""
    Path(path).mkdir(parents=True, exist_ok=True)
    base_path = os.path.join(path, f"shap_{model_name}")

    if shap_values is not None:
        plot_summary(shap_values, X_test, f"{base_path}_summary.png")
        plot_feature_importance(shap_values, X_test, f"{base_path}_importance.png")

    # Save top features JSON
    top_features = get_top_features(shap_values, X_test, model_wrapper=model_wrapper)
    summary_file = os.path.join(path, "shap_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(top_features, f, indent=4)

    logger.info(f"Saved SHAP explanations and feature summary to {summary_file}")


def run_explainability():
    """Run full explainability analysis."""
    registry = get_model_registry()
    model_name = "xgboost"
    try:
        model, metadata = registry.load_model(model_name)
    except Exception:
        model_name = "random_forest"
        try:
            model, metadata = registry.load_model(model_name)
        except Exception:
            model, metadata = registry.get_best_model()
            model_name = "best_model"

    store = get_feature_store()
    df = store.get_latest_features(n_hours=30 * 24)
    if df is None or df.empty:
        parquet_file = config.FEATURES_DIR / "aqi_features_all.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)

    if df is not None and not df.empty:
        features_to_use = [f for f in config.ALL_FEATURES if f in df.columns]
        X = df[features_to_use].select_dtypes(include=[np.number])
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]

        # Sample for fast computation
        X_test_sample = X_test.tail(100)
        shap_vals = compute_shap_values(model, X_train.tail(200), X_test_sample)
        save_explanations(shap_vals, X_test_sample, model_name, model, str(config.REPORTS_DIR))
        logger.info("Explainability analysis complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_explainability()
