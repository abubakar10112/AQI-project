"""
Optuna-powered Hyperparameter Optimization for AQI Time-Series Models.
Tunes XGBoost, Random Forest, and Ridge Regression models against validation RMSE.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """Automates hyperparameter search using Optuna."""

    def __init__(self, n_trials: int = 15, timeout: int = 180):
        self.n_trials = n_trials
        self.timeout = timeout

    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Tune XGBoost Regressor hyperparameters."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna is not installed; returning default XGBoost parameters.")
            return {}

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)

        import xgboost as xgb

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_train)
            preds = model.predict(X_va)
            return root_mean_squared_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        logger.info(f"Optuna XGBoost best validation RMSE: {study.best_value:.2f}")
        return study.best_params

    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Tune Random Forest Regressor hyperparameters."""
        if not OPTUNA_AVAILABLE:
            return {}

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(**params))
            ])
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return root_mean_squared_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(self.n_trials, 10), timeout=self.timeout)
        logger.info(f"Optuna Random Forest best validation RMSE: {study.best_value:.2f}")
        return study.best_params

    def tune_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Tune Ridge Regression hyperparameters."""
        if not OPTUNA_AVAILABLE:
            return {}

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline

        def objective(trial):
            alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            degree = trial.suggest_int("degree", 1, 2)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree)),
                ("ridge", Ridge(alpha=alpha, random_state=42))
            ])
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return root_mean_squared_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(self.n_trials, 10), timeout=self.timeout)
        logger.info(f"Optuna Ridge best validation RMSE: {study.best_value:.2f}")
        return study.best_params
