import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from src import config
from src.training_pipeline.models.ridge_regression import RidgeModel
from src.training_pipeline.models.random_forest import RandomForestModel
from src.training_pipeline.models.xgboost_model import XGBoostModel
from src.training_pipeline.evaluator import Evaluator
from src.training_pipeline.model_registry import get_model_registry
from src.feature_pipeline.feature_store import get_feature_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """Orchestrates model training and evaluation."""
    
    def __init__(self):
        self.registry = get_model_registry()
        self.feature_store = get_feature_store()
        self.evaluator = Evaluator()
        
    def _time_based_split(self, df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
        """Split data preserving temporal order (no shuffling)."""
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_idx].copy()
        val = df.iloc[train_idx:val_idx].copy()
        test = df.iloc[val_idx:].copy()
        return train, val, test
        
    def run_pipeline(self):
        """Run the full training pipeline."""
        logger.info("Starting training pipeline...")
        
        # 1. Fetch training data from the configured feature store. The
        # Hopsworks is the single source of truth for training features.
        end_date = pd.Timestamp.now().ceil("D").strftime("%Y-%m-%d")
        df = self.feature_store.get_training_data(config.BACKFILL_START_DATE, end_date)
        if df is None or df.empty:
            logger.error("No training features available from the configured feature store.")
            return None
        logger.info(
            "Loaded %s rows from %s feature store.",
            len(df),
            type(self.feature_store).__name__,
        )
        
        # 2. Train a genuine one-hour-ahead forecast, not a model that learns
        # the AQI value from the same timestamp. Recursive inference expands
        # this one-hour forecast to the required 72-hour horizon.
        df = df.sort_index()
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        if config.TARGET not in df.columns:
            logger.error(f"Target column {config.TARGET} not found in data")
            return None
        df = df.copy()
        df["target_aqi_t_plus_1h"] = df[config.TARGET].shift(-1)
        df = df.dropna(subset=["target_aqi_t_plus_1h"])

        n = len(df)
        if n < 100:
            logger.error("At least 100 clean hourly feature rows are required for training.")
            return None
        train_idx = int(n * 0.8)
        val_idx = int(n * 0.9)
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        # 3. Prepare feature inputs and the one-hour-ahead target.
        features_to_use = [f for f in config.ALL_FEATURES if f in df.columns]
        if set(config.ALL_FEATURES) - set(features_to_use):
            logger.error("Training data is missing required feature columns.")
            return None
        
        X_val_flat = val_df[features_to_use].values
        y_val = val_df["target_aqi_t_plus_1h"].values

        # 4. Hyperparameter Optimization via Optuna (if enabled)
        xgb_params = {}
        rf_params = {}
        ridge_params = {}

        if tune_hyperparameters:
            try:
                from src.training_pipeline.hyperparameter_tuner import HyperparameterTuner
                tuner = HyperparameterTuner(n_trials=15)
                logger.info("Running Optuna hyperparameter search for XGBoost...")
                xgb_params = tuner.tune_xgboost(X_train_flat, y_train, X_val_flat, y_val)
                logger.info("Running Optuna hyperparameter search for Random Forest...")
                rf_params = tuner.tune_random_forest(X_train_flat, y_train, X_val_flat, y_val)
                logger.info("Running Optuna hyperparameter search for Ridge...")
                ridge_params = tuner.tune_ridge(X_train_flat, y_train, X_val_flat, y_val)
            except Exception as e:
                logger.warning(f"Optuna tuning failed, using default parameters: {e}")

        # 5. Initialize models with best parameters
        models = {
            'ridge': (RidgeModel(**ridge_params), X_train_flat, y_train, X_test_flat, y_test),
            'random_forest': (RandomForestModel(**rf_params), X_train_flat, y_train, X_test_flat, y_test),
            'xgboost': (XGBoostModel(**xgb_params), X_train_flat, y_train, X_test_flat, y_test)
        }
        
        results = {}
        trained_models = {}

        # This is the minimum benchmark requested for a time-series forecast:
        # next hour equals the latest observed AQI.
        results["persistence_baseline"] = self.evaluator.evaluate(
            y_test, test_df[config.TARGET].values
        )
        
        # 6. Train and Evaluate
        for name, (model, X_tr, y_tr, X_te, y_te) in models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                metrics = self.evaluator.evaluate(y_te, preds)
                results[name] = metrics
                trained_models[name] = model
                logger.info(f"{name} trained successfully. RMSE: {metrics['rmse']:.2f}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                
        # 7. Log metrics & print report
        if results:
            self.evaluator.print_report(results)
            
            report_path = config.REPORTS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.evaluator.save_results(results, str(report_path))
            self.evaluator.save_results(results, str(config.MODELS_DIR / "training_results.json"))
            
            # 8. Register all models
            best_rmse = float('inf')
            best_model_name = None
            
            for name, model in trained_models.items():
                self.registry.save_model(model, name, results[name], features_to_use)
                if results[name]['rmse'] < best_rmse:
                    best_rmse = results[name]['rmse']
                    best_model_name = name
                    
            if best_model_name:
                logger.info(f"Best model is {best_model_name} with RMSE {best_rmse:.2f}")
                
        # 9. Return results
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AQI Model Training Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    args = parser.parse_args()

    trainer = Trainer()
    trainer.run_pipeline(tune_hyperparameters=args.tune)
