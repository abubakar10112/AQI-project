import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from src import config
from src.training_pipeline.models.ridge_regression import RidgeModel
from src.training_pipeline.models.random_forest import RandomForestModel
from src.training_pipeline.models.xgboost_model import XGBoostModel
from src.training_pipeline.models.tensorflow_model import TensorFlowModel
from src.training_pipeline.evaluator import Evaluator
from src.training_pipeline.model_registry import get_model_registry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """Orchestrates model training and evaluation."""
    
    def __init__(self):
        self.registry = get_model_registry()
        self.evaluator = Evaluator()
        
    def _create_sequences(self, data: pd.DataFrame, target: pd.Series, lookback: int):
        """Create sequences for LSTM."""
        X, y = [], []
        # Ensure we have enough data
        if len(data) <= lookback:
            return np.array([]), np.array([])
            
        # Convert to numpy for faster slicing
        data_vals = data.values
        target_vals = target.values
        
        for i in range(len(data_vals) - lookback):
            X.append(data_vals[i:(i + lookback)])
            y.append(target_vals[i + lookback])
        return np.array(X), np.array(y)
        
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
        
        # 1. Fetch training data from feature store
        feature_path = config.FEATURES_DIR / "aqi_features_all.parquet"
        if not feature_path.exists():
            # Fallback: try the alternative name
            feature_path = config.FEATURES_DIR / "features.parquet"
        if not feature_path.exists():
            logger.error(f"No feature file found in {config.FEATURES_DIR}")
            return None
            
        df = pd.read_parquet(feature_path)
        logger.info(f"Loaded {len(df)} rows from {feature_path}")
        
        # 2. Sort by time and split (80/10/10)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        n = len(df)
        train_idx = int(n * 0.8)
        val_idx = int(n * 0.9)
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        # 3. Prepare features and target
        if config.TARGET not in df.columns:
            logger.error(f"Target column {config.TARGET} not found in data")
            return None
            
        features_to_use = [f for f in config.ALL_FEATURES if f in df.columns]
        
        X_train_flat = train_df[features_to_use].values
        y_train = train_df[config.TARGET].values
        
        X_test_flat = test_df[features_to_use].values
        y_test = test_df[config.TARGET].values
        
        # For LSTM, create sequences
        X_seq, y_seq = self._create_sequences(df[features_to_use], df[config.TARGET], config.LOOKBACK_HOURS)
        
        seq_train_idx = int(len(X_seq) * 0.8)
        seq_val_idx = int(len(X_seq) * 0.9)
        
        if len(X_seq) > 0:
            X_train_seq = X_seq[:seq_train_idx]
            y_train_seq = y_seq[:seq_train_idx]
            X_test_seq = X_seq[seq_val_idx:]
            y_test_seq = y_seq[seq_val_idx:]
        else:
            X_train_seq, y_train_seq, X_test_seq, y_test_seq = None, None, None, None
        
        # 5. Initialize models
        models = {
            'ridge': (RidgeModel(), X_train_flat, y_train, X_test_flat, y_test),
            'random_forest': (RandomForestModel(), X_train_flat, y_train, X_test_flat, y_test),
            'xgboost': (XGBoostModel(), X_train_flat, y_train, X_test_flat, y_test)
        }

        if X_train_seq is not None and y_train_seq is not None and X_test_seq is not None and y_test_seq is not None:
            try:
                models['tensorflow'] = (TensorFlowModel(), X_train_seq, y_train_seq, X_test_seq, y_test_seq)
                logger.info("TensorFlow LSTM model enabled.")
            except Exception as exc:
                logger.warning(f"TensorFlow LSTM could not be initialized: {exc}")
        
        results = {}
        trained_models = {}
        
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
    trainer = Trainer()
    trainer.run_pipeline()
