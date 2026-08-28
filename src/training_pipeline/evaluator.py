import numpy as np
import pandas as pd
import json
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates model performance metrics for AQI predictions."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate regression metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        # Handle zero division for MAPE
        epsilon = np.finfo(np.float64).eps
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
    def compare_models(self, results: dict) -> pd.DataFrame:
        """Compare multiple models and return a sorted DataFrame."""
        df = pd.DataFrame.from_dict(results, orient='index')
        if not df.empty and 'rmse' in df.columns:
            df = df.sort_values(by='rmse')
        return df
        
    def print_report(self, results: dict):
        """Print a formatted report to console."""
        df = self.compare_models(results)
        print("\n" + "="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        print(df.to_string())
        print("="*50 + "\n")
        
    def save_results(self, results: dict, path: str):
        """Save evaluation results to JSON."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved evaluation results to {path}")

if __name__ == "__main__":
    # Test execution
    evaluator = Evaluator()
    sample_y = np.array([100, 150, 200])
    sample_pred = np.array([110, 145, 190])
    print(evaluator.evaluate(sample_y, sample_pred))
