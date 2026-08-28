import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from src import config
import logging

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model for AQI prediction."""
    
    def __init__(self):
        self._name = 'xgboost'
        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            **config.XGBOOST_PARAMS,
            early_stopping_rounds=10
        )
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances from the XGBoost model."""
        return self.model.feature_importances_
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model to training data with early stopping."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        from sklearn.model_selection import train_test_split
        X_t, X_v, y_t, y_v = train_test_split(X_train_scaled, y_train, test_size=0.1, shuffle=False)
        
        self.model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False
        )
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict AQI values."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

if __name__ == "__main__":
    # Test execution
    logging.basicConfig(level=logging.INFO)
    model = XGBoostModel()
    logging.info(f"Initialized {model.name} model")
