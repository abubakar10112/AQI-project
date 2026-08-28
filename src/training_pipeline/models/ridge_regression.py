import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from src import config

class RidgeModel:
    """Ridge regression model with polynomial features for AQI prediction."""
    
    def __init__(self):
        self._name = 'ridge'
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
            ('ridge', Ridge(**config.RIDGE_PARAMS))
        ])
        
    @property
    def name(self) -> str:
        return self._name
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict AQI values."""
        return self.model.predict(X)
        
    def get_params(self) -> dict:
        """Return model parameters."""
        return self.model.named_steps['ridge'].get_params()

if __name__ == "__main__":
    # Test execution
    import logging
    logging.basicConfig(level=logging.INFO)
    model = RidgeModel()
    logging.info(f"Initialized {model.name} model with params: {model.get_params()}")
