import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src import config

class RandomForestModel:
    """Random Forest model for AQI prediction."""
    
    def __init__(self, **custom_params):
        self._name = 'random_forest'
        self.scaler = StandardScaler()
        merged_params = {**config.RANDOM_FOREST_PARAMS, **custom_params}
        self.rf = RandomForestRegressor(**merged_params)
        self.model = Pipeline([
            ('scaler', self.scaler),
            ('rf', self.rf)
        ])
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances from the RF model."""
        return self.rf.feature_importances_
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict AQI values."""
        return self.model.predict(X)

if __name__ == "__main__":
    # Test execution
    import logging
    logging.basicConfig(level=logging.INFO)
    model = RandomForestModel()
    logging.info(f"Initialized {model.name} model")
