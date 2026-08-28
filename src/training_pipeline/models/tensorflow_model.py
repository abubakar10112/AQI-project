import numpy as np
import logging
from src import config
import os

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow is not installed. TensorFlowModel will be unavailable.")

class TensorFlowModel:
    """LSTM-based sequence model for time series AQI prediction."""
    
    def __init__(self):
        self._name = 'tensorflow'
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")
        self.params = config.TENSORFLOW_PARAMS
        self.model = None
        
    @property
    def name(self) -> str:
        return self._name
        
    def _build_model(self, input_shape):
        """Build the LSTM architecture."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(self.params.get('lstm_units', [64, 64])[0], return_sequences=True),
            Dropout(self.params.get('dropout_rate', 0.2)),
            LSTM(self.params.get('lstm_units', [64, 64])[1]),
            Dropout(self.params.get('dropout_rate', 0.2)),
            Dense(self.params.get('dense_units', 128), activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model to sequence data."""
        if self.model is None:
            # X_train is 3D: (samples, timesteps, features)
            self.model = self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
        callbacks = [
            EarlyStopping(
                patience=self.params.get('patience', 10),
                restore_best_weights=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                monitor='val_loss'
            )
        ]
        
        self.model.fit(
            X_train, y_train,
            epochs=self.params.get('epochs', 100),
            batch_size=self.params.get('batch_size', 32),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict AQI values."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
        
    def save_model(self, path: str):
        """Save model to disk in .keras format."""
        if not path.endswith('.keras'):
            path += '.keras'
        self.model.save(path)
        
    def load_model(self, path: str):
        """Load model from disk."""
        if not path.endswith('.keras'):
            path += '.keras'
        self.model = keras_load_model(path)
        return self

if __name__ == "__main__":
    # Test execution
    logging.basicConfig(level=logging.INFO)
    if TF_AVAILABLE:
        model = TensorFlowModel()
        logging.info(f"Initialized {model.name} model")
    else:
        logging.info("TensorFlow not available for testing")
