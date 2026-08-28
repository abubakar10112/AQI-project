"""
Tests for the training pipeline.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_training_data():
    """Create synthetic training data for model testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Generate target with some linear relationship + noise
    y = (
        3.0 * X["feature_0"]
        + 1.5 * X["feature_1"]
        - 2.0 * X["feature_2"]
        + np.random.randn(n_samples) * 5
        + 100  # Offset to simulate AQI range
    )
    y = np.clip(y, 0, 500)  # Clip to valid AQI range

    return X, y


class TestEvaluator:
    """Tests for the model evaluator."""

    def test_evaluate_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        from src.training_pipeline.evaluator import Evaluator

        evaluator = Evaluator()
        y_true = np.array([100, 150, 200, 250, 300])
        y_pred = np.array([100, 150, 200, 250, 300])

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_evaluate_imperfect_predictions(self):
        """Test metrics with imperfect predictions."""
        from src.training_pipeline.evaluator import Evaluator

        evaluator = Evaluator()
        y_true = np.array([100, 150, 200, 250, 300])
        y_pred = np.array([110, 140, 210, 240, 310])

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r2"] < 1.0
        assert metrics["r2"] > 0  # Should still be reasonably good

    def test_compare_models(self):
        """Test model comparison produces sorted DataFrame."""
        from src.training_pipeline.evaluator import Evaluator

        evaluator = Evaluator()
        results = {
            "model_a": {"rmse": 15.0, "mae": 12.0, "r2": 0.85},
            "model_b": {"rmse": 10.0, "mae": 8.0, "r2": 0.92},
            "model_c": {"rmse": 20.0, "mae": 16.0, "r2": 0.78},
        }

        comparison = evaluator.compare_models(results)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        # Should be sorted by RMSE ascending
        assert comparison.index[0] == "model_b"
        assert comparison.index[-1] == "model_c"


class TestRidgeModel:
    """Tests for the Ridge regression model."""

    def test_ridge_fit_predict(self, synthetic_training_data):
        """Test Ridge model can fit and predict."""
        from src.training_pipeline.models.ridge_regression import RidgeModel

        X, y = synthetic_training_data
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = RidgeModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    def test_ridge_model_name(self):
        """Test model name property."""
        from src.training_pipeline.models.ridge_regression import RidgeModel

        model = RidgeModel()
        assert model.name == "ridge"


class TestRandomForestModel:
    """Tests for the Random Forest model."""

    def test_rf_fit_predict(self, synthetic_training_data):
        """Test Random Forest can fit and predict."""
        from src.training_pipeline.models.random_forest import RandomForestModel

        X, y = synthetic_training_data
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = RandomForestModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    def test_rf_feature_importances(self, synthetic_training_data):
        """Test that feature importances are available after fitting."""
        from src.training_pipeline.models.random_forest import RandomForestModel

        X, y = synthetic_training_data
        model = RandomForestModel()
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances is not None
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)


class TestXGBoostModel:
    """Tests for the XGBoost model."""

    def test_xgb_fit_predict(self, synthetic_training_data):
        """Test XGBoost can fit and predict."""
        try:
            from src.training_pipeline.models.xgboost_model import XGBoostModel
        except ImportError:
            pytest.skip("XGBoost not installed")

        X, y = synthetic_training_data
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = XGBoostModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    def test_xgb_model_name(self):
        """Test model name property."""
        try:
            from src.training_pipeline.models.xgboost_model import XGBoostModel
        except ImportError:
            pytest.skip("XGBoost not installed")

        model = XGBoostModel()
        assert model.name == "xgboost"


class TestTrainer:
    """Tests for the training orchestrator."""

    def test_time_based_split(self):
        """Test that time-based split preserves temporal order."""
        from src.training_pipeline.trainer import Trainer

        # Create time-ordered data
        n = 100
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "feature_1": np.random.randn(n),
                "us_aqi": np.random.uniform(50, 300, n),
            }
        )

        trainer = Trainer()
        train, val, test = trainer._time_based_split(data)

        # Train should be earliest, test should be latest
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

        # Proportions should be roughly 80/10/10
        assert len(train) == pytest.approx(80, abs=5)
