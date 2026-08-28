import numpy as np
import pandas as pd

from src import config
from src.inference.predictor import ModelFallbackChain


def test_fallback_predictions_are_not_constant():
    chain = ModelFallbackChain(model_names=[])
    df = pd.DataFrame({config.TARGET: [160.0] * 24})

    result = chain.predict(df)
    preds = np.asarray(result['predictions'])

    assert result['model_used'] == 'fallback_last_known'
    assert len(preds) == config.FORECAST_HOURS
    assert np.std(preds) > 0
