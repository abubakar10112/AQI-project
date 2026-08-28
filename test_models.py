#!/usr/bin/env python
from src.training_pipeline.model_registry import get_model_registry

ms = get_model_registry().list_models()
print(f'Models in registry: {len(ms)}')
for i, m in enumerate(ms, 1):
    print(f'{i}. {m["model_name"]} v{m["version"]} - RMSE: {m["metrics"]["rmse"]:.2f}')
