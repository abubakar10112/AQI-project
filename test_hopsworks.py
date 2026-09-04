#!/usr/bin/env python
"""Fail-fast diagnostic for the required backend services."""

import sys

from src.feature_pipeline.feature_store import get_feature_store
from src.training_pipeline.model_registry import get_model_registry


def main() -> int:
    try:
        store = get_feature_store()
        status = store.get_feature_store_status()
        registry = get_model_registry()
        models = registry.list_models()
    except Exception as exc:
        print(f"BACKEND CHECK FAILED: {exc}")
        return 1

    print("BACKEND CHECK PASSED")
    print(f"Feature store: {status['backend']} / {status['row_count']} rows")
    print(f"Latest feature timestamp: {status['latest_timestamp']}")
    print(f"Registered models: {len(models)}")
    if not status["available"]:
        print("WARNING: the feature store is empty. Run populate_supabase.py before training.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
