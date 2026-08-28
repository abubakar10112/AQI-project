#!/usr/bin/env python
"""Test Hopsworks integration"""
import sys
import os
sys.path.insert(0, '.')

print("=" * 60)
print("HOPSWORKS INTEGRATION TEST")
print("=" * 60)

# Check environment
print("\n1. Environment Variables:")
print(f"   FEATURE_STORE_BACKEND: {os.getenv('FEATURE_STORE_BACKEND', 'Not set')}")
print(f"   HOPSWORKS_API_KEY: {'SET' if os.getenv('HOPSWORKS_API_KEY') else 'NOT SET'}")
print(f"   HOPSWORKS_HOST: {os.getenv('HOPSWORKS_HOST', 'default')}")
print(f"   HOPSWORKS_PROJECT_NAME: {os.getenv('HOPSWORKS_PROJECT_NAME', 'default')}")

# Check config
from src import config
print("\n2. Config Settings:")
print(f"   FEATURE_STORE_BACKEND: {config.FEATURE_STORE_BACKEND}")
print(f"   HOPSWORKS_API_KEY: {'SET' if config.HOPSWORKS_API_KEY else 'NOT SET'}")
print(f"   HOPSWORKS_HOST: {config.HOPSWORKS_HOST}")
print(f"   HOPSWORKS_PROJECT_NAME: {config.HOPSWORKS_PROJECT_NAME}")

# Test feature store
print("\n3. Feature Store Backend:")
try:
    from src.feature_pipeline.feature_store import get_feature_store
    fs = get_feature_store()
    print(f"   Type: {type(fs).__name__}")
    
    status = fs.get_feature_store_status()
    print(f"   Status: {status['backend']}")
    print(f"   Available: {status['available']}")
    print(f"   Row Count: {status['row_count']}")
    print(f"   Latest: {status['latest_timestamp']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test model registry
print("\n4. Model Registry Backend:")
try:
    from src.training_pipeline.model_registry import get_model_registry
    mr = get_model_registry()
    print(f"   Type: {type(mr).__name__}")
    models = mr.list_models()
    print(f"   Models: {len(models)}")
    for m in models:
        print(f"     - {m['model_name']} v{m['version']}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
