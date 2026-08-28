#!/usr/bin/env python
"""Populate Hopsworks with local features"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("POPULATING HOPSWORKS WITH LOCAL FEATURES")
print("=" * 70)

from src.feature_pipeline.feature_store import LocalFeatureStore, HopsworksFeatureStore
from pathlib import Path
import pandas as pd

try:
    # 1. Load local features
    print("\n1. Loading local features...")
    lfs = LocalFeatureStore()
    local_file = Path('data/features/aqi_features_all.parquet')
    
    if not local_file.exists():
        print(f"   ❌ Local feature file not found: {local_file}")
        sys.exit(1)
    
    df = pd.read_parquet(local_file)
    print(f"   Loaded {len(df)} rows from local storage")
    print(f"   Features: {len(df.columns)} columns")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # 2. Push to Hopsworks
    print("\n2. Pushing to Hopsworks...")
    hfs = HopsworksFeatureStore()
    if not hfs.save_features(df):
        print("   ERROR: Features were kept locally but were not stored in Hopsworks.")
        sys.exit(2)
    print("   Features pushed successfully.")
    
    # 3. Verify
    print("\n3. Verifying Hopsworks status...")
    status = hfs.get_feature_store_status()
    print(f"   Backend: {status['backend']}")
    print(f"   Available: {status['available']}")
    print(f"   Row Count: {status['row_count']}")
    print(f"   Latest: {status['latest_timestamp']}")
    
    print("\n" + "=" * 70)
    print("HOPSWORKS INTEGRATION COMPLETE!")
    print("=" * 70)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
