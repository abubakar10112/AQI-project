#!/usr/bin/env python
"""Quick test of predictor without API overhead"""
import sys
sys.path.insert(0, '.')
from src.inference.predictor import Predictor
import numpy as np

try:
    print("Initializing predictor...")
    predictor = Predictor()
    print("Making prediction...")
    result = predictor.predict_next_3_days()
    
    hourly = result['hourly_predictions']
    print(f"Model used: {result['model_used']}")
    print(f"Total predictions: {len(hourly)}")
    print("\nFirst 10 hourly AQI values:")
    for i, h in enumerate(hourly[:10]):
        print(f"  Hour {i+1}: {h['aqi']:.1f}")
    
    print("\nStatistics:")
    aqi_vals = [h['aqi'] for h in hourly]
    print(f"  Min: {min(aqi_vals):.1f}")
    print(f"  Max: {max(aqi_vals):.1f}")
    print(f"  Avg: {np.mean(aqi_vals):.1f}")
    print(f"  Std: {np.std(aqi_vals):.1f}")
    print(f"  Range: {max(aqi_vals) - min(aqi_vals):.1f}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
