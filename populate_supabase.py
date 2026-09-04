#!/usr/bin/env python
"""Run the historical Open-Meteo backfill directly into Supabase.

The historical source is reproducible and the resulting training data is stored
in the configured Supabase feature store.
"""

from src.feature_pipeline.backfill import run_backfill


if __name__ == "__main__":
    run_backfill()
