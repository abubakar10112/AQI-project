import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import src.config as config
from src.feature_pipeline.data_fetcher import OpenMeteoWeatherClient, OpenMeteoAirQualityClient
from src.feature_pipeline.feature_engineer import FeatureEngineer
from src.feature_pipeline.feature_store import get_feature_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_monthly_chunks(start_date: str, end_date: str):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    chunks = []
    
    current = start
    while current <= end:
        next_month = current + relativedelta(months=1) - timedelta(days=1)
        chunk_end = min(next_month, end)
        chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        current = current + relativedelta(months=1)
        
    return chunks

def run_backfill():
    logger.info(f"Starting backfill from {config.BACKFILL_START_DATE} to {config.BACKFILL_END_DATE}")
    
    weather_client = OpenMeteoWeatherClient()
    aq_client = OpenMeteoAirQualityClient()
    engineer = FeatureEngineer()
    store = get_feature_store()
    
    chunks = generate_monthly_chunks(config.BACKFILL_START_DATE, config.BACKFILL_END_DATE)
    
    for start_str, end_str in tqdm(chunks, desc="Backfilling monthly chunks"):
        logger.info(f"Processing chunk {start_str} to {end_str}")
        
        # Avoid redundant downloads if we already have this data locally
        # This is a naive check. A better approach would query the store index.
        chunk_file = config.FEATURES_DIR / f"aqi_features_{end_str}.parquet"
        if chunk_file.exists():
             logger.info(f"Chunk file {chunk_file.name} already exists. Skipping.")
             continue
             
        weather_df = weather_client.fetch_historical(start_str, end_str)
        time.sleep(1) # Rate limit
        aq_df = aq_client.fetch_historical(start_str, end_str)
        time.sleep(1) # Rate limit
        
        if weather_df is None or aq_df is None:
            logger.error(f"Failed to fetch data for chunk {start_str} to {end_str}")
            continue
            
        merged_df = weather_df.join(aq_df, how='inner')
        
        if merged_df.empty:
            logger.warning(f"Merged dataframe is empty for chunk {start_str} to {end_str}")
            continue
            
        features_df = engineer.engineer_features(merged_df)
        
        if not features_df.empty:
            store.save_features(features_df)
        else:
            logger.warning(f"Engineered features empty for chunk {start_str} to {end_str}")
            
    logger.info("Backfill complete.")

def main():
    run_backfill()

if __name__ == "__main__":
    main()
