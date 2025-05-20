import sys
import os
import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Dynamically determine project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = os.getenv('PROJECT_ROOT', str(SCRIPT_DIR.parent))  # One level up from Scripts/
sys.path.append(PROJECT_ROOT)

# Setup logging to file and console
log_dir = Path('/tmp/logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"weather_pipeline__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(log_file)  # Log to file
    ]
)
logging.info(f"sys.path: {sys.path}")

# Import modules
from src.config.config_manager import ConfigManager
from data.handler import GHCNDataHandler
from data.fetch import DataFetcher
from data.process import WeatherDataTransformer, WeatherDataImputer
from data.filter import WeatherDataFilter

async def main():
    logging.info("Starting weather data processing pipeline...")

    # Setup
    config_path = str(Path(PROJECT_ROOT) / 'src' / 'config')
    processed_data_path = str(Path(PROJECT_ROOT) / 'Data' / 'processed_data.csv')
    cleaned_data_path = str(Path(PROJECT_ROOT) / 'Data' / 'cleaned_data.csv')

    try:
        logging.info("Loading configuration...")
        config = ConfigManager(config_directory=config_path)
        config.load_config('settings.json')

        logging.info("Initializing data handler...")
        ghcn = GHCNDataHandler()

        logging.info("Fetching station and inventory data...")
        stations = ghcn.get_station_data(config.get('settings.json', 'data_sources.stations'))
        inventory = ghcn.get_inventory_data(config.get('settings.json', 'data_sources.inventory'))

        logging.info("Filtering for active New Hampshire stations...")
        s_state_list = stations[stations['STATE'] == 'NH']['ID'].tolist()
        s_live_list = inventory[
            (inventory['ID'].isin(s_state_list)) &
            (inventory['LASTYEAR'] > 2024) &
            (inventory['FIRSTYEAR'] < 2015)
        ]['ID'].unique().tolist()
        logging.info(f"Found {len(s_live_list)} active stations in NH.")

        logging.info("Fetching latest weather data...")
        data_fetcher = DataFetcher(config_file='settings.json', data_type='dataframe')
        data = await data_fetcher.save_data(s_live_list)
        logging.info("Data fetched successfully.")

        logging.debug("Dropping FLAG columns...")
        flag_columns = [col for col in data.columns if 'FLAG' in col]
        data = data.drop(columns=flag_columns)

        logging.info("Transforming weather data...")
        weather_variables = ['TMAX', 'TMIN', 'SNOW', 'SNWD', 'PRCP']
        processor = WeatherDataTransformer(data, weather_variables)
        df = processor.process_data()

        logging.debug("Replacing missing value placeholders...")
        df.replace([-9999.0, -999.9], np.nan, inplace=True)

        logging.info("Merging with station metadata...")
        df = pd.merge(df, stations, on='ID', how='left')
        df = df[['DATE', 'ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'Season',
                 'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']]

        logging.info("Filtering based on missing data thresholds...")
        data_filter = WeatherDataFilter(df)
        df = df[df['ID'].isin(data_filter.get_stations_with_zero_missing_dates())]
        df = df[df['ID'].isin(data_filter.get_stations_with_low_missing_values(threshold=5))]
        logging.info(f"Remaining stations after filtering: {df['ID'].nunique()}")

        logging.info("Saving processed data...")
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False)
        logging.info(f"Processed data saved to: {processed_data_path}")

        logging.info("Imputing missing data...")
        imputer = WeatherDataImputer(df)
        df = imputer.clean_all()

        logging.info("Saving final cleaned data...")
        df.to_csv(cleaned_data_path, index=False)
        logging.info(f"Cleaned data saved to: {cleaned_data_path}")
        logging.info("✅ Weather data pipeline completed successfully.")

    except Exception as e:
        logging.error(f"❌ Error during processing: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
    logging.info("Pipeline execution finished.")
    logging.info("Exiting script.")