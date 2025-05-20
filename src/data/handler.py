import logging
from src.data.parse import DataParser
from src.data.fetch import DataFetcher
import pandas as pd

# Configure logging to output to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only
    ]
)

class GHCNDataHandler:
    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GHCNDataHandler")
        
        try:
            self.parser = DataParser()
            self.fetcher = DataFetcher()
            self.logger.info("Successfully initialized DataParser and DataFetcher")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def get_station_data(self, station_url):

        self.logger.info(f"Fetching station data from {station_url}")
        try:
            return self.parser.parse_stations(station_url)
        except Exception as e:
            self.logger.error(f"Error fetching station data from {station_url}: {str(e)}")
            raise

    def get_inventory_data(self, inventory_url):

        self.logger.info(f"Fetching inventory data from {inventory_url}")
        try:
            return self.parser.parse_inventory(inventory_url)
        except Exception as e:
            self.logger.error(f"Error fetching inventory data from {inventory_url}: {str(e)}")
            raise

    def get_country_data(self, country_url):

        self.logger.info(f"Fetching country data from {country_url}")
        try:
            return self.parser.parse_countries(country_url)
        except Exception as e:
            self.logger.error(f"Error fetching country data from {country_url}: {str(e)}")
            raise

    def get_state_data(self, state_url):

        self.logger.info(f"Fetching state data from {state_url}")
        try:
            return self.parser.parse_states(state_url)
        except Exception as e:
            self.logger.error(f"Error fetching state data from {state_url}: {str(e)}")
            raise

    def get_weather_data(self, station_ids):

        self.logger.info(f"Fetching weather data for stations: {station_ids}")
        try:
            return self.fetcher.fetch_and_save_to_dataframe(station_ids)
        except Exception as e:
            self.logger.error(f"Error fetching weather data for stations {station_ids}: {str(e)}")
            raise