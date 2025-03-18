from .data_parsing import DataParser
from .data_fetching import DataFetcher
import pandas as pd

class GHCNDataHandler:
    def __init__(self):
        self.parser = DataParser()
        self.fetcher = DataFetcher()

    def get_station_data(self, station_url):
        return self.parser.parse_stations(station_url)

    def get_inventory_data(self, inventory_url):
        return self.parser.parse_inventory(inventory_url)

    def get_country_data(self, country_url):
        return self.parser.parse_countries(country_url)

    def get_state_data(self, state_url):
        return self.parser.parse_states(state_url)

    def get_weather_data(self, station_ids):
        return self.fetcher.fetch_and_save_to_dataframe(station_ids)
