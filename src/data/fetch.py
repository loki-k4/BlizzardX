import asyncio
import aiohttp
import requests
import pandas as pd
import os
import csv
from tqdm import tqdm
import logging
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.data.parse import DataParser
from src.utils.cpu_manager import CPUMonitor

# Configure logging to console only
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config_file='settings.json', config_directory=None, data_type=None):
        self.logger = setup_logging()
        self.logger.info("Initializing DataFetcher")
        
        # Determine config directory dynamically
        if config_directory is None:
            project_root = Path(__file__).resolve().parents[1]
            config_directory = project_root / "config"

        else:
            config_directory = Path(config_directory)
        
        # Ensure config directory exists
        try:
            config_directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Config directory set to: {config_directory}")
        except Exception as e:
            self.logger.error(f"Failed to create config directory: {e}")
            raise

        # Initialize ConfigManager and load config
        try:
            self.config_manager = ConfigManager(str(config_directory))
            self.config_manager.load_config(config_file)
            cpu_config = self.config_manager.get(config_file, 'cpu_config', {})
            self.async_fetch = cpu_config.get('async_fetch', True)
            self.chunk_size = cpu_config.get('chunk_size', 100)
            self.dynamic_worker_adjustment = cpu_config.get('dynamic_worker_adjustment', True)
            self.data_parser = DataParser()
            self.cpu_monitor = CPUMonitor(config_file=config_file, config_directory=config_directory)
            self.data_type = data_type if data_type is not None else self.config_manager.get(config_file, 'data_type', 'csv')
            self.logger.info(f"Loaded config: async_fetch={self.async_fetch}, chunk_size={self.chunk_size}, dynamic_worker_adjustment={self.dynamic_worker_adjustment}, data_type={self.data_type}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def data_from_url(self, url):
        """Fetch data synchronously using requests."""
        self.logger.info(f"Fetching data synchronously from {url}")
        data = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    data.append(self.data_parser.parse_data_dly(line))
                self.logger.info(f"Successfully fetched data from {url}")
            else:
                self.logger.error(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error fetching data from {url}: {e}")
        return data

    async def fetch_data_from_url(self, session, url):
        """Fetch data asynchronously using aiohttp."""
        self.logger.info(f"Fetching data asynchronously from {url}")
        data = []
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    for line in text.splitlines():
                        data.append(self.data_parser.parse_data_dly(line))
                    self.logger.info(f"Successfully fetched data from {url}")
                else:
                    self.logger.error(f"Failed to retrieve data for {url}. Status code: {response.status}")
        except Exception as e:
            self.logger.error(f"Error fetching data from {url}: {e}")
        return data

    async def fetch_all_data(self, station_ids):
        """Fetch all data asynchronously for the given station_ids."""
        self.logger.info(f"Fetching data for {len(station_ids)} stations asynchronously")
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        async with aiohttp.ClientSession() as session:
            tasks = []
            for station_id in station_ids:
                url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                tasks.append(self.fetch_data_from_url(session, url))

            results = await asyncio.gather(*tasks)

            for result in results:
                for entry in result:
                    row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                    row.extend(entry["DATA"])
                    all_data.append(row)

        self.logger.info(f"Fetched {len(all_data)} rows of data")
        return all_data

    def fetch_all_data_sync(self, station_ids):
        """Fetch all data synchronously for the given station_ids."""
        self.logger.info(f"Fetching data for {len(station_ids)} stations synchronously")
        all_data = []
        for station_id in station_ids:
            url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
            data_chunk = self.data_from_url(url)
            for entry in data_chunk:
                row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                row.extend(entry["DATA"])
                all_data.append(row)

        self.logger.info(f"Fetched {len(all_data)} rows of data")
        return all_data

    async def save_to_csv_incrementally(self, station_ids, output_filename, async_fetch=None):
        """Fetch data either synchronously or asynchronously and save it to a CSV file incrementally after each chunk."""
        if async_fetch is None:
            async_fetch = self.async_fetch  # Default to config value if not passed

        self.logger.info(f"Saving data to CSV: {output_filename}, async_fetch={async_fetch}")
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        # Ensure output directory exists
        try:
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            self.logger.info(f"Output directory created/verified: {os.path.dirname(output_filename)}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

            # Start CPU monitoring if dynamic worker adjustment is enabled
            if self.dynamic_worker_adjustment:
                self.logger.info("Starting CPU monitoring")
                self.cpu_monitor.start_cpu_monitoring()

            if async_fetch:
                for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                    data_chunk = await self.fetch_all_data(chunk)
                    for row in data_chunk:
                        writer.writerow(row)
                    self.logger.info(f"Processed chunk of {len(data_chunk)} rows")
            else:
                for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                    data_chunk = self.fetch_all_data_sync(chunk)
                    for row in data_chunk:
                        writer.writerow(row)
                    self.logger.info(f"Processed chunk of {len(data_chunk)} rows")

        self.logger.info(f"Data fetching and saving completed. Data saved to {output_filename}")

    async def save_to_dataframe_incrementally(self, station_ids):
        """Fetch data either synchronously or asynchronously and return it as a DataFrame incrementally."""
        self.logger.info("Fetching data for DataFrame")
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

        if self.dynamic_worker_adjustment:
            self.logger.info("Starting CPU monitoring")
            self.cpu_monitor.start_cpu_monitoring()

        if self.async_fetch:
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                data_chunk = await self.fetch_all_data(chunk)
                all_data.extend(data_chunk)
                self.logger.info(f"Processed chunk of {len(data_chunk)} rows")
        else:
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                data_chunk = self.fetch_all_data_sync(chunk)
                all_data.extend(data_chunk)
                self.logger.info(f"Processed chunk of {len(data_chunk)} rows")

        df = pd.DataFrame(all_data, columns=headers)
        self.logger.info(f"Created DataFrame with {len(df)} rows")
        return df

    async def save_data(self, station_ids):
        """Determine whether to save the data to CSV or DataFrame based on the configuration."""
        # Set output directory dynamically
        project_root = Path(__file__).resolve().parents[1]
        output_directory = project_root / "Data"

        output_filename = output_directory / "station_data.csv"

        self.logger.info(f"Starting data save process, data_type={self.data_type}")
        if self.data_type == 'csv':
            await self.save_to_csv_incrementally(station_ids, str(output_filename))
        elif self.data_type == 'dataframe':
            df = await self.save_to_dataframe_incrementally(station_ids)
            return df
        else:
            self.logger.error(f"Invalid data_type '{self.data_type}' specified in config")
            raise ValueError(f"Invalid data_type '{self.data_type}' specified in config")