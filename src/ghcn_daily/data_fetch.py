import asyncio
import aiohttp
import requests
import pandas as pd
import os
import csv
from datetime import datetime
from tqdm import tqdm
from src.config.config_manager import ConfigManager
from src.ghcn_daily.data_parsing import DataParser
from src.ghcn_daily.cpu_management import CPUMonitor  # Import CPUMonitor


class DataFetcher:
    def __init__(self, config_file='settings.json', config_directory="/workspaces/BlizzardX/src/config",data_type=None):
        self.config_manager = ConfigManager(config_directory)
        self.config_manager.load_config(config_file)
        cpu_config = self.config_manager.get(config_file, 'cpu_config', {})
        self.async_fetch = cpu_config.get('async_fetch', True)
        self.chunk_size = cpu_config.get('chunk_size', 100)
        self.dynamic_worker_adjustment = cpu_config.get('dynamic_worker_adjustment', True)  # New config option: 'csv' or 'dataframe'
        self.data_parser = DataParser()
        self.cpu_monitor = CPUMonitor(config_file=config_file, config_directory=config_directory)
        if data_type is not None:
            self.data_type = data_type
        else:
            self.data_type = self.config_manager.get(config_file, 'data_type', 'csv')

    def data_from_url(self, url):
        """Fetch data synchronously using requests."""
        data = []
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.splitlines():
                data.append(self.data_parser.parse_data_dly(line))
        else:
            print(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
        return data

    async def fetch_data_from_url(self, session, url):
        """Fetch data asynchronously using aiohttp."""
        data = []
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    for line in text.splitlines():
                        data.append(self.data_parser.parse_data_dly(line))
                else:
                    print(f"Failed to retrieve data for {url}. Status code: {response.status}")
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
        return data

    async def fetch_all_data(self, station_ids):
        """Fetch all data asynchronously for the given station_ids."""
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

        return all_data

    def fetch_all_data_sync(self, station_ids):
        """Fetch all data synchronously for the given station_ids."""
        all_data = []
        for station_id in station_ids:
            url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
            data_chunk = self.data_from_url(url)
            for entry in data_chunk:
                row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                row.extend(entry["DATA"])
                all_data.append(row)

        return all_data

    async def save_to_csv_incrementally(self, station_ids, output_filename, async_fetch=None):
        """Fetch data either synchronously or asynchronously and save it to a CSV file incrementally after each chunk."""
        if async_fetch is None:
            async_fetch = self.async_fetch  # Default to config value if not passed

        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers) 
            chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]
            # Start CPU monitoring if dynamic worker adjustment is enabled
            if self.dynamic_worker_adjustment:
                self.cpu_monitor.start_cpu_monitoring()
            if async_fetch:
                # Fetch and write data asynchronously for each chunk
                for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                    data_chunk = await self.fetch_all_data(chunk)
                    # Write each row of the data chunk to the CSV
                    for row in data_chunk:
                        writer.writerow(row)
            else:
                # Fetch and write data synchronously in chunks
                for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                    data_chunk = self.fetch_all_data_sync(chunk)
                    # Write each row of the data chunk to the CSV
                    for row in data_chunk:
                        writer.writerow(row)

        print(f"Data fetching and saving completed. Data saved to {output_filename}")

    async def save_to_dataframe_incrementally(self, station_ids):
        """Fetch data either synchronously or asynchronously and return it as a DataFrame incrementally."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

        if self.dynamic_worker_adjustment:
            self.cpu_monitor.start_cpu_monitoring()

        if self.async_fetch:
            # Fetch data asynchronously for each chunk
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                data_chunk = await self.fetch_all_data(chunk)
                all_data.extend(data_chunk)
        else:
            # Fetch data synchronously for each chunk
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                data_chunk = self.fetch_all_data_sync(chunk)
                all_data.extend(data_chunk)

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=headers)
        return df

    async def save_data(self, station_ids):
        """Determine whether to save the data to CSV or DataFrame based on the configuration."""
        # Set the output filename to 'station_data.csv'
        output_directory = "/workspaces/BlizzardX/Data"
        output_filename = os.path.join(output_directory, "station_data.csv")  # Fixed name 'station_data.csv'

        if self.data_type == 'csv':
            await self.save_to_csv_incrementally(station_ids, output_filename)
        elif self.data_type == 'dataframe':
            df = await self.save_to_dataframe_incrementally(station_ids)
            return df
        else:
            print(f"Invalid data_type '{self.data_type}' specified in config.")

from datetime import datetime
import aiohttp
import requests
import pandas as pd
import csv
import os
from tqdm import tqdm
import asyncio

class CurrentYearMonthDataFetcher(DataFetcher):
    def __init__(self, config_file='settings.json', config_directory="/workspaces/BlizzardX/src/config", data_type=None):
        # Initialize the parent class
        super().__init__(config_file, config_directory, data_type)

        # Get the current date and determine the current year and month
        self.current_date = datetime.now()
        self.current_year = self.current_date.year  # Current year
        self.current_month = self.current_date.month  # Current month
        
        # Handle previous month logic
        if self.current_month == 1:
            self.previous_month = 12  # December
            self.previous_year = self.current_year - 1  # Previous year
        else:
            self.previous_month = self.current_month - 1  # Previous month
            self.previous_year = self.current_year  # Same year

    def get_current_year(self):
        """Return the current year."""
        return self.current_year

    def get_current_month(self):
        """Return the current month."""
        return self.current_month

    def get_previous_month(self):
        """Return the previous month."""
        return self.previous_month

    def get_previous_year(self):
        """Return the previous year if the current month is January."""
        return self.previous_year

    def data_from_url(self, url):
        """Fetch data synchronously using requests, but only for the current year and the current or previous month."""
        data = []
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.splitlines():
                entry = self.data_parser.parse_data_dly(line)
                # Only include data for the current year and the current or previous month
                entry_year = int(entry["YEAR"])
                entry_month = int(entry["Month"])
                if entry_year == self.get_current_year() and (entry_month == self.get_current_month() or entry_month == self.get_previous_month()):
                    data.append(entry)
        else:
            print(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
        return data

    async def fetch_data_from_url(self, session, url):
        """Fetch data asynchronously using aiohttp, but only for the current year and the current or previous month."""
        data = []
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    for line in text.splitlines():
                        entry = self.data_parser.parse_data_dly(line)
                        entry_year = int(entry["YEAR"])
                        entry_month = int(entry["Month"])
                        # Only include data for the current year and the current or previous month
                        if entry_year == self.get_current_year() and (entry_month == self.get_current_month() or entry_month == self.get_previous_month()):
                            data.append(entry)
                else:
                    print(f"Failed to retrieve data for {url}. Status code: {response.status}")
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
        return data

    async def fetch_all_data(self, station_ids):
        """Fetch all data asynchronously for the given station_ids, but only for the current year and current/previous month."""
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
                    entry_year = int(entry["YEAR"])
                    entry_month = int(entry["Month"])
                    # Only include data for the current year and current/previous month
                    if entry_year == self.get_current_year() and (entry_month == self.get_current_month() or entry_month == self.get_previous_month()):
                        row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                        row.extend(entry["DATA"])
                        all_data.append(row)

        return all_data

    async def save_to_csv_incrementally(self, station_ids, output_filename, async_fetch=None):
        """Fetch data for the current year, current month, and previous month, then save it to a CSV file."""
        if async_fetch is None:
            async_fetch = self.async_fetch  # Default to config value if not passed

        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers) 
            chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

            # Start CPU monitoring if dynamic worker adjustment is enabled
            if self.dynamic_worker_adjustment:
                self.cpu_monitor.start_cpu_monitoring()

            if async_fetch:
                for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                    data_chunk = await self.fetch_all_data(chunk)
                    for row in data_chunk:
                        writer.writerow(row)
            else:
                for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                    data_chunk = self.fetch_all_data_sync(chunk)
                    for row in data_chunk:
                        writer.writerow(row)

        print(f"Data fetching and saving completed. Data saved to {output_filename}")

    async def save_to_dataframe_incrementally(self, station_ids):
        """Fetch data for the current year, current month, and previous month, then return it as a DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

        if self.dynamic_worker_adjustment:
            self.cpu_monitor.start_cpu_monitoring()

        if self.async_fetch:
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                data_chunk = await self.fetch_all_data(chunk)
                all_data.extend(data_chunk)
        else:
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                data_chunk = self.fetch_all_data_sync(chunk)
                all_data.extend(data_chunk)

        df = pd.DataFrame(all_data, columns=headers)
        return df

    async def save_data(self, station_ids):
        """Determine whether to save the data to CSV or DataFrame based on the configuration."""
        output_directory = "/workspaces/BlizzardX/Data"
        output_filename = os.path.join(output_directory, "current_year_month_station_data.csv")

        if self.data_type == 'csv':
            await self.save_to_csv_incrementally(station_ids, output_filename)
        elif self.data_type == 'dataframe':
            df = await self.save_to_dataframe_incrementally(station_ids)
            return df
        else:
            print(f"Invalid data_type '{self.data_type}' specified in config.")
