import asyncio
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor
import psutil
import pandas as pd
from tqdm import tqdm
import threading
from src.config.config_manager import ConfigManager  


class DataFetcher:
    def __init__(self, config_file='settings.json', config_directory="/workspaces/BlizzardX/src/config"):
        # Initialize ConfigManager
        self.config_manager = ConfigManager(config_directory)
        
        # Load the relevant configuration
        self.config_manager.load_config(config_file)
        
        # Retrieve CPU configuration from the loaded config
        cpu_config = self.config_manager.get(config_file, 'cpu_config', {})
        
        # Load async_fetch from the cpu_config section
        self.async_fetch = cpu_config.get('async_fetch', True)  # Default to True if not found
        self.cpu_usage_limit = cpu_config.get('cpu_usage_limit', 85)
        self.max_workers = cpu_config.get('max_concurrent_workers', 12)
        self.max_processes = cpu_config.get('max_concurrent_processes', 6)
        self.cpu_check_interval = cpu_config.get('cpu_check_interval', 2)
        self.chunk_size = cpu_config.get('chunk_size', 100)
        self.dynamic_worker_adjustment = cpu_config.get('dynamic_worker_adjustment', True)

        # Create a ThreadPoolExecutor for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.workers = self.max_workers

    def data_from_url(self, url):
        """Fetch data synchronously using requests."""
        data = []
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.splitlines():
                data.append(self.parse_data(line))  # Directly parsing the data without processing
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
                        data.append(self.parse_data(line))  # Directly parsing the data without processing
                else:
                    print(f"Failed to retrieve data for {url}. Status code: {response.status}")
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
        return data

    def parse_data(self, line):
        """Parse a single line of data into a dictionary."""
        data = []
        for i in range(21, 269, 8):
            value = int(line[i:i+5])
            mflag = line[i+5]
            qflag = line[i+6]
            sflag = line[i+7]
            data.extend([value, mflag, qflag, sflag])
        return {
            "ID": line[0:11].strip(),
            "YEAR": int(line[11:15]),
            "Month": int(line[15:17]),
            "ELEMENT": line[17:21].strip(),
            "DATA": data
        }

    async def fetch_all_data(self, station_ids):
        """Fetch all data asynchronously for the given station_ids."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        # Using aiohttp to fetch data asynchronously
        async with aiohttp.ClientSession() as session:
            tasks = []
            for station_id in station_ids:
                url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                tasks.append(self.fetch_data_from_url(session, url))

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Process the results as they come in
            for result in results:
                for entry in result:
                    row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                    row.extend(entry["DATA"])
                    all_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=headers)
        return df

    def monitor_cpu_usage(self):
        """Monitors CPU usage and dynamically adjusts the number of workers based on usage."""
        while True:
            cpu_usage = psutil.cpu_percent(interval=self.cpu_check_interval)
            if cpu_usage > self.cpu_usage_limit:
                self.decrease_workers()
            else:
                self.increase_workers()

    def decrease_workers(self):
        """Decrease the number of workers if CPU usage is too high."""
        if self.workers > 2:  # Ensure there's at least one worker
            self.workers -= 2
            self.executor._max_workers = self.workers
            print(f"CPU usage is high! Decreasing workers to {self.workers}")

    def increase_workers(self):
        """Increase the number of workers if CPU usage is low enough."""
        if self.workers < self.max_workers:
            self.workers += 2
            self.executor._max_workers = self.workers
            print(f"CPU usage is stable. Increasing workers to {self.workers}")

    async def save_to_dataframe(self, station_ids):
        """Fetch data either synchronously or asynchronously and save it to a DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        # Chunk the station IDs into smaller chunks
        chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

        # Start CPU monitoring in a separate thread if dynamic worker adjustment is enabled
        if self.dynamic_worker_adjustment:
            cpu_monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
            cpu_monitor_thread.daemon = True
            cpu_monitor_thread.start()

        if self.async_fetch:
            # Fetch data asynchronously for each chunk
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                df_chunk = await self.fetch_all_data(chunk)
                all_data.extend(df_chunk.values.tolist())  # Add the data to the final list
        else:
            # Fetch data synchronously in chunks
            for chunk in tqdm(chunks, desc="Fetching Data", ncols=100):
                for station_id in chunk:
                    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                    chunk_data = self.data_from_url(url)
                    for entry in chunk_data:
                        row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                        row.extend(entry["DATA"])
                        all_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=headers)
        print(f"Data fetching and saving completed. Data saved to DataFrame.")
        
        return df
