import requests
import pandas as pd
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .data_processing import DataProcessor
from .cpu_management import CPUManager

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.cpu_manager = CPUManager(cpu_usage_limit=config["cpu_usage_limit"], cpu_check_interval=config["cpu_check_interval"])
        self.cpu_manager.start_cpu_monitoring()

    @staticmethod
    def data_from_url(url):
        """Fetch data synchronously using requests."""
        data = []
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.splitlines():
                data.append(DataProcessor.parse_data_dly(line))
        else:
            print(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
        return data

    @staticmethod
    async def fetch_data_from_url(session, url):
        """Fetch data asynchronously using aiohttp."""
        async with session.get(url) as response:
            data = []
            if response.status == 200:
                text = await response.text()
                for line in text.splitlines():
                    data.append(DataProcessor.parse_data_dly(line))
            else:
                print(f"Failed to retrieve data for {url}. Status code: {response.status}")
            return data

    def save_to_dataframe_threaded(self, station_ids, config):
        """Fetch data using ThreadPoolExecutor while managing CPU usage."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        max_workers = self.cpu_manager.workers  # Dynamically adjust workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in tqdm(range(0, len(station_ids), config["chunk_size"]), desc="Fetching Data", unit="chunk", ncols=100):
                chunk_station_ids = station_ids[i:i + config["chunk_size"]]
                for station_id in chunk_station_ids:
                    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                    futures.append(executor.submit(DataFetcher.data_from_url, url))

            # Process the results as they come in
            for future in futures:
                chunk_data = future.result()
                for entry in chunk_data:
                    row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                    row.extend(entry["DATA"])
                    all_data.append(row)

        df = pd.DataFrame(all_data, columns=headers)
        print(f"Data fetching and saving completed using ThreadPoolExecutor. Data saved to DataFrame.")
        return df

    async def fetch_and_save_to_dataframe_async(self, station_ids, config):
        """Fetch data asynchronously using asyncio and aiohttp, then save to DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in tqdm(range(0, len(station_ids), config["chunk_size"]), desc="Fetching Data", unit="chunk", ncols=100):
                chunk_station_ids = station_ids[i:i + config["chunk_size"]]
                for station_id in chunk_station_ids:
                    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                    task = asyncio.create_task(DataFetcher.fetch_data_from_url(session, url))
                    tasks.append(task)

            # Gather results and process them
            results = await asyncio.gather(*tasks)
            for chunk_data in results:
                for entry in chunk_data:
                    row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                    row.extend(entry["DATA"])
                    all_data.append(row)

        df = pd.DataFrame(all_data, columns=headers)
        print(f"Data fetching and saving completed using asyncio. Data saved to DataFrame.")
        return df

    def fetch_data(self, station_ids, async_mode=False):
        """Fetch data using either ThreadPoolExecutor (synchronous) or asyncio (asynchronous)."""
        if async_mode:
            return asyncio.run(DataFetcher.fetch_and_save_to_dataframe_async(station_ids, self.config))
        else:
            return self.save_to_dataframe_threaded(station_ids, self.config)

