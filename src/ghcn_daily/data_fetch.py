import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import asyncio
import aiohttp
from .data_processing import DataProcessor
from .cpu_management import CPUManager

class DataFetcher:
    def __init__(self, cpu_config):
        # Pass the 'cpu_config' directly to the CPUManager
        self.cpu_manager = CPUManager(cpu_config)
        self.cpu_manager.monitor_cpu_usage()

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

    async def fetch_data_from_url(self, session, url):
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

    def save_to_dataframe(self, station_ids):
        """Fetch data using ThreadPoolExecutor and save it to DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        with ThreadPoolExecutor(max_workers=self.cpu_manager.get_worker_count()) as executor:
            futures = []
            
            # Using ThreadPoolExecutor to submit tasks for each station ID
            for i in tqdm(range(0, len(station_ids), self.config["cpu_config"]["chunk_size"]), desc="Fetching Data", unit="chunk", ncols=100):
                chunk_station_ids = station_ids[i:i + self.config["cpu_config"]["chunk_size"]]
                for station_id in chunk_station_ids:
                    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                    futures.append(executor.submit(self.data_from_url, url))

            # Process the results as they come in
            for future in futures:
                chunk_data = future.result()
                for entry in chunk_data:
                    row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                    row.extend(entry["DATA"])
                    all_data.append(row)

        df = pd.DataFrame(all_data, columns=headers)
        print(f"Data fetching and saving completed. Data saved to DataFrame.")
        return df
