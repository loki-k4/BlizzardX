import asyncio
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm
from .data_processing import DataProcessor  # Ensure that this is correctly imported

class DataFetcher:
    def __init__(self, async_fetch=True, max_workers=5, chunk_size=10):
        """
        Initialize the DataFetcher.
        
        :param async_fetch: If True, use asynchronous fetching. Default is True.
        :param max_workers: Number of concurrent workers in ThreadPoolExecutor (for CPU-bound tasks).
        :param chunk_size: Number of station_ids to fetch per chunk.
        """
        self.async_fetch = async_fetch  # Use async fetching by default
        self.max_workers = max_workers  # Max number of workers for ThreadPoolExecutor
        self.chunk_size = chunk_size  # Size of chunks for parallel fetching
        
        # Create a ThreadPoolExecutor for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def data_from_url(self, url):
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
        data = []
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    for line in text.splitlines():
                        data.append(DataProcessor.parse_data_dly(line))
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
        print(f"Data fetching and saving completed. Data saved to DataFrame.")
        return df

    def process_data_with_thread_pool(self, data):
        """Process data using ThreadPoolExecutor for CPU-bound tasks."""
        # Here you can add any CPU-bound task. For demonstration, we simply return the data
        # using the executor to simulate a CPU-bound operation in parallel.
        results = list(self.executor.map(DataProcessor.process_data, data))
        return results

    async def save_to_dataframe(self, station_ids):
        """Fetch data either synchronously or asynchronously and save it to a DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        # Chunk the station IDs into smaller chunks
        chunks = [station_ids[i:i + self.chunk_size] for i in range(0, len(station_ids), self.chunk_size)]

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

        # Process data in parallel using ThreadPoolExecutor (for CPU-bound tasks)
        processed_data = self.process_data_with_thread_pool(all_data)

        # Convert to DataFrame
        df = pd.DataFrame(processed_data, columns=headers)
        print(f"Data fetching and saving completed. Data saved to DataFrame.")
        
        return df
