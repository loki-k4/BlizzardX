import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import asyncio
import aiohttp
from .data_processing import DataProcessor

class DataFetcher_T:
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
    def save_to_dataframe(station_ids, chunk_size=100, max_workers=20):
        """Fetch data using ThreadPoolExecutor and save it to DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Using ThreadPoolExecutor to submit tasks for each station ID
            for i in tqdm(range(0, len(station_ids), chunk_size), desc="Fetching Data", unit="chunk", ncols=100):
                chunk_station_ids = station_ids[i:i + chunk_size]
                for station_id in chunk_station_ids:
                    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                    futures.append(executor.submit(DataFetcher_T.data_from_url, url))

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


class DataFetcher_A:
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

    @staticmethod
    async def fetch_and_save_to_dataframe(station_ids, chunk_size=100):
        """Fetch data asynchronously using asyncio and aiohttp, then save to DataFrame."""
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Fetch data in chunks
            for i in tqdm(range(0, len(station_ids), chunk_size), desc="Fetching Data", unit="chunk", ncols=100):
                chunk_station_ids = station_ids[i:i + chunk_size]
                for station_id in chunk_station_ids:
                    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                    task = asyncio.create_task(DataFetcher_A.fetch_data_from_url(session, url))
                    tasks.append(task)
            
            # Gather results and process them
            results = await asyncio.gather(*tasks)

            # Flatten the results and convert to DataFrame
            for chunk_data in results:
                for entry in chunk_data:
                    row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                    row.extend(entry["DATA"])
                    all_data.append(row)

        df = pd.DataFrame(all_data, columns=headers)
        print(f"Data fetching and saving completed using asyncio. Data saved to DataFrame.")
        return df


# To use ThreadPoolExecutor (TPE):
# You can call this method like:
# df_tpe = DataFetcher_T.save_to_dataframe(station_ids)

# To use Asyncio:
# You can call this method like:
# df_asyncio = asyncio.run(DataFetcher_A.fetch_and_save_to_dataframe(station_ids))

