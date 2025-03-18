import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .data_processing import DataProcessor

class DataFetcher_T:
    @staticmethod
    def data_from_url(url):
        data = []
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.splitlines():
                data.append(DataProcessor.parse_data_dly(line))
        else:
            print(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
        return data

    @staticmethod
    def save_to_dataframe(station_ids, chunk_size=1000):
        all_data = []
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        # Using ThreadPoolExecutor to fetch data in parallel
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
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
        print(f"Data fetching and saving completed. Data saved to DataFrame.")
        return df

