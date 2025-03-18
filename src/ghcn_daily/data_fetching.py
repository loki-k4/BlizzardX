import requests
import pandas as pd
from tqdm import tqdm
from .data_processing import DataProcessor

class DataFetcher:
    @staticmethod
    def read_data_from_url(url):
        data = []
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.splitlines():
                data.append(DataProcessor.parse_data_dly(line))
        else:
            print(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
        return data

    @staticmethod
    def fetch_and_save_to_dataframe(station_ids):
        all_data = []
        for station_id in tqdm(station_ids, desc="Fetching Data", unit="station", ncols=100):
            url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
            data = DataFetcher.read_data_from_url(url)
            all_data.extend(data)
        
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])
        
        df_data = []
        for entry in all_data:
            row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
            row.extend(entry["DATA"])
            df_data.append(row)
        
        return pd.DataFrame(df_data, columns=headers)
