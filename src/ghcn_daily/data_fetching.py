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
    def fetch_and_save_to_dataframe(station_ids, chunk_size=100):
        all_data = []
        
        # Prepare the headers for the DataFrame
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])

        # Fetch data in chunks
        for i in tqdm(range(0, len(station_ids), chunk_size), desc="Fetching Data", unit="chunk", ncols=100):
            chunk_station_ids = station_ids[i:i + chunk_size]
                
            chunk_data = []
            for station_id in chunk_station_ids:
                url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
                data = DataFetcher.read_data_from_url(url)
                chunk_data.extend(data)
            
            # Process chunk data and add to the all_data list
            for entry in chunk_data:
                row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
                row.extend(entry["DATA"])
                all_data.append(row)

        # Create DataFrame from all the data collected
        df = pd.DataFrame(all_data, columns=headers)
        
        print(f"Data fetching and saving completed. Data saved to DataFrame")
        return df
    
    @staticmethod
    def fetch_to_df(stations):
        data = []
        for station_id in tqdm(stations, desc="Fetching Data", unit="station", ncols=100):
            url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
            data = DataFetcher.read_data_from_url(url)
            data.extend(data)
        headers = ["ID", "YEAR", "Month", "ELEMENT"]
        for i in range(1, 32):
            headers.extend([f"VALUE{i}", f"MFLAG{i}", f"QFLAG{i}", f"SFLAG{i}"])
        df_data = []
        for entry in data:
            row = [entry["ID"], entry["YEAR"], entry["Month"], entry["ELEMENT"]]
            row.extend(entry["DATA"])
            df_data.append(row)
        return pd.DataFrame(df_data, columns=headers)
