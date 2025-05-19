# update_feature_data.py
# Purpose: Pull recent NOAA data for multiple NH stations, clean, engineer features, and OVERWRITE feature_data.csv

import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# Paths
DATA_DIR = '/workspaces/BlizzardX/Data'
FEATURE_CSV = os.path.join(DATA_DIR, 'feature_data.csv')
RAW_SAVE_DIR = DATA_DIR
GHCND_STATION_FILE = os.path.join(DATA_DIR, 'ghcnd-stations.txt')

# Date range for pull
START_DATE = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# NOAA API endpoint
BASE_URL = 'https://www.ncei.noaa.gov/access/services/data/v1'

# Feature columns to keep
EXPECTED_COLS = ['DATE', 'Station_ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 
                 'NAME', 'Season', 'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']

def get_nh_stations(file_path):
    """Extracts NH station IDs from ghcnd-stations.txt."""
    nh_stations = []
    with open(file_path, 'r') as f:
        for line in f:
            if line[38:40] == 'NH':
                station_id = line[0:11].strip()
                nh_stations.append(station_id)
    return nh_stations

def download_noaa_data(station_id, start_date, end_date):
    """Downloads NOAA daily summaries for a single station."""
    params = {
        'dataset': 'daily-summaries',
        'stations': station_id,
        'startDate': start_date,
        'endDate': end_date,
        'format': 'json',
        'units': 'metric',
        'includeAttributes': 'false'
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.rename(columns={'STATION': 'Station_ID'}, inplace=True)
        df['NAME'] = station_id
        df['LATITUDE'] = None
        df['LONGITUDE'] = None
        df['ELEVATION'] = None
        df['Season'] = pd.to_datetime(df['DATE']).dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        # Ensure all expected columns exist
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = None

        return df[EXPECTED_COLS]

    except Exception as e:
        print(f"[❌] Error fetching data for {station_id}: {e}")
        return pd.DataFrame()

def main():
    print("📍 Loading NH stations...")
    nh_stations = get_nh_stations(GHCND_STATION_FILE)
    print(f"📡 Found {len(nh_stations)} stations in NH.")

    all_data = []

    for station in nh_stations:
        print(f"→ Pulling data for {station}")
        df = download_noaa_data(station, START_DATE, END_DATE)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("⚠️ No data pulled from any station.")
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    # Save raw pulled data
    timestamp = datetime.now().strftime('%Y%m%d')
    raw_path = os.path.join(RAW_SAVE_DIR, f'noaa_raw_{timestamp}.csv')
    combined_data.to_csv(raw_path, index=False)
    print(f"📝 Saved raw data to {raw_path}")

    # ❗ OVERWRITE feature_data.csv instead of appending
    combined_data.to_csv(FEATURE_CSV, index=False)
    print(f"📁 Created new feature_data.csv with {combined_data.shape[0]} rows.")

if __name__ == "__main__":
    main()
