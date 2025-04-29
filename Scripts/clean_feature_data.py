# clean_feature_data.py
# Cleans feature_data.csv by handling missing weather values and enriching with station metadata

import os
import pandas as pd

# Paths
DATA_DIR = '/workspaces/BlizzardX/Data'
FEATURE_CSV = os.path.join(DATA_DIR, 'feature_data.csv')
STATION_META_FILE = os.path.join(DATA_DIR, 'ghcnd-stations.txt')
OUTPUT_CSV = os.path.join(DATA_DIR, 'feature_data_cleaned.csv')

# Columns used for filtering
WEATHER_COLS = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']


def load_station_metadata(filepath):
    """Parses ghcnd-stations.txt and returns a DataFrame with metadata."""
    stations = []
    with open(filepath, 'r') as file:
        for line in file:
            station_id = line[0:11].strip()
            lat = float(line[12:20].strip())
            lon = float(line[21:30].strip())
            elev = float(line[31:37].strip())
            name = line[41:71].strip()
            stations.append([station_id, lat, lon, elev, name])
    return pd.DataFrame(stations, columns=['Station_ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATION_NAME'])


def clean_feature_data(df):
    """Cleans feature data by removing invalid rows and optionally imputing."""
    # Drop rows where all key weather fields are missing
    df_cleaned = df.dropna(subset=WEATHER_COLS, how='all')

    # Optional: Impute TMIN/TMAX using rolling median per station
    df_cleaned['TMIN'] = df_cleaned.groupby('Station_ID')['TMIN'].transform(
        lambda x: x.fillna(x.rolling(7, min_periods=1).median()))
    df_cleaned['TMAX'] = df_cleaned.groupby('Station_ID')['TMAX'].transform(
        lambda x: x.fillna(x.rolling(7, min_periods=1).median()))

    return df_cleaned


def merge_metadata(df, station_meta_df):
    """Merges station lat/lon/elevation into the main dataset."""
    df = df.drop(columns=['LATITUDE', 'LONGITUDE', 'ELEVATION'], errors='ignore')
    merged = df.merge(station_meta_df, on='Station_ID', how='left')
    return merged


def main():
    print("🔍 Loading data...")
    df = pd.read_csv(FEATURE_CSV)
    meta = load_station_metadata(STATION_META_FILE)

    print("🧹 Cleaning feature data...")
    df_cleaned = clean_feature_data(df)

    print("🌐 Merging station metadata...")
    df_final = merge_metadata(df_cleaned, meta)

    print(f"💾 Saving cleaned data to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Cleaning complete! Rows in cleaned file: {df_final.shape[0]}")


if __name__ == "__main__":
    main()
