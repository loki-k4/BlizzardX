import os
import pandas as pd
from datetime import datetime, timedelta

# === FILE PATHS ===
DATA_DIR = "/workspaces/BlizzardX/Data"
HISTORICAL_CSV = os.path.join(DATA_DIR, "cleaned_data.csv")
RECENT_CSV = os.path.join(DATA_DIR, "feature_data.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "feature_data_cleaned.csv")
STATION_META_FILE = os.path.join(DATA_DIR, "ghcnd-stations.txt.1")

WEATHER_COLS = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']

# === LOAD STATION METADATA ===
def load_station_metadata(filepath):
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

# === INTERPOLATION + IMPUTATION ===
def interpolate_and_impute(df, column):
    return df.groupby('Station_ID')[column].transform(
        lambda x: x.interpolate(method='linear')
                  .fillna(x.rolling(7, min_periods=1).median())
                  .fillna(x.median())
    )

def clean_and_merge(df_all):
    df_all['DATE'] = pd.to_datetime(df_all['DATE'])
    df_all = df_all.dropna(subset=WEATHER_COLS, how='all')
    for col in WEATHER_COLS:
        df_all[col] = interpolate_and_impute(df_all, col)
    return df_all

# === MAIN PIPELINE ===
def main():
    print("🔄 Loading historical and recent feature data...")
    df_hist = pd.read_csv(HISTORICAL_CSV)
    df_recent = pd.read_csv(RECENT_CSV)

    print("📌 Merging and removing duplicates...")
    df_all = pd.concat([df_hist, df_recent], ignore_index=True)
    df_all.drop_duplicates(subset=['Station_ID', 'DATE'], keep='last', inplace=True)

    print("🧼 Imputing missing values...")
    df_all = clean_and_merge(df_all)

    print("🧽 Filtering to last 30 days...")
    cutoff = df_all['DATE'].max() - pd.Timedelta(days=30)
    df_recent_final = df_all[df_all['DATE'] > cutoff].copy()

    print("🌐 Merging station metadata...")
    station_meta = load_station_metadata(STATION_META_FILE)
    df_final = df_recent_final.drop(columns=['LATITUDE', 'LONGITUDE', 'ELEVATION'], errors='ignore')
    df_final = df_final.merge(station_meta, on='Station_ID', how='left')

    print("🧹 Final cleanup: dropping rows with missing metadata and legacy ID column...")
    df_final = df_final.dropna(subset=["Station_ID", "LATITUDE", "LONGITUDE", "ELEVATION"])
    df_final = df_final.drop(columns=["ID"], errors="ignore")

    print(f"💾 Saving cleaned feature set to: {OUTPUT_CSV}")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done! Final rows: {len(df_final)}, Stations: {df_final['Station_ID'].nunique()}")

if __name__ == "__main__":
    main()

