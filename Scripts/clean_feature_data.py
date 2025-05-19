import os
import pandas as pd

# Paths
DATA_DIR = '/workspaces/BlizzardX/Data'
FEATURE_CSV = os.path.join(DATA_DIR, 'feature_data.csv')
STATION_META_FILE = os.path.join(DATA_DIR, 'ghcnd-stations.txt.1')
OUTPUT_CSV = os.path.join(DATA_DIR, 'feature_data_cleaned.csv')

WEATHER_COLS = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']

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

def interpolate_and_impute(df, column):
    return df.groupby('Station_ID')[column].transform(
        lambda x: x.interpolate(method='linear')\
                  .fillna(x.rolling(7, min_periods=1).median())\
                  .fillna(x.median())
    )

def clean_feature_data(df):
    # Drop rows where all weather fields are missing
    df_cleaned = df.dropna(subset=WEATHER_COLS, how='all').copy()

    # Apply interpolation and imputation for each weather column
    for col in WEATHER_COLS:
        df_cleaned[col] = interpolate_and_impute(df_cleaned, col)

    return df_cleaned

def merge_metadata(df, station_meta_df):
    df = df.drop(columns=['LATITUDE', 'LONGITUDE', 'ELEVATION'], errors='ignore')
    return df.merge(station_meta_df, on='Station_ID', how='left')

def main():
    print("🔍 Loading data...")
    df = pd.read_csv(FEATURE_CSV)
    meta = load_station_metadata(STATION_META_FILE)

    print("🧼 Interpolating and imputing missing values...")
    df_cleaned = clean_feature_data(df)

    print("🌐 Merging station metadata...")
    df_final = merge_metadata(df_cleaned, meta)

    print(f"💾 Saving cleaned data to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Done! Cleaned dataset contains {df_final.shape[0]} rows.")

if __name__ == "__main__":
    main()

