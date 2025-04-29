# run_forecast.py
# Forecasts next 7 days TMIN, SNOW, and detects Cold Events per station
import os
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBRegressor
# Paths
DATA_DIR = '/workspaces/BlizzardX/Data'
INPUT_CSV = os.path.join(DATA_DIR, 'feature_data_cleaned.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, f'forecast_{datetime.now().strftime("%Y%m%d")}.csv')
# Forecast settings
FORECAST_DAYS = 7
FEATURE_COLS = ['TMIN_Lag1', 'Rolling_Mean_TMIN_7', 'PRCP', 'SNOW', 'SNWD']
# 🧹 Prepare features and handle missing values
def prepare_features(df):
    df = df.copy()
    df['TMIN'] = df.groupby('Station_ID')['TMIN'].transform(lambda x: x.interpolate(method='linear'))
    df['TMAX'] = df.groupby('Station_ID')['TMAX'].transform(lambda x: x.interpolate(method='linear'))
    df['PRCP'] = df['PRCP'].fillna(0)
    df['SNOW'] = df['SNOW'].fillna(0)
    df['SNWD'] = df['SNWD'].fillna(0)
    df['TMIN_Lag1'] = df.groupby('Station_ID')['TMIN'].shift(1)
    df['Rolling_Mean_TMIN_7'] = df.groupby('Station_ID')['TMIN'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    return df.dropna(subset=FEATURE_COLS)
# 🚀 Forecast function per station
def forecast_station(df_station, station_id):
    forecasts = []
    df_station = prepare_features(df_station)
    if df_station.empty:
        print(f"⚠️ No valid data for station {station_id}, skipping...")
        return None
    # Training data
    X = df_station[FEATURE_COLS]
    y_tmin = df_station['TMIN']
    y_snow = df_station['SNOW']
    model_tmin = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_snow = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_tmin.fit(X, y_tmin)
    model_snow.fit(X, y_snow)
    # Use real latest date for forecast start
    start_date = pd.to_datetime(df_station['DATE'].max())
    last_known = df_station.iloc[-1].copy()
    rolling_values = df_station['TMIN'].tail(7).tolist()
    for i in range(1, FORECAST_DAYS + 1):
        forecast_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        features = {
            'TMIN_Lag1': last_known['TMIN'],
            'Rolling_Mean_TMIN_7': sum(rolling_values) / len(rolling_values),
            'PRCP': last_known['PRCP'],
            'SNOW': last_known['SNOW'],
            'SNWD': last_known['SNWD']
        }
        X_pred = pd.DataFrame([features])
        pred_tmin = round(model_tmin.predict(X_pred)[0], 1)
        raw_pred_snow = model_snow.predict(X_pred)[0]
        pred_snow = 0.0 if raw_pred_snow < 0.05 else round(raw_pred_snow, 1)
        forecasts.append({
            'DATE': forecast_date,
            'Station_ID': station_id,
            'Predicted_TMIN': pred_tmin,
            'Predicted_SNOW': pred_snow,
            'Predicted_Cold_Event': int(pred_tmin < -10)
        })
        last_known['TMIN'] = pred_tmin
        last_known['SNOW'] = pred_snow
        rolling_values = rolling_values[1:] + [pred_tmin]
    return forecasts
# 📦 Main runner
def main():
    print("📥 Loading feature data...")
    df = pd.read_csv(INPUT_CSV)
    stations = df['Station_ID'].unique()
    all_forecasts = []
    print(f"🚀 Forecasting for {len(stations)} stations...")
    for station_id in stations:
        df_station = df[df['Station_ID'] == station_id]
        station_forecasts = forecast_station(df_station, station_id)
        if station_forecasts:
            all_forecasts.extend(station_forecasts)
    if all_forecasts:
        df_forecast = pd.DataFrame(all_forecasts)
        df_forecast.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Forecast saved to {OUTPUT_CSV} with {df_forecast.shape[0]} rows.")
    else:
        print("⚠️ No forecasts generated.")
if __name__ == "__main__":
    main()







