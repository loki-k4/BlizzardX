import os
import pandas as pd
import joblib
from datetime import datetime, timedelta

# === Config ===
DATA_DIR = '/workspaces/BlizzardX/Data'
MODEL_DIR = '/workspaces/BlizzardX/XGB_Tuned_Models'
INPUT_CSV = os.path.join(DATA_DIR, 'feature_data_cleaned.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, f'forecast_{datetime.now().strftime("%Y%m%d")}.csv')
FORECAST_DAYS = 7
FEATURE_COLS = ['TMIN_Lag1', 'Rolling_Mean_TMIN_7', 'PRCP', 'SNOW', 'SNWD']

# === Feature Engineering ===
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

# === Forecast per station ===
def forecast_station(df_station, station_id):
    forecasts = []
    df_station = prepare_features(df_station)
    if df_station.empty:
        print(f"⚠️ No valid data for station {station_id}, skipping...")
        return None

    # Load models
    tmin_model_path = os.path.join(MODEL_DIR, f"{station_id}_TMIN_xgb_model.pkl")
    snow_model_path = os.path.join(MODEL_DIR, f"{station_id}_SNOW_xgb_model.pkl")
    if not os.path.exists(tmin_model_path) or not os.path.exists(snow_model_path):
        print(f"⚠️ Model files missing for {station_id}, skipping...")
        return None

    model_tmin = joblib.load(tmin_model_path)
    model_snow = joblib.load(snow_model_path)

    # Forecast from today
    start_date = pd.to_datetime(datetime.today().date())
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
        pred_snow = max(0.0, round(raw_pred_snow, 1))

        cold_event = int(pred_tmin < -10)
        extreme_snow = int(pred_snow >= 25)
        snow_expected = int(pred_snow > 0.2)
        prcp_expected = int(last_known['PRCP'] > 0.2)
        snow_probability = min(100, int((pred_snow / (pred_snow + 5)) * 100)) if pred_snow > 0 else 0

        forecasts.append({
            'DATE': forecast_date,
            'Station_ID': station_id,
            'Predicted_TMIN': pred_tmin,
            'Predicted_SNOW': pred_snow,
            'Predicted_Cold_Event': cold_event,
            'Extreme_SNOW_Warning': extreme_snow,
            'Snow_Expected': snow_expected,
            'Precipitation_Expected': prcp_expected,
            'Snow_Probability(%)': snow_probability
        })

        last_known['TMIN'] = pred_tmin
        last_known['SNOW'] = pred_snow
        rolling_values = rolling_values[1:] + [pred_tmin]

    return forecasts

# === Main Runner ===
def main():
    print("📥 Loading feature data...")
    df = pd.read_csv(INPUT_CSV)
    stations = df['Station_ID'].unique()
    all_forecasts = []

    print(f"🔮 Forecasting for {len(stations)} stations...")
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








