# run_arima_forecast.py
# Forecasts next 7 days (backtesting or real) TMIN for each station using ARIMA

import os
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

# Paths
DATA_DIR = '/workspaces/BlizzardX/Data'
FEATURE_CSV = os.path.join(DATA_DIR, 'feature_data_cleaned.csv')

# Settings
FORECAST_DAYS = 7
BACKTESTING = False  # Keep False, ARIMA will always predict future from latest

# 📦 Main runner
def main():
    print("📥 Loading feature data...")
    df = pd.read_csv(FEATURE_CSV)

    stations = df['Station_ID'].unique()
    all_forecasts = []

    print(f"🚀 ARIMA Forecasting for {len(stations)} stations...")

    for station_id in stations:
        df_station = df[df['Station_ID'] == station_id].copy()

        # Prepare station data
        if df_station.shape[0] < 20:  # ARIMA needs some minimum data
            print(f"⚠️ Not enough data for {station_id}, skipping...")
            continue

        df_station = df_station.sort_values('DATE')
        series = df_station['TMIN'].interpolate(method='linear').fillna(method='bfill')

        try:
            # Fit simple ARIMA(2,1,2)
            model = ARIMA(series, order=(2, 1, 2))
            model_fit = model.fit()

            last_date = pd.to_datetime(df_station['DATE'].max())

            for i in range(1, FORECAST_DAYS + 1):
                forecast_date = (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
                pred_tmin = model_fit.forecast(steps=i)[-1]
                pred_tmin = round(pred_tmin, 1)

                all_forecasts.append({
                    'DATE': forecast_date,
                    'Station_ID': station_id,
                    'Predicted_TMIN_ARIMA': pred_tmin
                })

        except Exception as e:
            print(f"⚠️ Error fitting ARIMA for {station_id}: {e}")
            continue

    if all_forecasts:
        df_forecast = pd.DataFrame(all_forecasts)

        output_csv = os.path.join(DATA_DIR, f'arima_forecast_{datetime.now().strftime("%Y%m%d")}.csv')
        df_forecast.to_csv(output_csv, index=False)

        print(f"✅ ARIMA Forecast saved to {output_csv} with {df_forecast.shape[0]} rows.")
    else:
        print("⚠️ No ARIMA forecasts generated.")

if __name__ == "__main__":
    main()
