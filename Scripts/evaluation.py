# evaluation.py
# Purpose: Compare forecasted TMIN/SNOW with actuals, calculate MAE, RMSE per station

import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import glob

# Paths
DATA_DIR = '/workspaces/BlizzardX/Data'
FEATURE_CSV = os.path.join(DATA_DIR, 'feature_data_cleaned.csv')

# Find latest forecast file automatically
forecast_files = glob.glob(os.path.join(DATA_DIR, 'forecast_*.csv'))
if not forecast_files:
    raise FileNotFoundError("⚠️ No forecast files found in Data directory!")

latest_forecast_file = sorted(forecast_files)[-1]
FORECAST_CSV = latest_forecast_file

print(f"📄 Using latest forecast file: {os.path.basename(FORECAST_CSV)}")


# Load data
print("📥 Loading actual and forecast data...")
df_actual = pd.read_csv(FEATURE_CSV)
df_forecast = pd.read_csv(FORECAST_CSV)

# Align columns
df_actual['DATE'] = pd.to_datetime(df_actual['DATE'])
df_forecast['DATE'] = pd.to_datetime(df_forecast['DATE'])

# Filter actuals to forecasted dates
forecast_dates = df_forecast['DATE'].unique()
df_actual = df_actual[df_actual['DATE'].isin(forecast_dates)]

# Evaluation storage
evaluation_rows = []

stations = df_forecast['Station_ID'].unique()

for station_id in stations:
    df_pred = df_forecast[df_forecast['Station_ID'] == station_id]
    df_true = df_actual[df_actual['Station_ID'] == station_id]

    # Align by DATE
    merged = pd.merge(df_pred, df_true, on=['Station_ID', 'DATE'], suffixes=('_pred', '_true'))

    if merged.empty:
        print(f"⚠️ No matching actual data for {station_id}, skipping...")
        continue

    # Calculate Metrics
    mae_tmin = mean_absolute_error(merged['Predicted_TMIN'], merged['TMIN'])
    rmse_tmin = mean_squared_error(merged['Predicted_TMIN'], merged['TMIN'], squared=False)

    mae_snow = mean_absolute_error(merged['Predicted_SNOW'], merged['SNOW'])
    rmse_snow = mean_squared_error(merged['Predicted_SNOW'], merged['SNOW'], squared=False)

    evaluation_rows.append({
        'Station_ID': station_id,
        'MAE_TMIN': round(mae_tmin, 3),
        'RMSE_TMIN': round(rmse_tmin, 3),
        'MAE_SNOW': round(mae_snow, 3),
        'RMSE_SNOW': round(rmse_snow, 3)
    })

# Save evaluation report
eval_df = pd.DataFrame(evaluation_rows)

forecast_filename = os.path.basename(FORECAST_CSV)  # get forecast_20250425.csv
forecast_date = forecast_filename.split('_')[1].split('.')[0]

eval_report_path = os.path.join(DATA_DIR, f'evaluation_report_{forecast_date}.csv')
eval_df.to_csv(eval_report_path, index=False)

print(f"✅ Evaluation complete! Saved report to {eval_report_path}")
