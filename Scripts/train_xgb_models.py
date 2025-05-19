import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

# =============================
# BlizzardX - Train XGBoost Models Per Station
# =============================

# Constants
DATA_DIR = '/workspaces/BlizzardX/Data'
FEATURE_CSV = os.path.join(DATA_DIR, 'feature_data.csv')  # Can switch to cleaned CSV if needed
MODEL_DIR = '/workspaces/BlizzardX/XGB_Tuned_Models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
print("📥 Loading feature data for training...")
df = pd.read_csv(FEATURE_CSV)

# Feature Engineering
if 'TMIN_Lag1' not in df.columns:
    df['TMIN_Lag1'] = df.groupby('Station_ID')['TMIN'].shift(1)

if 'Rolling_Mean_TMIN_7' not in df.columns:
    df['Rolling_Mean_TMIN_7'] = df.groupby('Station_ID')['TMIN'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())

# Ensure required columns
required_features = ['TMIN_Lag1', 'Rolling_Mean_TMIN_7', 'TMIN', 'SNOW']
missing_cols = [col for col in required_features if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

stations = df['Station_ID'].unique()
print(f"🏢 Found {len(stations)} unique stations.")

# Train per station
for station_id in stations:
    df_station = df[df['Station_ID'] == station_id].copy()
    df_station['DATE'] = pd.to_datetime(df_station['DATE'])
    df_station = df_station.sort_values('DATE')

    if len(df_station) < 365:
        print(f"⚠️ Not enough data for {station_id} ({len(df_station)} rows), skipping...")
        continue

    # Temporal Split: 70/15/15
    n = len(df_station)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train = df_station.iloc[:train_size]
    val = df_station.iloc[train_size:train_size + val_size]
    test = df_station.iloc[train_size + val_size:]

    # Inputs
    X_train = train[['TMIN_Lag1', 'Rolling_Mean_TMIN_7']]
    y_train_tmin = train['TMIN']
    y_train_snow = train['SNOW']

    X_val = val[['TMIN_Lag1', 'Rolling_Mean_TMIN_7']]
    y_val_tmin = val['TMIN']
    y_val_snow = val['SNOW']

    X_test = test[['TMIN_Lag1', 'Rolling_Mean_TMIN_7']]
    y_test_tmin = test['TMIN']
    y_test_snow = test['SNOW']

    # Train TMIN model
    model_tmin = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model_tmin.fit(X_train, y_train_tmin, eval_set=[(X_val, y_val_tmin)], early_stopping_rounds=10, verbose=False)

    # Train SNOW model
    model_snow = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model_snow.fit(X_train, y_train_snow, eval_set=[(X_val, y_val_snow)], early_stopping_rounds=10, verbose=False)

    # Save models
    joblib.dump(model_tmin, os.path.join(MODEL_DIR, f"{station_id}_TMIN_xgb_model.pkl"))
    joblib.dump(model_snow, os.path.join(MODEL_DIR, f"{station_id}_SNOW_xgb_model.pkl"))

    # Evaluate TMIN
    rmse_train_tmin = np.sqrt(mean_squared_error(y_train_tmin, model_tmin.predict(X_train)))
    rmse_val_tmin = np.sqrt(mean_squared_error(y_val_tmin, model_tmin.predict(X_val)))
    rmse_test_tmin = np.sqrt(mean_squared_error(y_test_tmin, model_tmin.predict(X_test)))

    # Evaluate SNOW
    rmse_train_snow = np.sqrt(mean_squared_error(y_train_snow, model_snow.predict(X_train)))
    rmse_val_snow = np.sqrt(mean_squared_error(y_val_snow, model_snow.predict(X_val)))
    rmse_test_snow = np.sqrt(mean_squared_error(y_test_snow, model_snow.predict(X_test)))

    print(f"\n📊 {station_id} Performance:")
    print(f"   TMIN RMSEs → Train: {rmse_train_tmin:.2f}, Val: {rmse_val_tmin:.2f}, Test: {rmse_test_tmin:.2f}")
    print(f"   SNOW RMSEs → Train: {rmse_train_snow:.2f}, Val: {rmse_val_snow:.2f}, Test: {rmse_test_snow:.2f}")

print("\n🎯 Finished training and evaluation of XGBoost models for all stations.")


