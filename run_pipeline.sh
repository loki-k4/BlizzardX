#!/bin/bash

echo "🌤️ Running BlizzardX Weather Forecast Pipeline..."

# Step 1: Pull the latest NOAA data (last 30 days)
echo "📥 Step 1: Pulling NOAA data..."
python3 Scripts/update_feature_data.py

# Step 2: Clean the data and add features
echo "🧹 Step 2: Cleaning and engineering features..."
python3 Scripts/clean_feature_data.py

# Step 3: Generate 7-day forecasts
echo "📈 Step 3: Forecasting next 7 days (TMIN, SNOW, Cold Events)..."
python3 Scripts/run_forecast.py

echo "✅ BlizzardX pipeline completed at $(date)"
