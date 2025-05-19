from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load both CSVs
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'Data')
dfs = {
    "New Hampshire": pd.read_csv(os.path.join(data_dir, "NewHampshire_ColdEvent_Enhanced.csv")),
    "Vermont": pd.read_csv(os.path.join(data_dir, "Vermont_ColdEvent_Enhanced.csv"))
}

# Convert DATE to datetime and extract year
for state, df in dfs.items():
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['Year'] = df['DATE'].dt.year

# Print column names for debugging
for state, df in dfs.items():
    print(f"{state}: Available columns:", df.columns.tolist())

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get list of available metrics"""
    metrics = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
    return jsonify(metrics)

@app.route('/api/years', methods=['GET'])
def get_years():
    """Get list of available years"""
    try:
        years = sorted(dfs["New Hampshire"]['Year'].unique().tolist())
        return jsonify(years)
    except KeyError:
        print("Error: 'Year' column not found. Available columns:", dfs["New Hampshire"].columns.tolist())
        return jsonify([])

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of available locations"""
    try:
        locations = sorted(dfs["New Hampshire"]['NAME'].unique().tolist())
        return jsonify(locations)
    except KeyError:
        print("Error: 'NAME' column not found. Available columns:", dfs["New Hampshire"].columns.tolist())
        return jsonify([])

@app.route('/api/states', methods=['GET'])
def get_states():
    """Get list of available states"""
    return jsonify(list(dfs.keys()))

@app.route('/api/data', methods=['GET'])
def get_data():
    state = request.args.get('state')
    station = request.args.get('station')
    if state not in dfs or not station:
        return jsonify({'data': [], 'stats': {}})
    df = dfs[state]
    filtered_df = df[df['NAME'] == station]
    # Example stats and data (customize as needed)
    stats = {
        'predicted_mean': float(filtered_df['TMIN'].mean()),
        'actual_mean': float(filtered_df['TMAX'].mean()),
        'rmse': float(np.sqrt(np.mean((filtered_df['TMIN'] - filtered_df['TMAX']) ** 2))),
        'mae': float(np.mean(np.abs(filtered_df['TMIN'] - filtered_df['TMAX'])))
    }
    data = filtered_df[['DATE', 'TMIN', 'TMAX']].rename(columns={'TMIN': 'predicted', 'TMAX': 'actual', 'DATE': 'date'}).to_dict('records')
    return jsonify({'data': data, 'stats': stats})

@app.route('/api/distribution', methods=['GET'])
def get_distribution():
    """Get data distribution for a specific metric"""
    metric = request.args.get('metric')
    year = request.args.get('year')
    
    filtered_df = dfs["New Hampshire"].copy()
    if year:
        filtered_df = filtered_df[filtered_df['Year'] == int(year)]
    
    # Create bins for the distribution
    if metric in ['TMIN', 'TMAX']:
        bins = np.linspace(filtered_df[metric].min(), filtered_df[metric].max(), 10)
    elif metric in ['PRCP', 'SNOW', 'SNWD']:
        bins = np.linspace(0, filtered_df[metric].max(), 10)
    else:
        bins = np.linspace(filtered_df[metric].min(), filtered_df[metric].max(), 10)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(filtered_df[metric], bins=bins)
    
    # Format data for visualization
    distribution_data = []
    for i in range(len(hist)):
        distribution_data.append({
            'range': f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}',
            'count': int(hist[i])
        })
    
    return jsonify(distribution_data)

@app.route('/api/stations', methods=['GET'])
def get_stations():
    state = request.args.get('state')
    if state not in dfs:
        return jsonify([])
    stations = sorted(dfs[state]['NAME'].unique().tolist())
    return jsonify(stations)

@app.route('/api/station_comparison', methods=['GET'])
def station_comparison():
    state = request.args.get('state')
    if state not in dfs:
        return jsonify([])
    df = dfs[state]
    results = []
    for station in df['NAME'].unique():
        station_df = df[df['NAME'] == station]
        # Dummy RMSE calculation (replace with your actual logic)
        rmse = float(np.sqrt(np.mean((station_df['TMIN'] - station_df['TMAX']) ** 2)))
        results.append({'Station_ID': station, 'rmse': rmse})
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5001) 