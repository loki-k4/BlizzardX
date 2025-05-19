from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Get the absolute path to the data directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Notebooks')

# Load data
dfs = {
    "New Hampshire": pd.read_csv(os.path.join(data_dir, "NewHampshire_ColdEvent_Enhanced.csv")),
    "Vermont": pd.read_csv(os.path.join(data_dir, "Vermont_ColdEvent_Enhanced.csv"))
}

# Print available columns for debugging
for state, df in dfs.items():
    print(f"{state}: Available columns: {df.columns.tolist()}")

@app.route('/api/states', methods=['GET'])
def get_states():
    return jsonify(list(dfs.keys()))

@app.route('/api/stations', methods=['GET'])
def get_stations():
    state = request.args.get('state')
    if state not in dfs:
        return jsonify([])
    stations = sorted(dfs[state]['Station_ID'].unique().tolist())
    return jsonify(stations)

@app.route('/api/data', methods=['GET'])
def get_data():
    state = request.args.get('state')
    station = request.args.get('station')
    if state not in dfs or not station:
        return jsonify({'data': [], 'stats': {}})
    
    filtered_df = dfs[state][dfs[state]['Station_ID'] == station]
    
    # Calculate statistics
    stats = {
        'predicted_mean': float(filtered_df['Predicted_TMIN'].mean()),
        'actual_mean': float(filtered_df['Actual_TMIN'].mean()),
        'rmse': float(np.sqrt(np.mean((filtered_df['Predicted_TMIN'] - filtered_df['Actual_TMIN']) ** 2))),
        'mae': float(np.mean(np.abs(filtered_df['Predicted_TMIN'] - filtered_df['Actual_TMIN'])))
    }
    
    # Prepare data for visualization
    data = filtered_df[['DATE', 'Predicted_TMIN', 'Actual_TMIN']].rename(
        columns={
            'Predicted_TMIN': 'predicted',
            'Actual_TMIN': 'actual',
            'DATE': 'date'
        }
    ).to_dict('records')
    
    return jsonify({'data': data, 'stats': stats})

@app.route('/api/station_comparison', methods=['GET'])
def station_comparison():
    state = request.args.get('state')
    if state not in dfs:
        return jsonify([])
    
    results = []
    for station in dfs[state]['Station_ID'].unique():
        station_df = dfs[state][dfs[state]['Station_ID'] == station]
        rmse = float(np.sqrt(np.mean((station_df['Predicted_TMIN'] - station_df['Actual_TMIN']) ** 2)))
        results.append({'Station_ID': station, 'rmse': rmse})
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5001) 