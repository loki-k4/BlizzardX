from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Sample DataFrame
df = pd.read_csv('Data/Cleaned_Final.csv')

df = df.head(30)

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(df.to_dict(orient='records'))  # Convert DataFrame to JSON

if __name__ == '__main__':
    app.run(debug=True, port=5001)