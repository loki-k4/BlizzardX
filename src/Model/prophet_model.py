import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class ProphetForecaster:
    def __init__(self, data_path, model_dir='/workspaces/BlizzardX/Models/prophet'):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp()
        os.environ['CMDSTAN'] = self.temp_dir
        os.environ['STAN_THREADS'] = '1'
        
        self.df = None
        self.df_prophet = None
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.regressors = [
            'TMAX', 'PRCP', 'TMAX_Lag1', 'ColdEvent_Lag1', 'Seasonal_TMIN_Anomaly',
            'Season_Spring', 'Season_Summer', 'Season_Fall', 'Season_Winter'
        ]
        self.results = []
        self.train_losses = []
        self.val_losses = []
        self.param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
            'seasonality_prior_scale': [1.0, 5.0, 10.0, 20.0],
            'holidays_prior_scale': [0.1],
            'seasonality_mode': ['additive', 'multiplicative']
        }

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])
        self.df['Year'] = self.df['DATE'].dt.year
        self.df = self.df.sort_values('DATE')

        # Temporal split
        max_date = self.df['DATE'].max()
        max_year = max_date.year
        test_year = max_year
        val_years = [max_year - 1, max_year - 2]
        train_years_end = max_year - 3

        train_data = self.df[self.df['Year'] <= train_years_end]
        val_data = self.df[self.df['Year'].isin(val_years)]
        test_data = self.df[self.df['Year'] == test_year]

        self.df = pd.concat([train_data, val_data, test_data]).reset_index(drop=True)

        # Feature engineering
        self.df['TMAX_Lag1'] = self.df.groupby('Station_ID')['TMAX'].shift(1)
        self.df['ColdEvent_Lag1'] = self.df.groupby('Station_ID')['Cold_Event'].shift(1).fillna(0)
        self.df['TMAX_Lag1'] = self.df['TMAX_Lag1'].fillna(self.df['TMAX'].mean())

        if 'Season' not in self.df.columns:
            self.df['Season'] = self.df['DATE'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

        self.df = pd.get_dummies(self.df, columns=['Season'], prefix='Season')
        if self.df['Day_of_Week'].dtype == 'object':
            self.df = pd.get_dummies(self.df, columns=['Day_of_Week'], prefix='Day_of_Week')

        self.df_prophet = self.df.rename(columns={'DATE': 'ds', 'TMIN': 'y'})
        self.df_prophet['ds'] = pd.to_datetime(self.df_prophet['ds'])

        # Filter regressors to those present in the DataFrame
        self.regressors = [r for r in self.regressors if r in self.df_prophet.columns]

        # Clean and scale
        self.df_prophet[self.regressors] = self.df_prophet[self.regressors].replace([np.inf, -np.inf], np.nan)
        self.df_prophet['y'] = self.df_prophet['y'].replace([np.inf, -np.inf], np.nan)
        self.df_prophet[self.regressors] = self.df_prophet[self.regressors].fillna(self.df_prophet[self.regressors].mean())
        self.df_prophet['y'] = self.df_prophet['y'].fillna(self.df_prophet['y'].mean())

        self.df_prophet[self.regressors] = self.scaler.fit_transform(self.df_prophet[self.regressors])
        self.df_prophet[self.regressors] = self.df_prophet[self.regressors].clip(lower=-5, upper=5)
        self.df_prophet['y'] = self.y_scaler.fit_transform(self.df_prophet[['y']])

    def tune_prophet(self, train, val, regressors):
        best_model = None
        best_mae = float('inf')
        best_params = None
        self.train_losses = []
        self.val_losses = []

        for combo in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), combo))
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                mcmc_samples=0
            )
            model.add_seasonality(name='yearly', period=365.25, fourier_order=5)
            for reg in regressors:
                model.add_regressor(reg)

            try:
                model.fit(train[['ds', 'y'] + regressors])
                
                # Training loss
                train_forecast = model.predict(train[['ds'] + regressors])
                train_mae = mean_absolute_error(train['y'], train_forecast['yhat'])
                self.train_losses.append(train_mae)
                
                # Validation loss
                val_forecast = model.predict(val[['ds'] + regressors])
                val_mae = mean_absolute_error(val['y'], val_forecast['yhat'])
                self.val_losses.append(val_mae)

                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model = model
                    best_params = params
            except Exception as e:
                print(f"Tuning error with params {params}: {e}")
                continue

        return best_model, best_params, best_mae

    def process_station(self, station):
        print(f"Processing Station: {station}")
        station_df = self.df_prophet[self.df_prophet['Station_ID'] == station].copy()
        
        # Temporal split within station
        train_df = station_df[station_df['Year'] <= (max(station_df['Year']) - 3)]
        val_df = station_df[station_df['Year'].isin([(max(station_df['Year']) - 2), (max(station_df['Year']) - 1)])]
        test_df = station_df[station_df['Year'] == max(station_df['Year'])]

        if train_df.empty or val_df.empty or test_df.empty:
            print(f"Skipping {station}: Insufficient data")
            return None

        valid_regressors = [r for r in self.regressors if station_df[r].var() > 1e-6]
        if not valid_regressors or station_df['y'].var() < 1e-6:
            print(f"Skipping {station}: Invalid regressors or low target variance")
            return None

        for df in [station_df, train_df, val_df, test_df]:
            df[valid_regressors] = df[valid_regressors].fillna(df[valid_regressors].mean())

        model, best_params, val_mae = self.tune_prophet(train_df, val_df, valid_regressors)
        if model is None:
            return None

        model_path = os.path.join(self.model_dir, f"prophet_{station}.pkl")
        joblib.dump(model, model_path)

        # Predictions on test data
        forecast = model.predict(test_df[['ds'] + valid_regressors])
        test_pred = forecast[['ds', 'yhat']].merge(test_df[['ds', 'y']], on='ds')
        test_pred['yhat'] = self.y_scaler.inverse_transform(test_pred[['yhat']])
        test_pred['y'] = self.y_scaler.inverse_transform(test_pred[['y']])

        test_mae = mean_absolute_error(test_pred['y'], test_pred['yhat'])
        test_rmse = np.sqrt(mean_squared_error(test_pred['y'], test_pred['yhat']))
        mask = abs(test_pred['y']) > 0.1
        mape = np.mean(np.abs((test_pred['y'][mask] - test_pred['yhat'][mask]) / test_pred['y'][mask])) * 100 if mask.any() else np.nan
        smape = np.mean(2 * np.abs(test_pred['y'] - test_pred['yhat']) / (np.abs(test_pred['y']) + np.abs(test_pred['yhat'])) * 100)

        return {
            'Station_ID': station,
            'Best_Params': best_params,
            'Validation_MAE': val_mae,
            'Test_MAE': test_mae,
            'Test_RMSE': test_rmse,
            'Test_MAPE': mape,
            'Test_SMAPE': smape,
            'Test_Samples': len(test_pred)
        }

    def run_all_stations(self):
        stations = self.df_prophet['Station_ID'].unique()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_station, s) for s in stations]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    self.results.append(result)
        return pd.DataFrame(self.results)

    def plot_station_forecast(self, station_id):
        station_df = self.df_prophet[self.df_prophet['Station_ID'] == station_id].copy()
        train_df = station_df[station_df['Year'] <= (max(station_df['Year']) - 3)]
        val_df = station_df[station_df['Year'].isin([(max(station_df['Year']) - 2), (max(station_df['Year']) - 1)])]
        test_df = station_df[station_df['Year'] == max(station_df['Year'])]

        if train_df.empty or val_df.empty or test_df.empty:
            print(f"Not enough data for station {station_id} to plot.")
            return

        valid_regressors = [r for r in self.regressors if station_df[r].var() > 1e-6]
        if not valid_regressors or station_df['y'].var() < 1e-6:
            print(f"Invalid regressors or low variance for station {station_id}")
            return

        # Load the saved model
        model_path = os.path.join(self.model_dir, f"prophet_{station_id}.pkl")
        if not os.path.exists(model_path):
            print(f"No saved model found for station {station_id} at {model_path}")
            return

        model = joblib.load(model_path)

        # Predict on test data
        forecast = model.predict(test_df[['ds'] + valid_regressors])

        # Inverse transform for test data results
        test_df['y'] = self.y_scaler.inverse_transform(test_df[['y']])
        forecast['yhat'] = self.y_scaler.inverse_transform(forecast[['yhat']]).round(2)
        forecast['yhat_lower'] = self.y_scaler.inverse_transform(forecast[['yhat_lower']]).round(2)
        forecast['yhat_upper'] = self.y_scaler.inverse_transform(forecast[['yhat_upper']]).round(2)

        # Prepare test results DataFrame
        test_results_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(test_df[['ds', 'y']], on='ds')
        test_results_df['ds'] = pd.to_datetime(test_results_df['ds'])

        # Print test forecast results
        print(f"\n📈 Forecast Results for Station {station_id} (Test Data):")
        print(test_results_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

        # Create future DataFrame for 7 days from current date
        current_date = pd.Timestamp.now().normalize()
        future_dates = pd.date_range(start=current_date, periods=7, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

        # Populate regressors with historical means or last known values
        for reg in valid_regressors:
            if reg in ['Season_Spring', 'Season_Summer', 'Season_Fall', 'Season_Winter']:
                future_df[reg] = 0
                for date in future_df['ds']:
                    month = date.month
                    if month in [3, 4, 5] and reg == 'Season_Spring':
                        future_df.loc[future_df['ds'] == date, reg] = 1
                    elif month in [6, 7, 8] and reg == 'Season_Summer':
                        future_df.loc[future_df['ds'] == date, reg] = 1
                    elif month in [9, 10, 11] and reg == 'Season_Fall':
                        future_df.loc[future_df['ds'] == date, reg] = 1
                    elif month in [12, 1, 2] and reg == 'Season_Winter':
                        future_df.loc[future_df['ds'] == date, reg] = 1
            else:
                # Use mean of the regressor from the station's historical data
                future_df[reg] = station_df[reg].mean()

        # Scale regressors
        future_df[valid_regressors] = self.scaler.transform(future_df[valid_regressors])
        future_df[valid_regressors] = future_df[valid_regressors].clip(lower=-5, upper=5)

        # Predict for future 7 days
        future_forecast = model.predict(future_df[['ds'] + valid_regressors])

        # Inverse transform for future forecast
        future_forecast['yhat'] = self.y_scaler.inverse_transform(future_forecast[['yhat']]).round(2)
        future_forecast['yhat_lower'] = self.y_scaler.inverse_transform(future_forecast[['yhat_lower']]).round(2)
        future_forecast['yhat_upper'] = self.y_scaler.inverse_transform(future_forecast[['yhat_upper']]).round(2)

        # Prepare future results DataFrame
        future_results_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_results_df['ds'] = pd.to_datetime(future_results_df['ds'])

        # Print future forecast results
        print(f"\n📈 7-Day Forecast for Station {station_id} (Starting from {current_date.date()}):")
        print(future_results_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

        # Plotting only test data
        plt.figure(figsize=(12, 6))
        plt.scatter(test_df['ds'], test_df['y'], label='Actual TMIN', alpha=0.5, color='black')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted TMIN', color='blue')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                         color='blue', alpha=0.2, label='Confidence Interval')
        plt.title(f"TMIN Forecast for Station {station_id} (Test Data)")
        plt.xlabel("Date")
        plt.ylabel("TMIN (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()