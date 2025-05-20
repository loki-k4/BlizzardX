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
from math import radians, sin, cos, sqrt, atan2
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging for console output only
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.handlers = [console_handler]

class ProphetForecaster:
    def __init__(self, data_path=None, model_dir=None, plot_dir=None):
        # Dynamically set the project root based on script or notebook location
        self.project_root = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd()
        
        # Default paths relative to project root
        self.data_path = Path(data_path).resolve() if data_path else self.project_root / 'Data' / 'feature_data.csv'
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.model_dir = Path(model_dir).resolve() if model_dir else self.project_root / 'model' / 'prophet'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_dir = Path(plot_dir).resolve() if plot_dir else self.project_root / 'plots' / 'prophet'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(tempfile.mkdtemp())
        os.environ['CMDSTAN'] = str(self.temp_dir)
        os.environ['STAN_THREADS'] = '1'
        
        self.df = None
        self.df_prophet = None
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.regressors = [
            'TMAX', 'PRCP', 'SNOW', 'SNWD', 'Cold_Event',
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
        self.station_coords = {}
        self.station_distances = {}
        logger.info(f"Initialized ProphetForecaster with data path: {self.data_path}, model directory: {self.model_dir}, plot directory: {self.plot_dir}")

    def calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero with a small constant
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
        return np.mean(100 * np.abs(y_true - y_pred) / denominator)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def calculate_station_distances(self):
        """Compute distances between all pairs of stations."""
        stations = self.df['Station_ID'].unique()
        self.station_coords = {
            station: (
                self.df[self.df['Station_ID'] == station]['LATITUDE'].iloc[0],
                self.df[self.df['Station_ID'] == station]['LONGITUDE'].iloc[0]
            ) for station in stations
        }
        
        for station1 in stations:
            self.station_distances[station1] = {}
            lat1, lon1 = self.station_coords[station1]
            for station2 in stations:
                if station1 != station2:
                    lat2, lon2 = self.station_coords[station2]
                    distance = self.haversine_distance(lat1, lon1, lat2, lon2)
                    self.station_distances[station1][station2] = distance
        logger.info(f"Calculated distances for {len(stations)} stations")

    def find_nearest_station(self, target_station):
        """Find the nearest station to the target station."""
        if target_station not in self.station_distances:
            logger.warning(f"No nearby stations found for {target_station}")
            return None
        distances = self.station_distances[target_station]
        nearest = min(distances, key=distances.get)
        logger.info(f"Nearest station to {target_station} is {nearest}")
        return nearest

    def load_and_prepare_data(self):
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])
        self.df = self.df.sort_values('DATE')

        required_columns = ['Station_ID', 'LATITUDE', 'LONGITUDE', 'DATE', 'TMIN', 'TMAX', 'PRCP']
        missing_required = [col for col in required_columns if col not in self.df.columns]
        if missing_required:
            raise ValueError(f"Dataset must contain columns: {missing_required}")
        
        self.regressors = [r for r in self.regressors if r in self.df.columns or 'Season_' in r]
        logger.info(f"Using regressors: {self.regressors}")

        self.calculate_station_distances()

        if 'Season' not in self.df.columns:
            self.df['Season'] = self.df['DATE'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

        self.df = pd.get_dummies(self.df, columns=['Season'], prefix='Season')
        if 'Day_of_Week' in self.df.columns and self.df['Day_of_Week'].dtype == 'object':
            self.df = pd.get_dummies(self.df, columns=['Day_of_Week'], prefix='Day_of_Week')

        self.df_prophet = self.df.rename(columns={'DATE': 'ds', 'TMIN': 'y'})
        self.df_prophet['ds'] = pd.to_datetime(self.df_prophet['ds'])

        self.regressors = [r for r in self.regressors if r in self.df_prophet.columns]
        logger.info(f"Final regressors: {self.regressors}")

        self.df_prophet[self.regressors] = self.df_prophet[self.regressors].replace([np.inf, -np.inf], np.nan)
        self.df_prophet['y'] = self.df_prophet['y'].replace([np.inf, -np.inf], np.nan)
        self.df_prophet[self.regressors] = self.df_prophet[self.regressors].fillna(self.df_prophet[self.regressors].mean())
        self.df_prophet['y'] = self.df_prophet['y'].fillna(self.df_prophet['y'].mean())

        self.df_prophet[self.regressors] = self.scaler.fit_transform(self.df_prophet[self.regressors])
        self.df_prophet[self.regressors] = self.df_prophet[self.regressors].clip(lower=-5, upper=5)
        self.df_prophet['y'] = self.y_scaler.fit_transform(self.df_prophet[['y']])
        logger.info("Data preparation completed")

    def tune_prophet(self, train, regressors, cv_folds=5, cv_horizon_days=180):
        """Perform time-series cross-validation for hyperparameter tuning."""
        best_model = None
        best_mae = float('inf')
        best_rmse = float('inf')
        best_smape = float('inf')
        best_params = None
        self.train_losses = []
        self.val_losses = []

        train = train.sort_values('ds')
        min_date = train['ds'].min()
        max_date = train['ds'].max()
        total_days = (max_date - min_date).days
        fold_size = total_days // (cv_folds + 1)

        logger.info(f"Starting cross-validation with {cv_folds} folds, horizon {cv_horizon_days} days")

        for combo in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), combo))
            cv_mae = []
            cv_rmse = []
            cv_smape = []
            cv_train_mae = []
            cv_train_rmse = []
            cv_train_smape = []

            for fold in range(cv_folds):
                fold_end = min_date + pd.Timedelta(days=fold_size * (fold + 1))
                fold_horizon_end = fold_end + pd.Timedelta(days=cv_horizon_days)
                if fold_horizon_end > max_date:
                    logger.warning(f"Fold {fold+1} skipped: horizon exceeds max date")
                    continue

                fold_train = train[train['ds'] <= fold_end]
                fold_val = train[(train['ds'] > fold_end) & (train['ds'] <= fold_horizon_end)]

                if fold_train.empty or fold_val.empty:
                    logger.warning(f"Fold {fold+1} skipped: insufficient data")
                    continue

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
                    model.fit(fold_train[['ds', 'y'] + regressors])
                    
                    train_forecast = model.predict(fold_train[['ds'] + regressors])
                    train_y = self.y_scaler.inverse_transform(fold_train[['y']])
                    train_yhat = self.y_scaler.inverse_transform(train_forecast[['yhat']])
                    train_mae = mean_absolute_error(train_y, train_yhat)
                    train_rmse = np.sqrt(mean_squared_error(train_y, train_yhat))
                    train_smape = self.calculate_smape(train_y, train_yhat)
                    cv_train_mae.append(train_mae)
                    cv_train_rmse.append(train_rmse)
                    cv_train_smape.append(train_smape)
                    
                    val_forecast = model.predict(fold_val[['ds'] + regressors])
                    val_y = self.y_scaler.inverse_transform(fold_val[['y']])
                    val_yhat = self.y_scaler.inverse_transform(val_forecast[['yhat']])
                    val_mae = mean_absolute_error(val_y, val_yhat)
                    val_rmse = np.sqrt(mean_squared_error(val_y, val_yhat))
                    val_smape = self.calculate_smape(val_y, val_yhat)
                    cv_mae.append(val_mae)
                    cv_rmse.append(val_rmse)
                    cv_smape.append(val_smape)
                except Exception as e:
                    logger.error(f"Tuning error with params {params} in fold {fold+1}: {e}")
                    continue

            if cv_mae:
                avg_val_mae = np.mean(cv_mae)
                avg_val_rmse = np.mean(cv_rmse)
                avg_val_smape = np.mean(cv_smape)
                avg_train_mae = np.mean(cv_train_mae) if cv_train_mae else np.nan
                avg_train_rmse = np.mean(cv_train_rmse) if cv_train_rmse else np.nan
                avg_train_smape = np.mean(cv_train_smape) if cv_train_smape else np.nan
                self.train_losses.append(avg_train_mae)
                self.val_losses.append(avg_val_mae)
                if avg_val_mae < best_mae:
                    best_mae = avg_val_mae
                    best_rmse = avg_val_rmse
                    best_smape = avg_val_smape
                    best_params = params
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
                    model.fit(train[['ds', 'y'] + regressors])
                    best_model = model
                logger.info(f"Evaluated params {params}: Validation MAE = {avg_val_mae:.4f}°C, RMSE = {avg_val_rmse:.4f}°C, SMAPE = {avg_val_smape:.4f}%")

        if best_model is None:
            logger.warning("No valid model found during cross-validation")
        return best_model, best_params, best_mae, best_rmse, best_smape

    def process_station(self, station):
        logger.info(f"Processing station {station}")
        station_df = self.df_prophet[self.df_prophet['Station_ID'] == station].copy()
        
        if station_df.empty:
            logger.warning(f"Skipping {station}: Insufficient data")
            return None

        valid_regressors = [r for r in self.regressors if station_df[r].var() > 1e-6]
        if not valid_regressors or station_df['y'].var() < 1e-6:
            logger.warning(f"Skipping {station}: Invalid regressors or low target variance")
            return None

        station_df[valid_regressors] = station_df[valid_regressors].fillna(station_df[valid_regressors].mean())

        model, best_params, val_mae, val_rmse, val_smape = self.tune_prophet(station_df, valid_regressors)
        if model is None:
            return None

        model_path = self.model_dir / f"prophet_{station}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model for {station} at {model_path}")

        return {
            'Station_ID': station,
            'Best_Params': best_params,
            'Validation_MAE': val_mae,
            'Validation_RMSE': val_rmse,
            'Validation_SMAPE': val_smape,
            'Train_Samples': len(station_df)
        }

    def run_all_stations(self):
        stations = self.df_prophet['Station_ID'].unique()
        logger.info(f"Processing {len(stations)} stations")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_station, s) for s in stations]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    self.results.append(result)
        logger.info("Completed processing all stations")
        return pd.DataFrame(self.results)

    def meta_model_predict(self, station_id, start_date, periods=7):
        """Predict using the model of the station itself or the nearest station if no model exists."""
        model_path = self.model_dir / f"prophet_{station_id}.pkl"
        if model_path.exists():
            logger.info(f"Using model for station {station_id}")
            model_station = station_id
        else:
            logger.info(f"No model found for station {station_id}, using nearest station")
            model_station = self.find_nearest_station(station_id)
            if not model_station:
                logger.warning(f"No nearby stations found for {station_id}")
                return None
            model_path = self.model_dir / f"prophet_{model_station}.pkl"
            if not model_path.exists():
                logger.warning(f"No saved model found for station {model_station} at {model_path}")
                return None

        model = joblib.load(model_path)
        station_df = self.df_prophet[self.df_prophet['Station_ID'] == station_id].copy()
        valid_regressors = [r for r in self.regressors if station_df[r].var() > 1e-6]
        logger.info(f"Using regressors for prediction: {valid_regressors}")

        start_date = pd.to_datetime(start_date)
        future_dates = pd.date_range(start=start_date, periods=periods, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

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
                future_df[reg] = station_df[reg].mean()

        future_df[valid_regressors] = self.scaler.transform(future_df[valid_regressors])
        future_df[valid_regressors] = future_df[valid_regressors].clip(lower=-5, upper=5)

        forecast = model.predict(future_df[['ds'] + valid_regressors])
        forecast['yhat'] = self.y_scaler.inverse_transform(forecast[['yhat']]).round(2)
        forecast['yhat_lower'] = self.y_scaler.inverse_transform(forecast[['yhat_lower']]).round(2)
        forecast['yhat_upper'] = self.y_scaler.inverse_transform(forecast[['yhat_upper']]).round(2)

        result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result_df['ds'] = pd.to_datetime(result_df['ds'])
        
        logger.info(f"Generated {periods}-day forecast for station {station_id} using model from {model_station}")
        print(result_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))
        
        return result_df

    def test_meta_model_stations(self, station_ids):
        """Test the meta-model on specified stations using their own models if available, otherwise nearest station's model."""
        results = []
        for station_id in station_ids:
            logger.info(f"Testing station {station_id}")
            station_df = self.df_prophet[self.df_prophet['Station_ID'] == station_id].copy()
            
            max_date = station_df['ds'].max()
            test_start = max_date - pd.Timedelta(days=180)
            test_df = station_df[station_df['ds'] >= test_start]

            if test_df.empty:
                logger.warning(f"Not enough test data for station {station_id}")
                continue

            valid_regressors = [r for r in self.regressors if station_df[r].var() > 1e-6]
            if not valid_regressors or station_df['y'].var() < 1e-6:
                logger.warning(f"Invalid regressors or low variance for station {station_id}")
                continue

            model_path = self.model_dir / f"prophet_{station_id}.pkl"
            if model_path.exists():
                logger.info(f"Using model for station {station_id}")
                model_station = station_id
            else:
                logger.info(f"No model found for station {station_id}, using nearest station")
                model_station = self.find_nearest_station(station_id)
                if not model_station:
                    logger.warning(f"No nearby stations found for {station_id}")
                    continue
                model_path = self.model_dir / f"prophet_{model_station}.pkl"
                if not model_path.exists():
                    logger.warning(f"No saved model found for station {model_station} at {model_path}")
                    continue

            model = joblib.load(model_path)

            forecast = model.predict(test_df[['ds'] + valid_regressors])

            test_df['y'] = self.y_scaler.inverse_transform(test_df[['y']])
            forecast['yhat'] = self.y_scaler.inverse_transform(forecast[['yhat']]).round(2)
            forecast['yhat_lower'] = self.y_scaler.inverse_transform(forecast[['yhat_lower']]).round(2)
            forecast['yhat_upper'] = self.y_scaler.inverse_transform(forecast[['yhat_upper']]).round(2)

            test_results_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(test_df[['ds', 'y']], on='ds')
            test_results_df['ds'] = pd.to_datetime(test_results_df['ds'])

            logger.info(f"Generated test forecast for station {station_id} using model from {model_station}")
            print(test_results_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

            test_mae = mean_absolute_error(test_results_df['y'], test_results_df['yhat'])
            test_rmse = np.sqrt(mean_squared_error(test_results_df['y'], test_results_df['yhat']))
            test_smape = self.calculate_smape(test_results_df['y'], test_results_df['yhat'])
            logger.info(f"Station {station_id} - Test MAE: {test_mae:.2f}°C, RMSE: {test_rmse:.2f}°C, SMAPE: {test_smape:.2f}%")

            results.append({
                'Station_ID': station_id,
                'Model_Station': model_station,
                'Test_MAE': test_mae,
                'Test_RMSE': test_rmse,
                'Test_SMAPE': test_smape,
                'Test_Samples': len(test_results_df)
            })

            plt.figure(figsize=(12, 6))
            plt.scatter(test_df['ds'], test_df['y'], label='Actual TMIN (°C)', alpha=0.5, color='black')
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted TMIN (°C)', color='blue')
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                             color='blue', alpha=0.2, label='Confidence Interval')
            plt.title(f"TMIN Forecast for Station {station_id} using model from Station {model_station} (Test Data - Last 6 Months)")
            plt.xlabel("Date")
            plt.ylabel("TMIN (°C)")
            plt.legend()
            plt.tight_layout()
            plot_path = self.plot_dir / f'plots_{station_id}.png'
            plt.savefig(plot_path)
            logger.info(f"Saved forecast plot for station {station_id} to {plot_path}")
            plt.close()

        return pd.DataFrame(results)

    def plot_station_forecast(self, station_id):
        logger.info(f"Plotting forecast for station {station_id}")
        station_df = self.df_prophet[self.df_prophet['Station_ID'] == station_id].copy()
        
        max_date = station_df['ds'].max()
        test_start = max_date - pd.Timedelta(days=180)
        train_df = station_df[station_df['ds'] < test_start]
        test_df = station_df[station_df['ds'] >= test_start]

        if train_df.empty or test_df.empty:
            logger.warning(f"Not enough data for station {station_id} to plot")
            return

        valid_regressors = [r for r in self.regressors if station_df[r].var() > 1e-6]
        if not valid_regressors or station_df['y'].var() < 1e-6:
            logger.warning(f"Invalid regressors or low variance for station {station_id}")
            return

        model_path = self.model_dir / f"prophet_{station_id}.pkl"
        if model_path.exists():
            logger.info(f"Using model for station {station_id}")
            model_station = station_id
        else:
            logger.info(f"No model found for station {station_id}, using nearest station")
            model_station = self.find_nearest_station(station_id)
            if not model_station:
                logger.warning(f"No nearby stations found for {station_id}")
                return
            model_path = self.model_dir / f"prophet_{model_station}.pkl"
            if not model_path.exists():
                logger.warning(f"No saved model found for station {model_station} at {model_path}")
                return

        model = joblib.load(model_path)

        forecast = model.predict(test_df[['ds'] + valid_regressors])

        test_df['y'] = self.y_scaler.inverse_transform(test_df[['y']])
        forecast['yhat'] = self.y_scaler.inverse_transform(forecast[['yhat']]).round(2)
        forecast['yhat_lower'] = self.y_scaler.inverse_transform(forecast[['yhat_lower']]).round(2)
        forecast['yhat_upper'] = self.y_scaler.inverse_transform(forecast[['yhat_upper']]).round(2)

        test_results_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(test_df[['ds', 'y']], on='ds')
        test_results_df['ds'] = pd.to_datetime(test_results_df['ds'])

        logger.info(f"Generated test forecast for station {station_id} using model from {model_station}")
        print(test_results_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

        test_mae = mean_absolute_error(test_results_df['y'], test_results_df['yhat'])
        test_rmse = np.sqrt(mean_squared_error(test_results_df['y'], test_results_df['yhat']))
        test_smape = self.calculate_smape(test_results_df['y'], test_results_df['yhat'])
        logger.info(f"Station {station_id} - Test MAE: {test_mae:.2f}°C, RMSE: {test_rmse:.2f}°C, SMAPE: {test_smape:.2f}%")

        current_date = pd.Timestamp.now().normalize()
        future_dates = pd.date_range(start=current_date, periods=7, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

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
                future_df[reg] = station_df[reg].mean()

        future_df[valid_regressors] = self.scaler.transform(future_df[valid_regressors])
        future_df[valid_regressors] = future_df[valid_regressors].clip(lower=-5, upper=5)

        future_forecast = model.predict(future_df[['ds'] + valid_regressors])

        future_forecast['yhat'] = self.y_scaler.inverse_transform(future_forecast[['yhat']]).round(2)
        future_forecast['yhat_lower'] = self.y_scaler.inverse_transform(future_forecast[['yhat_lower']]).round(2)
        future_forecast['yhat_upper'] = self.y_scaler.inverse_transform(future_forecast[['yhat_upper']]).round(2)

        future_results_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_results_df['ds'] = pd.to_datetime(future_results_df['ds'])

        logger.info(f"Generated 7-day forecast for station {station_id} using model from {model_station}")
        print(future_results_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

        plt.figure(figsize=(12, 6))
        plt.scatter(test_df['ds'], test_df['y'], label='Actual TMIN (°C)', alpha=0.5, color='black')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted TMIN (°C)', color='blue')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                         color='blue', alpha=0.2, label='Confidence Interval')
        plt.title(f"TMIN Forecast for Station {station_id} using model from Station {model_station} (Test Data - Last 6 Months)")
        plt.xlabel("Date")
        plt.ylabel("TMIN (°C)")
        plt.legend()
        plt.tight_layout()
        plot_path = self.plot_dir / f'plots_{station_id}.png'
        plt.savefig(plot_path)
        logger.info(f"Saved forecast plot for station {station_id} to {plot_path}")
        plt.close()