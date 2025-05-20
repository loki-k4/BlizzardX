import pandas as pd
import logging
from datetime import datetime

# Configure logging to console only
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class WeatherDataFilter:
    
    def __init__(self, df, date_col='DATE', id_col='ID'):
        
        self.logger = setup_logging()
        self.logger.info("Initializing WeatherDataFilter")
        
        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input 'df' must be a pandas DataFrame")
            raise ValueError("Input 'df' must be a pandas DataFrame")
        if df.empty:
            self.logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame is empty")
        
        # Validate column names
        if date_col not in df.columns:
            self.logger.error(f"Date column '{date_col}' not found in DataFrame")
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        if id_col not in df.columns:
            self.logger.error(f"ID column '{id_col}' not found in DataFrame")
            raise ValueError(f"ID column '{id_col}' not found in DataFrame")
        
        self.df = df.copy()
        self.date_col = date_col
        self.id_col = id_col
        
        try:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
            self.df = self.df.dropna(subset=[self.date_col])
            self.logger.info(f"Converted '{self.date_col}' to datetime and dropped {len(df) - len(self.df)} rows with invalid dates")
            if self.df.empty:
                self.logger.error("DataFrame is empty after dropping invalid dates")
                raise ValueError("DataFrame is empty after dropping invalid dates")
        except Exception as e:
            self.logger.error(f"Error processing date column '{self.date_col}': {e}")
            raise

    def get_stations_with_zero_missing_dates(self):
       
        self.logger.info("Checking for stations with zero missing dates")
        complete_stations = []
        
        try:
            for station_id, group in self.df.groupby(self.id_col):
                min_date = group[self.date_col].min()
                max_date = group[self.date_col].max()
                if pd.isna(min_date) or pd.isna(max_date):
                    self.logger.warning(f"Station {station_id} has invalid date range, skipping")
                    continue
                
                expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
                actual_dates = group[self.date_col].dt.normalize().unique()
                if len(expected_dates) == len(actual_dates):
                    complete_stations.append(station_id)
                    self.logger.info(f"Station {station_id} has no missing dates")
                else:
                    self.logger.debug(f"Station {station_id} has {len(expected_dates) - len(actual_dates)} missing dates")
        except Exception as e:
            self.logger.error(f"Error checking missing dates: {e}")
            raise
        
        self.logger.info(f"Found {len(complete_stations)} stations with zero missing dates")
        return complete_stations

    def get_stations_with_low_missing_values(self, threshold=5, columns=None):
        
        self.logger.info(f"Checking for stations with missing values below {threshold}%")
        
        if columns is None:
            columns = ['TMIN', 'TMAX', 'SNOW', 'SNWD', 'PRCP']
        
        # Filter columns that exist in DataFrame
        columns = [col for col in columns if col in self.df.columns]
        if not columns:
            self.logger.error("No valid weather columns found in DataFrame")
            raise ValueError("No valid weather columns found in DataFrame")
        
        self.logger.info(f"Checking columns: {columns}")
        stations = self.df[self.id_col].unique()
        stations_with_low_missing = []
        
        try:
            for station in stations:
                station_data = self.df[self.df[self.id_col] == station].copy()
                station_data = station_data.sort_values(self.date_col)
                missing_percentage = station_data[columns].isnull().mean() * 100
                if (missing_percentage < threshold).all():
                    stations_with_low_missing.append(station)
                    self.logger.info(f"Station {station} has missing values below {threshold}%: {missing_percentage.to_dict()}")
                else:
                    self.logger.debug(f"Station {station} exceeds missing value threshold: {missing_percentage.to_dict()}")
        except Exception as e:
            self.logger.error(f"Error checking missing values: {e}")
            raise
        
        self.logger.info(f"Found {len(stations_with_low_missing)} stations with missing values below {threshold}%")
        return stations_with_low_missing

    def get_station_missing_stats(self, station_id):
        
        self.logger.info(f"Generating missing data statistics for station {station_id}")
        
        station_data = self.df[self.df[self.id_col] == station_id].copy()
        if station_data.empty:
            self.logger.warning(f"No data found for station {station_id}")
            return None
        
        try:
            min_date = station_data[self.date_col].min()
            max_date = station_data[self.date_col].max()
            if pd.isna(min_date) or pd.isna(max_date):
                self.logger.error(f"Invalid date range for station {station_id}")
                return None
            
            expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            actual_dates = station_data[self.date_col].dt.normalize().unique()
            missing_dates = len(expected_dates) - len(actual_dates)
            
            weather_columns = ['TMIN', 'TMAX', 'SNOW', 'SNWD', 'PRCP']
            columns_present = [col for col in weather_columns if col in station_data.columns]
            missing_percentages = station_data[columns_present].isnull().mean() * 100
            
            stats = {
                'station_id': station_id,
                'missing_dates': missing_dates,
                'total_expected_dates': len(expected_dates),
                'missing_value_percentages': missing_percentages.to_dict()
            }
            self.logger.info(f"Statistics for station {station_id}: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error generating stats for station {station_id}: {e}")
            return None