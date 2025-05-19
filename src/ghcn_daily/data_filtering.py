# weather_data_filter.py
import pandas as pd

class WeatherDataFilter:
    """
    A class to filter and analyze weather station data for missing dates and values.
    
    Attributes:
        df (pandas.DataFrame): Input DataFrame containing weather data
        date_col (str): Name of the date column
        id_col (str): Name of the station ID column
    """
    
    def __init__(self, df, date_col='DATE', id_col='ID'):
        """
        Initialize the WeatherDataFilter with a DataFrame and column names.
        
        Parameters:
            df (pandas.DataFrame): DataFrame with weather data
            date_col (str): Name of the date column (default: 'DATE')
            id_col (str): Name of the station ID column (default: 'ID')
        """
        self.df = df.copy()
        self.date_col = date_col
        self.id_col = id_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        self.df = self.df.dropna(subset=[self.date_col])

    def get_stations_with_zero_missing_dates(self):
        """
        Returns a list of station IDs with zero missing dates.
        
        Returns:
            list: Station IDs with complete date sequences
        """
        complete_stations = []
        for station_id, group in self.df.groupby(self.id_col):
            min_date = group[self.date_col].min()
            max_date = group[self.date_col].max()
            expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            actual_dates = group[self.date_col].unique()
            if len(expected_dates) == len(actual_dates):
                complete_stations.append(station_id)
        return complete_stations

    def get_stations_with_low_missing_values(self, threshold=5, columns=None):
        """
        Returns a list of station IDs with less than specified percentage of missing values.
        
        Parameters:
            threshold (float): Maximum percentage of missing values allowed (default: 5)
            columns (list): List of columns to check for missing values
                          (default: ['TMIN', 'TMAX', 'SNOW', 'SNWD', 'PRCP'])
        
        Returns:
            list: Station IDs with missing values below threshold
        """
        if columns is None:
            columns = ['TMIN', 'TMAX', 'SNOW', 'SNWD', 'PRCP']
        
        columns = [col for col in columns if col in self.df.columns]
        
        stations = self.df[self.id_col].unique()
        stations_with_low_missing = []
        
        for station in stations:
            station_data = self.df[self.df[self.id_col] == station].copy()
            station_data = station_data.sort_values(self.date_col)
            missing_percentage = station_data[columns].isnull().mean() * 100
            if (missing_percentage < threshold).all():
                stations_with_low_missing.append(station)
                
        return stations_with_low_missing

    def get_station_missing_stats(self, station_id):
        """
        Returns missing data statistics for a specific station.
        
        Parameters:
            station_id: ID of the station to analyze
        
        Returns:
            dict: Statistics including missing dates and values percentage
        """
        station_data = self.df[self.df[self.id_col] == station_id].copy()
        if station_data.empty:
            return None
            
        min_date = station_data[self.date_col].min()
        max_date = station_data[self.date_col].max()
        expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        actual_dates = station_data[self.date_col].unique()
        missing_dates = len(expected_dates) - len(actual_dates)
        
        weather_columns = ['TMIN', 'TMAX', 'SNOW', 'SNWD', 'PRCP']
        columns_present = [col for col in weather_columns if col in station_data.columns]
        missing_percentages = station_data[columns_present].isnull().mean() * 100
        
        return {
            'station_id': station_id,
            'missing_dates': missing_dates,
            'total_expected_dates': len(expected_dates),
            'missing_value_percentages': missing_percentages.to_dict()
        }
