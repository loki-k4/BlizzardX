import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, df):
        """Initialize with a dataframe."""
        self.df = df.copy()  # Work on a copy to avoid modifying the original

    def add_station_features(self):
        """Add station-specific features based on latitude and longitude."""
        self.df['Station_Location'] = self.df['LATITUDE'].astype(str) + '_' + self.df['LONGITUDE'].astype(str)
        self.df['Station_Lat_Long_Interaction'] = self.df['LATITUDE'] * self.df['LONGITUDE']
        self.df.rename(columns={'ID': 'Station_ID'}, inplace=True)

    def add_temporal_features(self):
        """Add temporal features based on the DATE column."""
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df['Day_of_Week'] = self.df['DATE'].dt.dayofweek
        self.df['Day_of_Year'] = self.df['DATE'].dt.dayofyear
        self.df['Month'] = self.df['DATE'].dt.month  # Added for seasonal grouping

    def add_temperature_features(self, station_df):
        """Add temperature-related features within each station with imputation."""
        station_df = station_df.sort_values(by='DATE')
        
        # Basic temperature difference
        station_df['Temp_Diff'] = station_df['TMAX'] - station_df['TMIN']
        
        # Rolling features with imputation
        station_df['Rolling_Mean_TMIN_7'] = station_df['TMIN'].rolling(window=7, min_periods=1).mean()
        station_df['Rolling_10thPercentile_TMIN_7'] = station_df['TMIN'].rolling(window=7, min_periods=1).apply(lambda x: np.percentile(x, 10), raw=True)
        station_df['Rolling_Mean_TMIN_30'] = station_df['TMIN'].rolling(window=30, min_periods=1).mean()
        station_df['Rolling_Max_TMIN_30'] = station_df['TMIN'].rolling(window=30, min_periods=1).max()
        station_df['Rolling_Min_TMIN_30'] = station_df['TMIN'].rolling(window=30, min_periods=1).min()
        
        # Difference from rolling mean
        station_df['TMIN_Rolling_30_Diff'] = station_df['TMIN'] - station_df['Rolling_Mean_TMIN_30']
        
        # Exponentially Weighted Moving Average
        station_df['EWMA_TMIN_7'] = station_df['TMIN'].ewm(span=7, adjust=False).mean()
        station_df['EWMA_TMIN_30'] = station_df['TMIN'].ewm(span=30, adjust=False).mean()
        
        # Seasonal anomaly (grouped by month and day)
        station_df['Seasonal_TMIN_Anomaly'] = station_df['TMIN'] - station_df.groupby(['Month', station_df['DATE'].dt.day])['TMIN'].transform('mean')
        
        # Lag feature
        station_df['TMIN_Lag1'] = station_df['TMIN'].shift(1)
        
        return station_df

    def add_snow_features(self, station_df):
        """Add snow-related features within each station with imputation."""
        station_df = station_df.sort_values(by='DATE')
        
        # Binary snowy day indicator
        station_df['SnowyDay'] = (station_df['SNOW'] > 0).astype(int)
        
        # Rolling snow features
        station_df['SnowyDaysCount_7'] = station_df['SNOW'].rolling(window=7, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
        station_df['Cumulative_SnowDepth_7'] = station_df['SNWD'].rolling(window=7, min_periods=1).sum()
        station_df['Rolling_Sum_SNWD_7'] = station_df['SNWD'].rolling(window=7, min_periods=1).sum()
        
        # Lagged snow features
        station_df['SNWD_Lag1'] = station_df['SNWD'].shift(1)
        station_df['SNWD_Lag2'] = station_df['SNWD'].shift(2)
        station_df['Cumulative_Snowfall_Lag7'] = station_df['SNOW'].shift(7).rolling(window=7, min_periods=1).sum()
        
        # Interaction and derived features
        station_df['SNWD_TMIN_Interaction'] = station_df['SNWD'] * station_df['TMIN']
        station_df['Snowfall_Intensity'] = station_df['SNOW'] / (station_df['SNWD'] + 1)  # Avoid division by zero
        station_df['SNWD_Snowfall_Diff'] = station_df['SNWD'] - station_df['Snowfall_Intensity']
        
        return station_df

    def add_precipitation_features(self, station_df):
        """Add precipitation-related features within each station with imputation."""
        station_df = station_df.sort_values(by='DATE')
        
        # Lagged precipitation
        station_df['PRCP_Lag1'] = station_df['PRCP'].shift(1)
        station_df['PRCP_Lag2'] = station_df['PRCP'].shift(2)
        
        # Rolling precipitation features
        station_df['Cumulative_Precipitation_7'] = station_df['PRCP'].rolling(window=7, min_periods=1).sum()
        station_df['Rolling_Sum_PRCP_14'] = station_df['PRCP'].rolling(window=14, min_periods=1).sum()
        
        # Interaction feature
        station_df['TMAX_PRCP_Interaction'] = station_df['TMAX'] * station_df['PRCP']
        
        return station_df

    def add_additional_features(self, station_df):
        """Add additional station-specific features."""
        station_df['TMIN_SNOW_Interaction'] = station_df['TMIN'] * station_df['SNOW']
        station_df['PRCP_SNOW_Interaction'] = station_df['PRCP'] * station_df['SNOW']  # Added for precipitation-snow relationship
        
        return station_df

    def impute_rolling_values(self, station_df):
        """Impute missing rolling and lag values station-wise without altering original data distribution."""
        station_df = station_df.sort_values(by='DATE')
        
        # Define feature groups
        lag_features = ['TMIN_Lag1', 'SNWD_Lag1', 'SNWD_Lag2', 'PRCP_Lag1', 'PRCP_Lag2', 'Cumulative_Snowfall_Lag7']
        rolling_features = [
            'Rolling_Mean_TMIN_7', 'Rolling_10thPercentile_TMIN_7', 'Rolling_Mean_TMIN_30', 
            'Rolling_Max_TMIN_30', 'Rolling_Min_TMIN_30', 'SnowyDaysCount_7', 
            'Cumulative_SnowDepth_7', 'Rolling_Sum_SNWD_7', 'Cumulative_Precipitation_7', 
            'Rolling_Sum_PRCP_14'
        ]
        interaction_features = [
            'SNWD_TMIN_Interaction', 'TMAX_PRCP_Interaction', 'TMIN_SNOW_Interaction', 
            'PRCP_SNOW_Interaction'
        ]
        
        # Impute lag features: Forward fill, then use station median for remaining NaNs
        for feature in lag_features:
            station_df[feature] = station_df[feature].ffill().fillna(station_df[feature].median())
        
        # Impute rolling features: Use station median (preserves distribution)
        for feature in rolling_features:
            station_df[feature] = station_df[feature].fillna(station_df[feature].median())
        
        # Impute interaction features: Fill with 0 (logical for missing interactions)
        for feature in interaction_features:
            station_df[feature] = station_df[feature].fillna(0)
        
        # Impute anomaly and difference features: Use station median
        station_df['Seasonal_TMIN_Anomaly'] = station_df['Seasonal_TMIN_Anomaly'].fillna(station_df['Seasonal_TMIN_Anomaly'].median())
        station_df['TMIN_Rolling_30_Diff'] = station_df['TMIN_Rolling_30_Diff'].fillna(station_df['TMIN_Rolling_30_Diff'].median())
        
        return station_df

    def apply_all_features(self):
        """Apply all feature engineering steps station-wise."""
        # Apply initial transformations
        self.add_station_features()
        self.add_temporal_features()

        # Apply station-wise transformations
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_temperature_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_snow_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_precipitation_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_additional_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.impute_rolling_values)

        # Round numeric columns (excluding LATITUDE and LONGITUDE)
        cols_to_exclude = ['LATITUDE', 'LONGITUDE', 'DATE', 'Station_ID', 'Station_Location', 'NAME', 'Season']
        cols_to_round = [col for col in self.df.columns if col not in cols_to_exclude and self.df[col].dtype in ['float64', 'int64']]
        self.df[cols_to_round] = self.df[cols_to_round].round(2)

        return self.df
class ColdEventDetector:
    def __init__(self, df):
        """Initialize with a dataframe."""
        self.df = df.copy()  # Use a copy to avoid modifying the input

    def define_cold_event(self, station_df):
        """Identify cold events based on temperature thresholds and anomalies within each station."""
        # Ensure required columns are present and imputed
        required_cols = ['TMIN', 'Rolling_10thPercentile_TMIN_7', 'TMIN_Rolling_30_Diff', 'Seasonal_TMIN_Anomaly']
        for col in required_cols:
            if col not in station_df.columns:
                raise ValueError(f"Missing required column: {col}")
            station_df[col] = station_df[col].fillna(station_df[col].median())  # Fallback imputation

        # Define binary Cold_Event
        station_df['Cold_Event'] = (
            (station_df['TMIN'] < station_df['Rolling_10thPercentile_TMIN_7']) |  
            (station_df['TMIN_Rolling_30_Diff'] < station_df['TMIN_Rolling_30_Diff'].quantile(0.1)) |  
            (station_df['Seasonal_TMIN_Anomaly'] < station_df['Seasonal_TMIN_Anomaly'].quantile(0.1))
        ).astype(int)  
        return station_df

    def apply_cold_event_detection(self):
        """Apply cold event detection station-wise."""
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.define_cold_event)
        return self.df