import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, df):
        """Initialize with a dataframe."""
        self.df = df

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

    def add_temperature_features(self, station_df):
        """Add temperature-related features within each station."""
        station_df = station_df.sort_values(by='DATE')
        station_df['Temp_Diff'] = station_df['TMAX'] - station_df['TMIN']
        station_df['Rolling_Mean_TMIN_7'] = station_df['TMIN'].rolling(window=7, min_periods=1).mean()
        station_df['Rolling_10thPercentile_TMIN_7'] = station_df['TMIN'].rolling(window=7, min_periods=1).apply(lambda x: np.percentile(x, 10), raw=True)
        station_df['TMIN_Rolling_30_Diff'] = station_df['TMIN'] - station_df['Rolling_Mean_TMIN_7']
        station_df['EWMA_TMIN_7'] = station_df['TMIN'].ewm(span=7, adjust=False).mean()
        station_df['Seasonal_TMIN_Anomaly'] = station_df['TMIN'] - station_df.groupby([station_df['DATE'].dt.month, station_df['DATE'].dt.day])['TMIN'].transform('mean')
        station_df['Rolling_Max_TMIN_30'] = station_df['TMIN'].rolling(window=30, min_periods=1).max()
        station_df['Rolling_Min_TMIN_30'] = station_df['TMIN'].rolling(window=30, min_periods=1).min()
        station_df['EWMA_TMIN_30'] = station_df['TMIN'].ewm(span=30, adjust=False).mean()
        station_df['TMIN_Lag1'] = station_df['TMIN'].shift(1)
        return station_df

    def add_snow_features(self, station_df):
        """Add snow-related features within each station."""
        station_df = station_df.sort_values(by='DATE')
        station_df['SnowyDay'] = (station_df['SNOW'] > 0).astype(int)
        station_df['SnowyDaysCount_7'] = station_df['SNOW'].rolling(window=7, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
        station_df['Cumulative_SnowDepth_7'] = station_df['SNWD'].rolling(window=7, min_periods=1).sum()
        station_df['Cumulative_Snowfall_Lag7'] = station_df['SNOW'].shift(7).rolling(window=7, min_periods=1).sum()
        station_df['SNWD_Lag1'] = station_df['SNWD'].shift(1)
        station_df['SNWD_Lag2'] = station_df['SNWD'].shift(2)
        station_df['Rolling_Sum_SNWD_7'] = station_df['SNWD'].rolling(window=7, min_periods=1).sum()
        station_df['SNWD_TMIN_Interaction'] = station_df['SNWD'] * station_df['TMIN']
        station_df['Snowfall_Intensity'] = station_df['SNOW'] / (station_df['SNWD'] + 1)
        station_df['SNWD_Snowfall_Diff'] = station_df['SNWD'] - station_df['Snowfall_Intensity']
        return station_df

    def add_precipitation_features(self, station_df):
        """Add precipitation-related features within each station."""
        station_df = station_df.sort_values(by='DATE')
        station_df['PRCP_Lag1'] = station_df['PRCP'].shift(1)
        station_df['PRCP_Lag2'] = station_df['PRCP'].shift(2)
        station_df['Cumulative_Precipitation_7'] = station_df['PRCP'].rolling(window=7, min_periods=1).sum()
        station_df['Rolling_Sum_PRCP_14'] = station_df['PRCP'].rolling(window=14, min_periods=1).sum()
        station_df['TMAX_PRCP_Interaction'] = station_df['TMAX'] * station_df['PRCP']
        return station_df

    def add_additional_features(self, station_df):
        """Add additional station-specific features."""
        station_df['Rolling_Mean_TMIN_30'] = station_df['TMIN'].rolling(window=30, min_periods=1).mean()
        station_df['TMIN_SNOW_Interaction'] = station_df['TMIN'] * station_df['SNOW']
        return station_df
    
    def handle_nulls(self, station_df):
        """Handle null values appropriately for each station."""
        station_df = station_df.sort_values(by='DATE')

        # Fill forward for lag features, then fill remaining with station-wise mean
        lag_features = ['TMIN_Lag1', 'SNWD_Lag1', 'SNWD_Lag2', 'PRCP_Lag1', 'PRCP_Lag2', 'Cumulative_Snowfall_Lag7']
        station_df[lag_features] = station_df[lag_features].ffill().fillna(station_df[lag_features].mean())

        # Fill rolling and cumulative features with station-wise mean
        rolling_features = ['Rolling_Mean_TMIN_7', 'Rolling_10thPercentile_TMIN_7', 'Rolling_Mean_TMIN_30',
                            'Cumulative_SnowDepth_7', 'Cumulative_Precipitation_7', 'Rolling_Sum_PRCP_14']
        station_df[rolling_features] = station_df[rolling_features].fillna(station_df[rolling_features].mean())

        # Fill interaction terms with 0
        interaction_features = ['TMIN_SNOW_Interaction', 'TMAX_PRCP_Interaction', 'SNWD_TMIN_Interaction']
        station_df[interaction_features] = station_df[interaction_features].fillna(0)

        return station_df
    
    def apply_all_features(self):
        """Apply all feature engineering steps station-wise."""
        self.add_station_features()
        self.add_temporal_features()

        # Apply station-wise transformations
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_temperature_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_snow_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_precipitation_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.add_additional_features)
        self.df = self.df.groupby('Station_ID', group_keys=False).apply(self.handle_nulls)

        # Exclude LATITUDE and LONGITUDE from rounding
        cols_to_exclude = ['LATITUDE', 'LONGITUDE']
        cols_to_round = [col for col in self.df.columns if col not in cols_to_exclude]
        self.df[cols_to_round] = self.df[cols_to_round].round(2)

        return self.df
    
class ColdEventDetector:
    def __init__(self, df):
        """Initialize with a dataframe."""
        self.df = df

    def define_cold_event(self, station_df):
        """Identify cold events based on temperature thresholds and anomalies within each station."""
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

