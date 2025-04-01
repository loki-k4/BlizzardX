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
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        """Add temporal features based on the DATE column."""
        self.df['Day_of_Week'] = self.df['DATE'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['Day_of_Year'] = self.df['DATE'].dt.dayofyear  # 1=Jan 1st, 365=Dec 31st
        

    def add_temperature_features(self):
        """Add features related to temperature (TMIN and TMAX)."""
        self.df['Temp_Diff'] = self.df['TMAX'] - self.df['TMIN']  # Difference between TMAX and TMIN
        
        # Rolling mean and percentile of TMIN
        self.df['Rolling_Mean_TMIN_7'] = self.df['TMIN'].rolling(window=7).mean()  # 7-day rolling mean of TMIN
        self.df['Rolling_10thPercentile_TMIN_7'] = self.df['TMIN'].rolling(window=7).apply(lambda x: np.percentile(x, 10), raw=True)  # 7-day rolling 10th percentile of TMIN
        
        # Other rolling statistics
        self.df['TMIN_Rolling_30_Diff'] = self.df['TMIN'] - self.df['Rolling_Mean_TMIN_7']  # Difference between TMIN and 7-day rolling mean
        self.df['EWMA_TMIN_7'] = self.df['TMIN'].ewm(span=7, adjust=False).mean()  # Exponentially Weighted Moving Average of TMIN (7 days)
        
        # Previous season TMIN and seasonal anomaly in TMIN
        self.df['Seasonal_TMIN_Anomaly'] = self.df['TMIN'] - self.df.groupby([self.df['DATE'].dt.month, self.df['DATE'].dt.day])['TMIN'].transform('mean')  # Seasonal anomaly in TMIN

        
        # Rolling max/min and EWMA for TMIN
        self.df['Rolling_Max_TMIN_30'] = self.df['TMIN'].rolling(window=30).max()  # 30-day rolling max of TMIN
        self.df['Rolling_Min_TMIN_30'] = self.df['TMIN'].rolling(window=30).min()  # 30-day rolling min of TMIN
        self.df['EWMA_TMIN_30'] = self.df['TMIN'].ewm(span=30, adjust=False).mean()  # Exponentially Weighted Moving Average of TMIN (30 days)
        
        # Lag feature for TMIN
        self.df['TMIN_Lag1'] = self.df['TMIN'].shift(1)  # 1-day lag feature for TMIN

    def add_snow_features(self):
        """Add features related to snow (SNOW and SNWD)."""
        self.df['SnowyDay'] = (self.df['SNOW'] > 0).astype(int)  # Flag indicating if snow was observed
        self.df['SnowyDaysCount_7'] = self.df['SNOW'].rolling(window=7).apply(lambda x: (x > 0).sum(), raw=True)  # Count of snowy days in the last 7 days
        self.df['Cumulative_SnowDepth_7'] = self.df['SNWD'].rolling(window=7).sum()  # Cumulative snow depth over the last 7 days
        self.df['Cumulative_Snowfall_Lag7'] = self.df['SNOW'].shift(7).rolling(window=7).sum()  # Cumulative snowfall in the last 7 days
        self.df['SNWD_Lag1'] = self.df['SNWD'].shift(1)  # 1-day lag feature for snow depth
        self.df['SNWD_Lag2'] = self.df['SNWD'].shift(2)  # 2-day lag feature for snow depth
        self.df['Rolling_Sum_SNWD_7'] = self.df['SNWD'].rolling(window=7).sum()  # 7-day rolling sum of snow depth
        self.df['SNWD_TMIN_Interaction'] = self.df['SNWD'] * self.df['TMIN']  # Interaction term between snow depth and TMIN
        self.df['Snowfall_Intensity'] = self.df['SNOW'] / (self.df['SNWD'] + 1)  # Snowfall intensity (to avoid division by zero)
        self.df['SNWD_Snowfall_Diff'] = self.df['SNWD'] - self.df['Snowfall_Intensity']  # Difference between snow depth and snowfall intensity

    def add_precipitation_features(self):
        """Add features related to precipitation (PRCP)."""
        self.df['PRCP_Lag1'] = self.df['PRCP'].shift(1)  # 1-day lag feature for precipitation
        self.df['PRCP_Lag2'] = self.df['PRCP'].shift(2)  # 2-day lag feature for precipitation
        self.df['Cumulative_Precipitation_7'] = self.df['PRCP'].rolling(window=7).sum()  # Cumulative precipitation over the last 7 days
        self.df['Rolling_Sum_PRCP_14'] = self.df['PRCP'].rolling(window=14).sum()  # 14-day rolling sum of precipitation
        self.df['TMAX_PRCP_Interaction'] = self.df['TMAX'] * self.df['PRCP']  # Interaction term between maximum temperature and precipitation
    def add_additional_features(self):
        """Add additional features such as interaction between TMIN and SNOW."""
        self.df['Rolling_Mean_TMIN_30'] = self.df['TMIN'].rolling(window=30).mean()  # 30-day rolling mean of TMIN
        self.df['TMIN_SNOW_Interaction'] = self.df['TMIN'] * self.df['SNOW']  # Interaction term between TMIN and SNOW

    def apply_all_features(self):
        """Apply all feature engineering steps."""
        self.add_station_features()
        self.add_temporal_features()
        self.add_temperature_features()
        self.add_snow_features()
        self.add_precipitation_features()
        self.add_additional_features()
        
        # Round all columns with decimals to 2 places
        self.df = self.df.round(2)

        return self.df
class MissingValueImputation:
    def __init__(self, df):
        """Initialize with a dataframe."""
        self.df = df

    def fill_missing_values(self):
        """Fill missing values using appropriate strategies."""
        
        # Time Series Features (e.g., Rolling and EWMA)
        self.df['Rolling_Mean_TMIN_7'] = self.df['Rolling_Mean_TMIN_7'].fillna(method='ffill')  # Forward fill
        self.df['Rolling_10thPercentile_TMIN_7'] = self.df['Rolling_10thPercentile_TMIN_7'].fillna(method='ffill')
        self.df['TMIN_Rolling_30_Diff'] = self.df['TMIN_Rolling_30_Diff'].fillna(method='ffill')
        self.df['EWMA_TMIN_7'] = self.df['EWMA_TMIN_7'].fillna(method='ffill')
        self.df['Rolling_Max_TMIN_30'] = self.df['Rolling_Max_TMIN_30'].fillna(method='ffill')
        self.df['Rolling_Min_TMIN_30'] = self.df['Rolling_Min_TMIN_30'].fillna(method='ffill')
        self.df['EWMA_TMIN_30'] = self.df['EWMA_TMIN_30'].fillna(method='ffill')

        # Lag Features (e.g., TMIN_Lag1)
        self.df['TMIN_Lag1'] = self.df['TMIN_Lag1'].fillna(method='bfill')  # Backward fill

        # Count Features (e.g., SnowyDays, Cumulative Snowfall)
        self.df['SnowyDaysCount_7'] = self.df['SnowyDaysCount_7'].fillna(0)  # Fill with 0
        self.df['Cumulative_SnowDepth_7'] = self.df['Cumulative_SnowDepth_7'].fillna(0)  # Fill with 0
        self.df['Cumulative_Snowfall_Lag7'] = self.df['Cumulative_Snowfall_Lag7'].fillna(0)  # Fill with 0
        self.df['SNWD_Lag1'] = self.df['SNWD_Lag1'].fillna(0)  # Fill with 0
        self.df['SNWD_Lag2'] = self.df['SNWD_Lag2'].fillna(0)  # Fill with 0
        self.df['Rolling_Sum_SNWD_7'] = self.df['Rolling_Sum_SNWD_7'].fillna(0)  # Fill with 0

        # Continuous Features (e.g., SNWD, TMIN) using Mean Imputation
        self.df['SNWD'] = self.df['SNWD'].fillna(self.df['SNWD'].mean())  # Fill with mean
        self.df['TMIN'] = self.df['TMIN'].fillna(self.df['TMIN'].mean())  # Fill with mean
        self.df['TMAX'] = self.df['TMAX'].fillna(self.df['TMAX'].mean())  # Fill with mean
        self.df['PRCP'] = self.df['PRCP'].fillna(self.df['PRCP'].mean())  # Fill with mean

        # Return the modified dataframe
        return self.df