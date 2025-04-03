import os
import pandas as pd
import numpy as np
import glob
from concurrent.futures import ThreadPoolExecutor

class WeatherDataProcessor:
    def __init__(self, data, weather_variables, variable_data_output_dir='/workspaces/BlizzardX/Data/Variable_Data', transform_data_output_dir='/workspaces/BlizzardX/Data/Transform_Data'):
        self.data = data
        self.weather_variables = weather_variables
        self.variable_data_output_dir = variable_data_output_dir
        self.transform_data_output_dir = transform_data_output_dir
        self.variable_dfs = {}

    def save_variable_dataframes(self):
        """
        Extracts data for each weather variable and saves them as separate CSV files.
        """
        if not os.path.exists(self.variable_data_output_dir):
            os.makedirs(self.variable_data_output_dir)
        
        for var in self.weather_variables:
            var_df = self.data[self.data['ELEMENT'] == var].copy()
            self.variable_dfs[var] = var_df
            var_df.to_csv(f'{self.variable_data_output_dir}/{var}_data.csv', index=False)
        
        return self.variable_dfs

    def convert_to_daily_data(self, df, element):
        """
        Converts monthly data to daily data for a specific weather variable (element).
        """
        transformed_data = []
        for _, row in df.iterrows():
            station_id = row['ID']
            year = row['YEAR']
            month = row['Month']
            for day in range(1, 32):
                date_str = f"{year}-{month:02d}-{day:02d}"
                value_column = f'VALUE{day}'
                element_value = row.get(value_column, None)
                if pd.notna(element_value):
                    transformed_data.append({
                        'DATE': date_str,
                        'ID': station_id,
                        element: element_value
                    })
        return pd.DataFrame(transformed_data)

    def transform_all_variables_to_daily(self):
        """
        Transforms all weather variables into daily data and saves them.
        """
        if not os.path.exists(self.transform_data_output_dir):
            os.makedirs(self.transform_data_output_dir)

        for var, df in self.variable_dfs.items():
            transformed_df = self.convert_to_daily_data(df, var)
            transformed_df.to_csv(f'{self.transform_data_output_dir}/{var}_data.csv', index=False)

    def merge_csv_files(self, directory_path):
        """
        Combines all CSV files in the specified directory based on the 'ID' and 'DATE' columns.
        """
        csv_files = glob.glob(f'{directory_path}/*_data.csv')
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the directory: {directory_path}")
        
        combined_df = pd.read_csv(csv_files[0])
        for file in csv_files[1:]:
            df = pd.read_csv(file)
            df = df.rename(columns=lambda x: x.split('_')[0] if '_' in x else x)
            combined_df = pd.merge(combined_df, df, on=['ID', 'DATE'], how='outer', suffixes=('_left', '_right'))
        
        return combined_df

    def add_season_column(self, df):
        """
        Adds a 'Season' column based on the 'DATE' column.
        """
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'

        df['Season'] = df['DATE'].dt.month.apply(get_season)
        return df

    def normalize_weather_columns(self, df):
        """
        Normalizes the weather columns by converting them to numeric and dividing by 10
        (except SNOW and SNWD which remain unchanged).
        """
        for col in self.weather_variables:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in self.weather_variables:
            if col == 'SNOW' or col == 'SNWD':
                df.rename(columns={col: f"{col}"}, inplace=True)
            else:
                df[col] = df[col] / 10
                df.rename(columns={col: f"{col}"}, inplace=True)

        return df
    
    def process_data(self):
        """
        Main method that runs the entire data processing pipeline.
        """
        self.save_variable_dataframes()
        self.transform_all_variables_to_daily()
        combined_df = self.merge_csv_files(self.transform_data_output_dir)
        combined_df = self.add_season_column(combined_df)
        final_df = self.normalize_weather_columns(combined_df)

        return final_df


class WeatherDataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.station_correlations = {}
        required_cols = ['DATE', 'ID', 'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD', 'Season']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain all required columns: {required_cols}")

    def preprocess_dates(self):
        """Convert DATE to datetime and extract Month and Day for historical means."""
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])
        self.df['Month'] = self.df['DATE'].dt.month
        self.df['Day'] = self.df['DATE'].dt.day

    def compute_station_correlations(self):
        """Precompute correlations between TMIN/TMAX and PRCP for each station."""
        for station_id, group in self.df.groupby('ID'):
            if group['TMIN'].notna().sum() > 1 and group['TMAX'].notna().sum() > 1:
                correlation_tmin_prcp = group['TMIN'].corr(group['PRCP'])
                correlation_tmax_prcp = group['TMAX'].corr(group['PRCP'])
                self.station_correlations[station_id] = {
                    'correlation_tmin_prcp': correlation_tmin_prcp,
                    'correlation_tmax_prcp': correlation_tmax_prcp
                }
            else:
                self.station_correlations[station_id] = {
                    'correlation_tmin_prcp': np.nan,
                    'correlation_tmax_prcp': np.nan
                }

    def fill_missing_temperatures(self, alpha=0.6):
        """Fill missing TMIN and TMAX station-wise."""
        original_tmin_missing = self.df['TMIN'].isnull()
        original_tmax_missing = self.df['TMAX'].isnull()
        
        # Step 1: Station-wise interpolation
        for station_id, group in self.df.groupby('ID'):
            self.df.loc[group.index, ['TMIN', 'TMAX']] = group[['TMIN', 'TMAX']].interpolate(method='linear')
        
        # Step 2: Station-wise same-date means
        for station_id, group in self.df.groupby('ID'):
            for col in ['TMIN', 'TMAX']:
                still_missing = self.df.loc[group.index, col].isnull()
                if still_missing.any():
                    date_means = group.groupby(['Month', 'Day'])[col].mean()
                    missing_indices = self.df.loc[group.index][still_missing].index
                    for idx in missing_indices:
                        month = self.df.at[idx, 'Month']
                        day = self.df.at[idx, 'Day']
                        if (month, day) in date_means.index:
                            self.df.at[idx, col] = date_means[(month, day)]
        
        # Step 3: Station-wise seasonal means
        for station_id, group in self.df.groupby('ID'):
            for col in ['TMIN', 'TMAX']:
                still_missing = self.df.loc[group.index, col].isnull()
                if still_missing.any():
                    seasonal_means = group.groupby('Season')[col].mean()
                    missing_indices = self.df.loc[group.index][still_missing].index
                    for idx in missing_indices:
                        season = self.df.at[idx, 'Season']
                        if season in seasonal_means.index:
                            self.df.at[idx, col] = seasonal_means[season]
        
        # Step 4: Station-specific adjustments using PRCP correlations
        with ThreadPoolExecutor() as executor:
            futures = []
            for station_id, group in self.df.groupby('ID'):
                futures.append(executor.submit(self._fill_missing_station_temperatures, group, station_id, alpha, 
                                               original_tmin_missing[group.index], original_tmax_missing[group.index]))
            for future in futures:
                future.result()

    def _fill_missing_station_temperatures(self, group, station_id, alpha, tmin_missing, tmax_missing):
        """Adjust missing TMIN/TMAX based on PRCP correlations."""
        correlation_tmin_prcp = self.station_correlations[station_id].get('correlation_tmin_prcp', None)
        correlation_tmax_prcp = self.station_correlations[station_id].get('correlation_tmax_prcp', None)

        if tmin_missing.any() and not group['PRCP'].isnull().all():
            group.loc[tmin_missing, 'TMIN'] = self._adjust_missing_value(group[tmin_missing], 'TMIN', correlation_tmin_prcp, alpha)
        if tmax_missing.any() and not group['PRCP'].isnull().all():
            group.loc[tmax_missing, 'TMAX'] = self._adjust_missing_value(group[tmax_missing], 'TMAX', correlation_tmax_prcp, alpha)
        
        self.df.loc[group.index, ['TMIN', 'TMAX']] = group[['TMIN', 'TMAX']].round(2)

    def _adjust_missing_value(self, group, column, correlation, alpha):
        """Helper to adjust values based on correlation or fallback to fill."""
        if pd.notna(correlation):
            prev_vals = group[column].shift(1)
            next_vals = group[column].shift(-1)
            adjustment = (next_vals - prev_vals) * correlation
            return (group[column] + alpha * adjustment).round(2)
        return group[column].ffill().bfill().round(2)

    def fill_missing_prcp(self):
        """Fill missing PRCP station-wise (not needed for current data, but included for completeness)."""
        # Step 1: Station-wise interpolation
        for station_id, group in self.df.groupby('ID'):
            self.df.loc[group.index, 'PRCP'] = group['PRCP'].interpolate(method='linear')
        
        # Step 2: Station-wise same-date means
        for station_id, group in self.df.groupby('ID'):
            still_missing = self.df.loc[group.index, 'PRCP'].isnull()
            if still_missing.any():
                date_means = group.groupby(['Month', 'Day'])['PRCP'].mean()
                missing_indices = self.df.loc[group.index][still_missing].index
                for idx in missing_indices:
                    month = self.df.at[idx, 'Month']
                    day = self.df.at[idx, 'Day']
                    if (month, day) in date_means.index:
                        self.df.at[idx, 'PRCP'] = date_means[(month, day)]
        
        # Step 3: Station-wise seasonal means
        for station_id, group in self.df.groupby('ID'):
            still_missing = self.df.loc[group.index, 'PRCP'].isnull()
            if still_missing.any():
                seasonal_means = group.groupby('Season')['PRCP'].mean()
                missing_indices = self.df.loc[group.index][still_missing].index
                for idx in missing_indices:
                    season = self.df.at[idx, 'Season']
                    if season in seasonal_means.index:
                        self.df.at[idx, 'PRCP'] = seasonal_means[season]
        
        # Step 4: Station-specific adjustments
        with ThreadPoolExecutor() as executor:
            futures = []
            for station_id, group in self.df.groupby('ID'):
                futures.append(executor.submit(self._fill_missing_station_prcp, group, station_id))
            for future in futures:
                future.result()
        
        self.df['PRCP'] = self.df['PRCP'].clip(lower=0).round(2)

    def _fill_missing_station_prcp(self, group, station_id):
        """Adjust PRCP based on TMIN/TMAX correlations."""
        correlation_tmin_prcp = self.station_correlations[station_id].get('correlation_tmin_prcp', None)
        correlation_tmax_prcp = self.station_correlations[station_id].get('correlation_tmax_prcp', None)
        missing_prcp_mask = group['PRCP'].isnull()
        
        if missing_prcp_mask.any():
            prev_prcp = group['PRCP'].shift(1)
            next_prcp = group['PRCP'].shift(-1)
            avg_prcp = pd.concat([prev_prcp, next_prcp], axis=1).mean(axis=1)
            if pd.notna(correlation_tmin_prcp) and pd.notna(correlation_tmax_prcp):
                adjusted_prcp = avg_prcp + (group['TMIN'] - group['TMIN'].shift(1)) * correlation_tmin_prcp
                adjusted_prcp += (group['TMAX'] - group['TMAX'].shift(1)) * correlation_tmax_prcp
                group.loc[missing_prcp_mask, 'PRCP'] = adjusted_prcp.round(2)
        
        self.df.loc[group.index, 'PRCP'] = group['PRCP'].round(2)

    def fill_missing_snow(self):
        """Fill missing SNOW station-wise (not needed for current data, but included)."""
        # Step 1: Station-wise interpolation
        for station_id, group in self.df.groupby('ID'):
            self.df.loc[group.index, 'SNOW'] = group['SNOW'].interpolate(method='linear')
        
        # Step 2: Station-wise same-date means
        for station_id, group in self.df.groupby('ID'):
            still_missing = self.df.loc[group.index, 'SNOW'].isnull()
            if still_missing.any():
                date_means = group.groupby(['Month', 'Day'])['SNOW'].mean()
                missing_indices = self.df.loc[group.index][still_missing].index
                for idx in missing_indices:
                    month = self.df.at[idx, 'Month']
                    day = self.df.at[idx, 'Day']
                    if (month, day) in date_means.index:
                        self.df.at[idx, 'SNOW'] = date_means[(month, day)]
        
        # Step 3: Station-wise seasonal logic
        for station_id, group in self.df.groupby('ID'):
            still_missing = self.df.loc[group.index, 'SNOW'].isnull()
            if still_missing.any():
                seasonal_means = group.groupby('Season')['SNOW'].mean()
                missing_indices = self.df.loc[group.index][still_missing].index
                for idx in missing_indices:
                    season = self.df.at[idx, 'Season']
                    prcp = self.df.at[idx, 'PRCP']
                    if prcp == 0:
                        self.df.at[idx, 'SNOW'] = 0
                    elif season == 'Winter' and season in seasonal_means.index:
                        self.df.at[idx, 'SNOW'] = seasonal_means[season]
                    else:
                        self.df.at[idx, 'SNOW'] = 0
        
        self.df['SNOW'] = self.df['SNOW'].clip(lower=0).round(2)

    def fill_missing_snwd(self):
        """Fill missing SNWD station-wise (not needed for current data, but included)."""
        # Step 1: Station-wise interpolation
        for station_id, group in self.df.groupby('ID'):
            self.df.loc[group.index, 'SNWD'] = group['SNWD'].interpolate(method='linear')
        
        # Step 2: Station-wise same-date means
        for station_id, group in self.df.groupby('ID'):
            still_missing = self.df.loc[group.index, 'SNWD'].isnull()
            if still_missing.any():
                date_means = group.groupby(['Month', 'Day'])['SNWD'].mean()
                missing_indices = self.df.loc[group.index][still_missing].index
                for idx in missing_indices:
                    month = self.df.at[idx, 'Month']
                    day = self.df.at[idx, 'Day']
                    if (month, day) in date_means.index:
                        self.df.at[idx, 'SNWD'] = date_means[(month, day)]
        
        # Step 3: Station-specific SNWD adjustment
        with ThreadPoolExecutor() as executor:
            futures = []
            for station_id, group in self.df.groupby('ID'):
                futures.append(executor.submit(self._fill_missing_station_snwd, group, station_id))
            for future in futures:
                future.result()

    def _fill_missing_station_snwd(self, group, station_id):
        """Adjust SNWD based on SNOW correlation or accumulation."""
        correlation_snwd_snow = group['SNWD'].corr(group['SNOW']) if group['SNWD'].notna().any() else np.nan
        missing_snwd_mask = group['SNWD'].isnull()
        
        if missing_snwd_mask.any():
            previous_snwd = group['SNWD'].shift(1).fillna(0)
            if pd.notna(correlation_snwd_snow):
                adjusted_snwd = previous_snwd + correlation_snwd_snow * (group['SNOW'] - group['SNOW'].shift(1).fillna(0))
            else:
                adjusted_snwd = previous_snwd + group['SNOW']
            group.loc[missing_snwd_mask, 'SNWD'] = adjusted_snwd.clip(lower=0).round(2)

        group.loc[group['SNOW'] == 0, 'SNWD'] = 0
        self.df.loc[group.index, 'SNWD'] = group['SNWD'].clip(lower=0).round(2)

    def clean_all(self, alpha=0.6):
        """Run all cleaning methods."""
        self.preprocess_dates()
        self.compute_station_correlations()
        self.fill_missing_temperatures(alpha)
        self.fill_missing_prcp()
        self.fill_missing_snow()
        self.fill_missing_snwd()
        self.df = self.df.drop(columns=['Month', 'Day'])
        return self.df