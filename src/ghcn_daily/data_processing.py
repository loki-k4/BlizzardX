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
        self.df = df
        self.station_correlations = {}

    def preprocess_dates(self):
        """Preprocess the 'DATE' column and convert it to datetime."""
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])

    def compute_station_correlations(self):
        """Precompute correlations between TMIN/TMAX and PRCP."""
        for station_id, group in self.df.groupby('ID'):
            correlation_tmin_prcp = group['TMIN'].corr(group['PRCP'])
            correlation_tmax_prcp = group['TMAX'].corr(group['PRCP'])
            self.station_correlations[station_id] = {
                'correlation_tmin_prcp': correlation_tmin_prcp,
                'correlation_tmax_prcp': correlation_tmax_prcp
            }

    def fill_missing_temperatures(self, alpha=0.6):
        """Fill missing TMIN and TMAX using interpolation and adjust based on correlations."""
        self.df[['TMIN', 'TMAX']] = self.df[['TMIN', 'TMAX']].interpolate(method='linear')

        # Use ThreadPoolExecutor for parallel execution across stations
        with ThreadPoolExecutor() as executor:
            futures = []
            for station_id, group in self.df.groupby('ID'):
                futures.append(executor.submit(self._fill_missing_station_temperatures, group, station_id, alpha))

            for future in futures:
                future.result()  # Ensure all threads complete

    def _fill_missing_station_temperatures(self, group, station_id, alpha):
        """Helper method to fill missing temperatures for each station."""
        correlation_tmin_prcp = self.station_correlations[station_id].get('correlation_tmin_prcp', None)
        correlation_tmax_prcp = self.station_correlations[station_id].get('correlation_tmax_prcp', None)

        missing_tmin_mask = group['TMIN'].isnull()
        missing_tmax_mask = group['TMAX'].isnull()

        # Adjust missing TMIN values based on correlation if it exists
        if missing_tmin_mask.any():
            group.loc[missing_tmin_mask, 'TMIN'] = self._adjust_missing_value(
                group[missing_tmin_mask], 'TMIN', correlation_tmin_prcp, alpha)

        # Adjust missing TMAX values based on correlation if it exists
        if missing_tmax_mask.any():
            group.loc[missing_tmax_mask, 'TMAX'] = self._adjust_missing_value(
                group[missing_tmax_mask], 'TMAX', correlation_tmax_prcp, alpha)

        self.df.loc[group.index, ['TMIN', 'TMAX']] = group[['TMIN', 'TMAX']].round(2)

    def _adjust_missing_value(self, group, column, correlation, alpha):
        """Helper function to adjust missing values based on correlation or fallback to previous/next day."""
        if pd.notna(correlation):  # Use correlation if available
            prev_vals = group[column].shift(1)
            next_vals = group[column].shift(-1)
            adjustment = (next_vals - prev_vals) * correlation
            adjusted_values = group[column] + alpha * adjustment
            return adjusted_values.round(2)
        else:  # If no correlation, fallback to previous/next day values
            adjusted_values = group[column].fillna(method='ffill').fillna(method='bfill')
            return adjusted_values.round(2)

    def fill_missing_snow(self):
        """Handle missing or zero SNOW values."""
        snow_zero_mask = self.df['SNOW'].isnull() | (self.df['SNOW'] == 0)
        prcp_zero_mask = self.df['PRCP'] == 0
        
        # Fill missing snow values with the previous value if possible
        self.df.loc[snow_zero_mask & prcp_zero_mask, 'SNOW'] = 0
        self.df.loc[snow_zero_mask & ~prcp_zero_mask, 'SNOW'] = self.df['SNOW'].shift(1)

        # Fallback to next available snow value if still missing
        self.df['SNOW'] = self.df['SNOW'].bfill().round(2)

    def fill_missing_prcp(self):
        """Fill missing PRCP values based on correlation with TMIN/TMAX."""
        # Use ThreadPoolExecutor for parallel execution across stations
        with ThreadPoolExecutor() as executor:
            futures = []
            for station_id, group in self.df.groupby('ID'):
                futures.append(executor.submit(self._fill_missing_station_prcp, group, station_id))

            for future in futures:
                future.result()  # Ensure all threads complete

    def _fill_missing_station_prcp(self, group, station_id):
        """Helper method to fill missing PRCP values for each station."""
        correlation_tmin_prcp = self.station_correlations[station_id].get('correlation_tmin_prcp', None)
        correlation_tmax_prcp = self.station_correlations[station_id].get('correlation_tmax_prcp', None)
        
        missing_prcp_mask = group['PRCP'].isnull()
        
        if missing_prcp_mask.any():
            prev_prcp = group['PRCP'].shift(1)
            next_prcp = group['PRCP'].shift(-1)
            avg_prcp = pd.concat([prev_prcp, next_prcp], axis=1).mean(axis=1)

            # Use correlation to adjust the missing PRCP values if correlation exists
            if pd.notna(correlation_tmin_prcp) and pd.notna(correlation_tmax_prcp):
                adjusted_prcp = avg_prcp + (group['TMIN'] - group['TMIN'].shift(1)) * correlation_tmin_prcp
                adjusted_prcp += (group['TMAX'] - group['TMAX'].shift(1)) * correlation_tmax_prcp
            else:
                # If no correlation, fill with previous or next day values
                adjusted_prcp = group['PRCP'].fillna(method='ffill').fillna(method='bfill')

            group.loc[missing_prcp_mask, 'PRCP'] = adjusted_prcp.round(2)
        
        self.df.loc[group.index, 'PRCP'] = group['PRCP'].round(2)

    def fill_missing_snwd(self):
        """Handle missing SNWD values when SNOW is present or missing."""
        # Use ThreadPoolExecutor for parallel execution across stations
        with ThreadPoolExecutor() as executor:
            futures = []
            for station_id, group in self.df.groupby('ID'):
                futures.append(executor.submit(self._fill_missing_station_snwd, group, station_id))

            for future in futures:
                future.result()  # Ensure all threads complete

    def _fill_missing_station_snwd(self, group, station_id):
        """Helper method to fill missing SNWD values for each station."""
        correlation_snwd_snow = group['SNWD'].corr(group['SNOW']) if group['SNWD'].notna().any() else np.nan
        
        missing_snwd_mask = group['SNWD'].isnull() & (group['SNOW'] > 0)
        
        if missing_snwd_mask.any():
            # Adjust missing SNWD values based on the correlation with SNOW
            previous_snwd = group['SNWD'].shift(1)
            next_snwd = group['SNWD'].shift(-1)
            adjusted_snwd = previous_snwd + correlation_snwd_snow * (group['SNOW'] - group['SNOW'].shift(1))

            group.loc[missing_snwd_mask, 'SNWD'] = adjusted_snwd.round(2)

        # If no snow, set SNWD to 0
        group.loc[group['SNOW'] == 0, 'SNWD'] = 0

        self.df.loc[group.index, 'SNWD'] = group['SNWD'].round(2)

    def clean_all(self, alpha=0.6):
        """Run all the cleaning methods."""
        self.preprocess_dates()
        self.compute_station_correlations()  # Precompute correlations
        self.fill_missing_temperatures(alpha)
        self.fill_missing_snow()
        self.fill_missing_prcp()
        self.fill_missing_snwd()  # New method to handle missing SNWD
        return self.df

