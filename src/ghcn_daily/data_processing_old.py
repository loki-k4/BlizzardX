import os
import pandas as pd
import numpy as np
import glob
import calendar
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor

class WeatherDataProcessorPandas:
    def __init__(self, data, weather_variables, max_workers=4):
        self.data = data
        self.weather_variables = weather_variables
        self.variable_dfs = {}
        self.transformed_dfs = {}
        self.max_workers = max_workers

    def extract_variable_dataframes(self):
        """
        Extracts data for each weather variable into DataFrames without saving to CSV.
        """
        def process_variable(var):
            var_df = self.data[self.data['ELEMENT'] == var].copy()
            return var, var_df

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(process_variable, self.weather_variables)
        
        self.variable_dfs = dict(results)
        return self.variable_dfs

    def convert_to_daily_data(self, df, element):
        """
        Converts monthly data to daily data.
        """
        current_date = datetime.now().date()
        df = df[['ID', 'YEAR', 'Month'] + [f'VALUE{i}' for i in range(1, 32)]]
        
        def expand_row(row):
            year, month = row['YEAR'], row['Month']
            days_in_month = calendar.monthrange(year, month)[1]
            dates = pd.date_range(f"{year}-{month:02d}-01", periods=days_in_month, freq='D')
            dates = dates[dates <= pd.Timestamp(current_date)]
            if len(dates) == 0:
                return pd.DataFrame()
            return pd.DataFrame({
                'DATE': dates.strftime('%Y-%m-%d'),
                'ID': row['ID'],
                element: [row.get(f'VALUE{day}', np.nan) for day in range(1, len(dates) + 1)]
            })

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            daily_dfs = list(executor.map(expand_row, [row for _, row in df.iterrows()]))
        
        return pd.concat(daily_dfs, ignore_index=True) if daily_dfs else pd.DataFrame()

    def transform_all_variables_to_daily(self):
        """
        Transforms all weather variables into daily data without saving to CSV.
        """
        def process_variable(var_df_tuple):
            var, df = var_df_tuple
            transformed_df = self.convert_to_daily_data(df, var)
            return var, transformed_df

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(process_variable, self.variable_dfs.items())
        
        self.transformed_dfs = dict(results)
        return self.transformed_dfs

    def merge_dataframes(self):
        """
        Combines all transformed DataFrames in memory.
        """
        if not self.transformed_dfs:
            raise ValueError("No transformed DataFrames available. Run transform_all_variables_to_daily first.")
        
        combined_df = list(self.transformed_dfs.values())[0]
        for var, df in list(self.transformed_dfs.items())[1:]:
            combined_df = pd.merge(combined_df, df, on=['ID', 'DATE'], how='outer', 
                                 suffixes=('_left', '_right'), copy=False)
        return combined_df

    def add_season_column(self, df):
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        month = df['DATE'].dt.month
        df['Season'] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ['Winter', 'Spring', 'Summer'], default='Fall'
        )
        return df

    def normalize_weather_columns(self, df):
        df[self.weather_variables] = df[self.weather_variables].apply(pd.to_numeric, errors='coerce')
        weather_cols = [col for col in self.weather_variables if col not in ['SNOW', 'SNWD']]
        if weather_cols:
            df[weather_cols] = df[weather_cols] / 10
        return df

    def process_data(self):
        self.extract_variable_dataframes()
        self.transform_all_variables_to_daily()
        combined_df = self.merge_dataframes()
        combined_df = self.add_season_column(combined_df)
        final_df = self.normalize_weather_columns(combined_df)
        return final_df
    
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
        Converts monthly data to daily data for a specific weather variable (element),
        ensuring that missing days are included with NaN values for missing data, only valid
        days are added for each month, and no data is included after the current date.
        """
        transformed_data = []
        
        # Get the current date
        current_date = datetime.now().date()  # Current date (without time)

        for _, row in df.iterrows():
            station_id = row['ID']
            year = row['YEAR']
            month = row['Month']
            
            # Determine the number of days in the month (account for leap year for February)
            if month == 2:  # February
                if calendar.isleap(year):
                    days_in_month = 29  # Leap year
                else:
                    days_in_month = 28  # Common year
            elif month in [4, 6, 9, 11]:  # April, June, September, November
                days_in_month = 30
            else:  # All other months
                days_in_month = 31
            
            # Loop over each valid day in the month (1 to days_in_month)
            for day in range(1, days_in_month + 1):
                date_str = f"{year}-{month:02d}-{day:02d}"
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()  # Convert to date object
            
                # Skip adding data if the date is after the current date
                if date_obj > current_date:
                    continue  # Skip this row
                
                value_column = f'VALUE{day}'
                
                # Get the value for this specific day, or use NaN if not available
                element_value = row.get(value_column, np.nan)
                
                # Append the transformed data
                transformed_data.append({
                    'DATE': date_str,
                    'ID': station_id,
                    element: element_value
                })
        
        # Return the transformed data as a DataFrame
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
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

class WeatherDataCleaner:
    def __init__(self, df, max_distance_km=100):
        self.df = df.copy()
        self.station_correlations = {}
        self.spatial_correlations = {}
        self.max_distance_km = max_distance_km
        self.required_cols = ['DATE', 'ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'Season', 'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        if not all(col in df.columns for col in self.required_cols):
            raise ValueError(f"Missing columns: {set(self.required_cols) - set(df.columns)}")
        self.stations = self.df[['ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION']].drop_duplicates().set_index('ID')

    def preprocess_dates(self):
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df['Month'] = self.df['DATE'].dt.month
        self.df['Day'] = self.df['DATE'].dt.day
        self.df['Year'] = self.df['DATE'].dt.year

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def compute_station_correlations(self):
        for station_id, group in self.df.groupby('ID'):
            n_tmin = group['TMIN'].notna().sum()
            if n_tmin > 1:
                self.station_correlations[station_id] = {
                    'tmin_prcp': group['TMIN'].corr(group['PRCP']) if group['PRCP'].notna().sum() > 1 else 0,
                    'tmax_prcp': group['TMAX'].corr(group['PRCP']) if group['PRCP'].notna().sum() > 1 else 0,
                    'snwd_snow': group['SNWD'].corr(group['SNOW']) if group['SNOW'].notna().sum() > 1 else 1,
                    'prcp_snow': group['PRCP'].corr(group['SNOW']) if group['SNOW'].notna().sum() > 1 else 0
                }
            else:
                self.station_correlations[station_id] = {'tmin_prcp': 0, 'tmax_prcp': 0, 'snwd_snow': 1, 'prcp_snow': 0}

    def compute_spatial_correlations(self):
        variables = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        for var in variables:
            self.spatial_correlations[var] = {}
            for station_id in self.stations.index:
                self.spatial_correlations[var][station_id] = {}
                station_data = self.df[self.df['ID'] == station_id][['DATE', var]].set_index('DATE')
                lat1, lon1, elev1 = self.stations.loc[station_id]
                for other_id in self.stations.index:
                    if other_id != station_id:
                        lat2, lon2, elev2 = self.stations.loc[other_id]
                        distance = self.haversine_distance(lat1, lon1, lat2, lon2)
                        if distance <= self.max_distance_km:
                            other_data = self.df[self.df['ID'] == other_id][['DATE', var]].set_index('DATE')
                            merged = station_data.join(other_data, lsuffix='_self', rsuffix='_other', how='inner')
                            if len(merged) > 1:
                                corr = merged[f'{var}_self'].corr(merged[f'{var}_other'])
                                self.spatial_correlations[var][station_id][other_id] = {
                                    'distance': distance,
                                    'corr': corr if pd.notna(corr) else 0,
                                    'elev_diff': elev1 - elev2,
                                    'lat_diff': lat1 - lat2  # Latitude difference (positive if north)
                                }

    def fill_missing_with_interpolation(self):
        for station_id, group in self.df.groupby('ID'):
            for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                mask = self.df.loc[group.index, col].isna()
                if mask.any():
                    interpolated = group[col].interpolate(method='linear').ffill().bfill()
                    self.df.loc[group.index[mask], col] = interpolated[mask]

    def fill_missing_with_spatial_interpolation(self):
        variables = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        lapse_rate = -6.5 / 1000  # °C per meter
        lat_gradient = -0.5  # °C per degree latitude (cooler northward)
        for station_id, group in self.df.groupby('ID'):
            for var in variables:
                missing_mask = group[var].isna()
                if missing_mask.any():
                    nearby_data = []
                    lat1, lon1, elev1 = self.stations.loc[station_id]
                    for other_id in self.spatial_correlations[var].get(station_id, {}):
                        distance = self.spatial_correlations[var][station_id][other_id]['distance']
                        corr = self.spatial_correlations[var][station_id][other_id]['corr']
                        elev_diff = self.spatial_correlations[var][station_id][other_id]['elev_diff']
                        lat_diff = self.spatial_correlations[var][station_id][other_id]['lat_diff']
                        other_group = self.df[self.df['ID'] == other_id][['DATE', var]].set_index('DATE')
                        for idx in group.index[missing_mask]:
                            date = self.df.at[idx, 'DATE']
                            if date in other_group.index and pd.notna(other_group.loc[date, var]):
                                weight = 1 / distance * (corr if corr > 0 else 0.1)  # Minimum weight to avoid division by zero
                                value = other_group.loc[date, var]
                                if var in ['TMIN', 'TMAX']:
                                    elev_adjust = lapse_rate * elev_diff
                                    lat_adjust = lat_gradient * lat_diff
                                    adjusted_value = value + elev_adjust + lat_adjust
                                else:  # PRCP, SNOW, SNWD
                                    adjusted_value = value * (corr if corr > 0 else 0.1)  # Scale by correlation
                                nearby_data.append((idx, adjusted_value, weight))
                    if nearby_data:
                        for idx in group.index[missing_mask]:
                            values_weights = [(val, w) for i, val, w in nearby_data if i == idx]
                            if values_weights:
                                total_weight = sum(w for _, w in values_weights)
                                weighted_value = sum(val * w for val, w in values_weights) / total_weight
                                self.df.at[idx, var] = weighted_value
                            else:
                                # Fallback to mean of nearby stations if no exact date match
                                nearby_values = [v for _, v, _ in nearby_data]
                                if nearby_values:
                                    self.df.at[idx, var] = np.mean(nearby_values)

    def fill_missing_with_relationships(self):
        for station_id, group in self.df.groupby('ID'):
            corr = self.station_correlations.get(station_id, {'tmin_prcp': 0, 'tmax_prcp': 0, 'prcp_snow': 0})
            # Fill TMIN using PRCP
            mask = group['TMIN'].isna() & group['PRCP'].notna()
            if mask.any():
                mean_tmin = group['TMIN'].mean()
                mean_prcp = group['PRCP'].mean()
                if pd.notna(mean_tmin) and pd.notna(mean_prcp):
                    self.df.loc[group.index[mask], 'TMIN'] = mean_tmin + corr['tmin_prcp'] * (group['PRCP'] - mean_prcp)
            # Fill TMAX using PRCP
            mask = group['TMAX'].isna() & group['PRCP'].notna()
            if mask.any():
                mean_tmax = group['TMAX'].mean()
                mean_prcp = group['PRCP'].mean()
                if pd.notna(mean_tmax) and pd.notna(mean_prcp):
                    self.df.loc[group.index[mask], 'TMAX'] = mean_tmax + corr['tmax_prcp'] * (group['PRCP'] - mean_prcp)
            # Fill PRCP using TMIN
            mask = group['PRCP'].isna() & group['TMIN'].notna()
            if mask.any():
                mean_prcp = group['PRCP'].mean()
                mean_tmin = group['TMIN'].mean()
                if pd.notna(mean_prcp) and pd.notna(mean_tmin):
                    self.df.loc[group.index[mask], 'PRCP'] = mean_prcp + corr['tmin_prcp'] * (group['TMIN'] - mean_tmin)
            # Fill SNOW using PRCP, TMIN, and TMAX
            mask = group['SNOW'].isna() & group['PRCP'].notna() & group['TMIN'].notna() & group['TMAX'].notna()
            if mask.any():
                snow_fraction = np.where(
                    group['TMIN'] < 0,  # Below freezing
                    np.where(group['TMAX'] <= 0, 1.0,  # All snow if TMAX ≤ 0°C
                             np.where(group['TMAX'] < 2, 0.5, 0.25)),  # Mixed precipitation if TMAX < 2°C
                    0.0  # No snow if TMIN ≥ 0°C
                )
                self.df.loc[group.index[mask], 'SNOW'] = group['PRCP'] * snow_fraction

    def adjust_snwd(self):
        for station_id, group in self.df.groupby('ID'):
            corr = self.station_correlations.get(station_id, {'snwd_snow': 1})
            snwd = group['SNWD'].copy()
            snow = group['SNOW'].fillna(0)
            tmax = group['TMAX'].fillna(0)
            for i in range(len(group)):
                if pd.isna(snwd.iloc[i]):
                    if i == 0:
                        snwd.iloc[i] = snow.iloc[i] * corr['snwd_snow']  # Initial snow depth
                    else:
                        prev_snwd = snwd.iloc[i-1] if pd.notna(snwd.iloc[i-1]) else 0
                        new_snow = snow.iloc[i] * corr['snwd_snow']
                        if new_snow > 0:
                            snwd.iloc[i] = prev_snwd + new_snow
                        elif tmax.iloc[i] > 0 and prev_snwd > 0:  # Melting if TMAX > 0°C
                            melt_factor = 0.8 - 0.1 * (tmax.iloc[i] / 10)  # More melt with higher TMAX
                            snwd.iloc[i] = prev_snwd * max(melt_factor, 0.5)  # Minimum 50% retention
                        else:
                            snwd.iloc[i] = prev_snwd
            self.df.loc[group.index, 'SNWD'] = snwd

    def fill_missing_with_same_date_means(self):
        for station_id, group in self.df.groupby('ID'):
            date_means = group.groupby(['Month', 'Day'])[['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']].mean()
            for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                still_missing = self.df.loc[group.index, col].isna()
                if still_missing.any():
                    for idx in group.index[still_missing]:
                        month, day = self.df.at[idx, 'Month'], self.df.at[idx, 'Day']
                        if (month, day) in date_means.index and pd.notna(date_means.loc[(month, day), col]):
                            self.df.at[idx, col] = date_means.loc[(month, day), col]

    def fill_missing_with_seasonal_means(self):
        for station_id, group in self.df.groupby('ID'):
            seasonal_means = group.groupby('Season')[['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']].mean()
            for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                still_missing = self.df.loc[group.index, col].isna()
                if still_missing.any():
                    for idx in group.index[still_missing]:
                        season = self.df.at[idx, 'Season']
                        if season in seasonal_means.index and pd.notna(seasonal_means.loc[season, col]):
                            self.df.at[idx, col] = seasonal_means.loc[season, col]
                        else:
                            station_mean = group[col].mean()
                            self.df.at[idx, col] = station_mean if pd.notna(station_mean) else (0 if col in ['PRCP', 'SNOW', 'SNWD'] else 10)

    def clean_all(self):
        self.preprocess_dates()
        self.compute_station_correlations()
        self.compute_spatial_correlations()
        self.fill_missing_with_interpolation()
        self.fill_missing_with_spatial_interpolation()
        self.fill_missing_with_relationships()
        self.adjust_snwd()
        self.fill_missing_with_same_date_means()
        self.fill_missing_with_seasonal_means()
        weather_cols = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        self.df[weather_cols] = self.df[weather_cols].round(2)
        self.df = self.df.drop(columns=['Month', 'Day', 'Year'])
        return self.df
