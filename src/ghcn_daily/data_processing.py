import os
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor

class WeatherDataTransformer:
    """
    A class to transform weather data from monthly to daily format and clean it.
    """
    def __init__(self, data, weather_variables, max_workers=4):
        self.data = data
        self.weather_variables = weather_variables
        self.variable_dfs = {}
        self.transformed_dfs = {}
        self.max_workers = max_workers

    def extract_variable_dataframes(self):
        """Extracts data for each weather variable into DataFrames."""
        def process_variable(var):
            var_df = self.data[self.data['ELEMENT'] == var].copy()
            return var, var_df

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(process_variable, self.weather_variables)
        
        self.variable_dfs = dict(results)
        return self.variable_dfs

    def convert_to_daily_data(self, df, element):
        """Converts monthly data to daily data."""
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
        """Transforms all weather variables into daily data."""
        def process_variable(var_df_tuple):
            var, df = var_df_tuple
            transformed_df = self.convert_to_daily_data(df, var)
            return var, transformed_df

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(process_variable, self.variable_dfs.items())
        
        self.transformed_dfs = dict(results)
        return self.transformed_dfs

    def merge_dataframes(self):
        """Combines all transformed DataFrames in memory."""
        if not self.transformed_dfs:
            raise ValueError("No transformed DataFrames available. Run transform_all_variables_to_daily first.")
        
        combined_df = list(self.transformed_dfs.values())[0]
        for var, df in list(self.transformed_dfs.items())[1:]:
            combined_df = pd.merge(combined_df, df, on=['ID', 'DATE'], how='outer', 
                                   suffixes=('_left', '_right'), copy=False)
        return combined_df

    def add_season_column(self, df):
        """Adds a 'Season' column based on the 'DATE' column."""
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        month = df['DATE'].dt.month
        df['Season'] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ['Winter', 'Spring', 'Summer'], default='Fall'
        )
        return df

    def normalize_weather_columns(self, df):
        """Normalizes weather columns."""
        df[self.weather_variables] = df[self.weather_variables].apply(pd.to_numeric, errors='coerce')
        weather_cols = [col for col in self.weather_variables if col not in ['SNOW', 'SNWD']]
        if weather_cols:
            df[weather_cols] = df[weather_cols] / 10
        return df

    def process_data(self):
        """Runs the data processing pipeline."""
        self.extract_variable_dataframes()
        self.transform_all_variables_to_daily()
        combined_df = self.merge_dataframes()
        combined_df = self.add_season_column(combined_df)
        final_df = self.normalize_weather_columns(combined_df)
        return final_df

class WeatherDataImputer:
    def __init__(self, df, max_distance_km=100):
        self.df = df.copy()
        self.station_correlations = {}
        self.spatial_correlations = {}
        self.max_distance_km = max_distance_km
        self.required_cols = ['DATE', 'ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'Season', 
                              'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        if not all(col in df.columns for col in self.required_cols):
            raise ValueError(f"Missing columns: {set(self.required_cols) - set(df.columns)}")
        self.stations = self.df[['ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION']].drop_duplicates().set_index('ID')

    def preprocess_dates(self):
        """Preprocesses date-related columns."""
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df['Month'] = self.df['DATE'].dt.month
        self.df['Day'] = self.df['DATE'].dt.day
        self.df['Year'] = self.df['DATE'].dt.year

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates Haversine distance between two points."""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def compute_station_correlations(self):
        """Computes correlations within each station."""
        for station_id, group in self.df.groupby('ID'):
            n_tmin = group['TMIN'].notna().sum()
            self.station_correlations[station_id] = {
                'tmin_prcp': group['TMIN'].corr(group['PRCP']) if n_tmin > 1 and group['PRCP'].notna().sum() > 1 else 0,
                'tmax_prcp': group['TMAX'].corr(group['PRCP']) if n_tmin > 1 and group['PRCP'].notna().sum() > 1 else 0,
                'snwd_snow': group['SNWD'].corr(group['SNOW']) if group['SNOW'].notna().sum() > 1 else 1,
                'prcp_snow': group['PRCP'].corr(group['SNOW']) if group['SNOW'].notna().sum() > 1 else 0
            }

    def compute_spatial_correlations(self):
        """Computes spatial correlations between stations."""
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
                                    'lat_diff': lat1 - lat2
                                }

    def fill_missing_with_interpolation(self):
        """Fills missing values with linear interpolation."""
        for station_id, group in self.df.groupby('ID'):
            for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                mask = self.df.loc[group.index, col].isna()
                if mask.any():
                    interpolated = group[col].interpolate(method='linear').ffill().bfill()
                    self.df.loc[group.index[mask], col] = interpolated[mask]

    def fill_missing_with_spatial_interpolation(self):
        """Fills missing values using spatial interpolation."""
        variables = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        lapse_rate = -6.5 / 1000  # °C per meter
        lat_gradient = -0.5  # °C per degree latitude
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
                                weight = 1 / distance * (corr if corr > 0 else 0.1)
                                value = other_group.loc[date, var]
                                if var in ['TMIN', 'TMAX']:
                                    adjusted_value = value + lapse_rate * elev_diff + lat_gradient * lat_diff
                                else:
                                    adjusted_value = value * (corr if corr > 0 else 0.1)
                                nearby_data.append((idx, adjusted_value, weight))
                    if nearby_data:
                        for idx in group.index[missing_mask]:
                            values_weights = [(val, w) for i, val, w in nearby_data if i == idx]
                            if values_weights:
                                total_weight = sum(w for _, w in values_weights)
                                self.df.at[idx, var] = sum(val * w for val, w in values_weights) / total_weight
                            else:
                                nearby_values = [v for _, v, _ in nearby_data]
                                if nearby_values:
                                    self.df.at[idx, var] = np.mean(nearby_values)

    def fill_missing_with_relationships(self):
        """Fills missing values using station-specific relationships."""
        for station_id, group in self.df.groupby('ID'):
            corr = self.station_correlations.get(station_id, {'tmin_prcp': 0, 'tmax_prcp': 0, 'prcp_snow': 0})
            mean_tmin, mean_tmax, mean_prcp = group['TMIN'].mean(), group['TMAX'].mean(), group['PRCP'].mean()
            
            # Fill TMIN using PRCP
            mask = group['TMIN'].isna() & group['PRCP'].notna()
            if mask.any() and pd.notna(mean_tmin) and pd.notna(mean_prcp):
                self.df.loc[group.index[mask], 'TMIN'] = mean_tmin + corr['tmin_prcp'] * (group['PRCP'] - mean_prcp)
            
            # Fill TMAX using PRCP
            mask = group['TMAX'].isna() & group['PRCP'].notna()
            if mask.any() and pd.notna(mean_tmax) and pd.notna(mean_prcp):
                self.df.loc[group.index[mask], 'TMAX'] = mean_tmax + corr['tmax_prcp'] * (group['PRCP'] - mean_prcp)
            
            # Fill PRCP using TMIN
            mask = group['PRCP'].isna() & group['TMIN'].notna()
            if mask.any() and pd.notna(mean_prcp) and pd.notna(mean_tmin):
                self.df.loc[group.index[mask], 'PRCP'] = mean_prcp + corr['tmin_prcp'] * (group['TMIN'] - mean_tmin)
            
            # Fill SNOW using PRCP, TMIN, and TMAX
            mask = group['SNOW'].isna() & group['PRCP'].notna() & group['TMIN'].notna() & group['TMAX'].notna()
            if mask.any():
                snow_fraction = np.where(
                    group['TMIN'] < 0,
                    np.where(group['TMAX'] <= 0, 1.0, np.where(group['TMAX'] < 2, 0.5, 0.25)),
                    0.0
                )
                self.df.loc[group.index[mask], 'SNOW'] = group['PRCP'] * snow_fraction

    def adjust_snwd(self):
        """Adjusts SNWD based on SNOW and TMAX."""
        for station_id, group in self.df.groupby('ID'):
            corr = self.station_correlations.get(station_id, {'snwd_snow': 1})
            snwd = group['SNWD'].copy()
            snow = group['SNOW'].fillna(0)
            tmax = group['TMAX'].fillna(0)
            for i in range(len(group)):
                if pd.isna(snwd.iloc[i]):
                    prev_snwd = snwd.iloc[i-1] if i > 0 and pd.notna(snwd.iloc[i-1]) else 0
                    new_snow = snow.iloc[i] * corr['snwd_snow']
                    if new_snow > 0:
                        snwd.iloc[i] = prev_snwd + new_snow
                    elif tmax.iloc[i] > 0 and prev_snwd > 0:
                        melt_factor = max(0.5, 0.8 - 0.1 * (tmax.iloc[i] / 10))
                        snwd.iloc[i] = prev_snwd * melt_factor
                    else:
                        snwd.iloc[i] = prev_snwd
            self.df.loc[group.index, 'SNWD'] = snwd

    def fill_missing_with_same_date_means(self):
        """Fills missing values with same date means."""
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
        """Fills remaining missing values with seasonal means."""
        for station_id, group in self.df.groupby('ID'):
            seasonal_means = group.groupby('Season')[['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']].mean()
            for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                still_missing = self.df.loc[group.index, col].isna()
                if still_missing.any():
                    for idx in group.index[still_missing]:
                        season = self.df.at[idx, 'Season']
                        value = (seasonal_means.loc[season, col] if season in seasonal_means.index and pd.notna(seasonal_means.loc[season, col])
                                 else group[col].mean() if pd.notna(group[col].mean()) else (0 if col in ['PRCP', 'SNOW', 'SNWD'] else 10))
                        self.df.at[idx, col] = value

    def clean_all(self):
        """Runs the full cleaning pipeline."""
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
