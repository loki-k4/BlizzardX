import os
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor
import logging

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

class WeatherDataTransformer:
    def __init__(self, data, weather_variables, max_workers=4):
        self.logger = setup_logging()
        self.logger.info("Initializing WeatherDataTransformer")
        
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Input 'data' must be a pandas DataFrame")
            raise ValueError("Input 'data' must be a pandas DataFrame")
        if data.empty:
            self.logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame is empty")
        if not weather_variables:
            self.logger.error("Weather variables list cannot be empty")
            raise ValueError("Weather variables list cannot be empty")
        
        self.data = data.copy()
        self.weather_variables = weather_variables
        self.variable_dfs = {}
        self.transformed_dfs = {}
        self.max_workers = max_workers
        self.logger.info(f"Initialized with {len(weather_variables)} weather variables and max_workers={max_workers}")

    def extract_variable_dataframes(self):
        self.logger.info("Extracting variable DataFrames")
        def process_variable(var):
            try:
                var_df = self.data[self.data['ELEMENT'] == var].copy()
                self.logger.debug(f"Extracted {len(var_df)} rows for variable {var}")
                return var, var_df
            except Exception as e:
                self.logger.error(f"Error extracting DataFrame for variable {var}: {e}")
                return var, pd.DataFrame()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = executor.map(process_variable, self.weather_variables)
            
            self.variable_dfs = dict(results)
            self.logger.info(f"Extracted DataFrames for {len(self.variable_dfs)} variables")
            return self.variable_dfs
        except Exception as e:
            self.logger.error(f"Error in extract_variable_dataframes: {e}")
            raise

    def convert_to_daily_data(self, df, element):
        self.logger.info(f"Converting {element} to daily data")
        current_date = datetime.now().date()
        
        try:
            if df.empty:
                self.logger.warning(f"Input DataFrame for {element} is empty")
                return pd.DataFrame()
            
            df = df[['ID', 'YEAR', 'Month'] + [f'VALUE{i}' for i in range(1, 32)]]
            
            def expand_row(row):
                try:
                    year, month = row['YEAR'], row['Month']
                    days_in_month = calendar.monthrange(year, month)[1]
                    dates = pd.date_range(f"{year}-{month:02d}-01", periods=days_in_month, freq='D')
                    dates = dates[dates <= pd.Timestamp(current_date)]
                    if len(dates) == 0:
                        self.logger.debug(f"No valid dates for {year}-{month:02d} in {element}")
                        return pd.DataFrame()
                    return pd.DataFrame({
                        'DATE': dates.strftime('%Y-%m-%d'),
                        'ID': row['ID'],
                        element: [row.get(f'VALUE{day}', np.nan) for day in range(1, len(dates) + 1)]
                    })
                except Exception as e:
                    self.logger.warning(f"Error expanding row for {element}, ID={row['ID']}: {e}")
                    return pd.DataFrame()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                daily_dfs = list(executor.map(expand_row, [row for _, row in df.iterrows()]))
            
            if not daily_dfs:
                self.logger.warning(f"No daily data generated for {element}")
                return pd.DataFrame()
            
            result = pd.concat(daily_dfs, ignore_index=True)
            self.logger.info(f"Converted {element} to {len(result)} daily records")
            return result
        except Exception as e:
            self.logger.error(f"Error converting {element} to daily data: {e}")
            raise

    def transform_all_variables_to_daily(self):
        self.logger.info("Transforming all variables to daily data")
        def process_variable(var_df_tuple):
            var, df = var_df_tuple
            try:
                transformed_df = self.convert_to_daily_data(df, var)
                return var, transformed_df
            except Exception as e:
                self.logger.error(f"Error transforming variable {var}: {e}")
                return var, pd.DataFrame()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = executor.map(process_variable, self.variable_dfs.items())
            
            self.transformed_dfs = dict(results)
            self.logger.info(f"Transformed {len(self.transformed_dfs)} variables to daily data")
            return self.transformed_dfs
        except Exception as e:
            self.logger.error(f"Error in transform_all_variables_to_daily: {e}")
            raise

    def merge_dataframes(self):
        self.logger.info("Merging transformed DataFrames")
        try:
            if not self.transformed_dfs:
                self.logger.error("No transformed DataFrames available")
                raise ValueError("No transformed DataFrames available. Run transform_all_variables_to_daily first.")
            
            combined_df = list(self.transformed_dfs.values())[0]
            for var, df in list(self.transformed_dfs.items())[1:]:
                combined_df = pd.merge(combined_df, df, on=['ID', 'DATE'], how='outer', 
                                       suffixes=('_left', '_right'), copy=False)
            self.logger.info(f"Merged {len(self.transformed_dfs)} DataFrames into one with {len(combined_df)} rows")
            return combined_df
        except Exception as e:
            self.logger.error(f"Error merging DataFrames: {e}")
            raise

    def add_season_column(self, df):
        self.logger.info("Adding Season column")
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            month = df['DATE'].dt.month
            df['Season'] = np.select(
                [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
                ['Winter', 'Spring', 'Summer'], default='Fall'
            )
            self.logger.info(f"Added Season column to DataFrame with {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"Error adding Season column: {e}")
            raise

    def normalize_weather_columns(self, df):
        self.logger.info("Normalizing weather columns")
        try:
            df[self.weather_variables] = df[self.weather_variables].apply(pd.to_numeric, errors='coerce')
            weather_cols = [col for col in self.weather_variables if col not in ['SNOW', 'SNWD']]
            if weather_cols:
                df[weather_cols] = df[weather_cols] / 10
            self.logger.info(f"Normalized {len(weather_cols)} weather columns")
            return df
        except Exception as e:
            self.logger.error(f"Error normalizing weather columns: {e}")
            raise

    def process_data(self):
        self.logger.info("Running data processing pipeline")
        try:
            self.extract_variable_dataframes()
            self.transform_all_variables_to_daily()
            combined_df = self.merge_dataframes()
            combined_df = self.add_season_column(combined_df)
            final_df = self.normalize_weather_columns(combined_df)
            self.logger.info(f"Completed data processing pipeline, resulting in {len(final_df)} rows")
            return final_df
        except Exception as e:
            self.logger.error(f"Error in data processing pipeline: {e}")
            raise

class WeatherDataImputer:
    def __init__(self, df, max_distance_km=100):
        self.logger = setup_logging()
        self.logger.info("Initializing WeatherDataImputer")
        
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input 'df' must be a pandas DataFrame")
            raise ValueError("Input 'df' must be a pandas DataFrame")
        if df.empty:
            self.logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame is empty")
        
        self.df = df.copy()
        self.station_correlations = {}
        self.spatial_correlations = {}
        self.max_distance_km = max_distance_km
        self.required_cols = ['DATE', 'ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'Season', 
                             'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        if not all(col in df.columns for col in self.required_cols):
            missing_cols = set(self.required_cols) - set(df.columns)
            self.logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        self.stations = self.df[['ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION']].drop_duplicates().set_index('ID')
        self.logger.info(f"Initialized with {len(self.stations)} unique stations and max_distance_km={max_distance_km}")

    def preprocess_dates(self):
        self.logger.info("Preprocessing date-related columns")
        try:
            self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
            self.df['Month'] = self.df['DATE'].dt.month
            self.df['Day'] = self.df['DATE'].dt.day
            self.df['Year'] = self.df['DATE'].dt.year
            self.logger.info(f"Preprocessed dates for {len(self.df)} rows")
        except Exception as e:
            self.logger.error(f"Error preprocessing dates: {e}")
            raise

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        self.logger.debug("Calculating Haversine distance")
        try:
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c
        except Exception as e:
            self.logger.error(f"Error calculating Haversine distance: {e}")
            raise

    def compute_station_correlations(self):
        self.logger.info("Computing station correlations")
        try:
            for station_id, group in self.df.groupby('ID'):
                n_tmin = group['TMIN'].notna().sum()
                self.station_correlations[station_id] = {
                    'tmin_prcp': group['TMIN'].corr(group['PRCP']) if n_tmin > 1 and group['PRCP'].notna().sum() > 1 else 0,
                    'tmax_prcp': group['TMAX'].corr(group['PRCP']) if n_tmin > 1 and group['PRCP'].notna().sum() > 1 else 0,
                    'snwd_snow': group['SNWD'].corr(group['SNOW']) if group['SNOW'].notna().sum() > 1 else 1,
                    'prcp_snow': group['PRCP'].corr(group['SNOW']) if group['SNOW'].notna().sum() > 1 else 0
                }
                self.logger.debug(f"Computed correlations for station {station_id}: {self.station_correlations[station_id]}")
            self.logger.info(f"Computed correlations for {len(self.station_correlations)} stations")
        except Exception as e:
            self.logger.error(f"Error computing station correlations: {e}")
            raise

    def compute_spatial_correlations(self):
        self.logger.info("Computing spatial correlations")
        variables = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        try:
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
                self.logger.debug(f"Computed spatial correlations for variable {var}")
            self.logger.info(f"Computed spatial correlations for {len(variables)} variables")
        except Exception as e:
            self.logger.error(f"Error computing spatial correlations: {e}")
            raise

    def fill_missing_with_interpolation(self):
        self.logger.info("Filling missing values with interpolation")
        try:
            for station_id, group in self.df.groupby('ID'):
                for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                    mask = self.df.loc[group.index, col].isna()
                    if mask.any():
                        interpolated = group[col].interpolate(method='linear').ffill().bfill()
                        self.df.loc[group.index[mask], col] = interpolated[mask]
                        self.logger.debug(f"Interpolated {mask.sum()} missing values for {col} in station {station_id}")
            self.logger.info("Completed interpolation for missing values")
        except Exception as e:
            self.logger.error(f"Error in interpolation: {e}")
            raise

    def fill_missing_with_spatial_interpolation(self):
        self.logger.info("Filling missing values with spatial interpolation")
        variables = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
        lapse_rate = -6.5 / 1000  # °C per meter
        lat_gradient = -0.5  # °C per degree latitude
        try:
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
                                        self.logger.debug(f"Imputed {var} for station {station_id} using mean of nearby values")
            self.logger.info("Completed spatial interpolation for missing values")
        except Exception as e:
            self.logger.error(f"Error in spatial interpolation: {e}")
            raise

    def fill_missing_with_relationships(self):
        self.logger.info("Filling missing values with station-specific relationships")
        try:
            for station_id, group in self.df.groupby('ID'):
                corr = self.station_correlations.get(station_id, {'tmin_prcp': 0, 'tmax_prcp': 0, 'prcp_snow': 0})
                mean_tmin, mean_tmax, mean_prcp = group['TMIN'].mean(), group['TMAX'].mean(), group['PRCP'].mean()
                
                # Fill TMIN using PRCP
                mask = group['TMIN'].isna() & group['PRCP'].notna()
                if mask.any() and pd.notna(mean_tmin) and pd.notna(mean_prcp):
                    self.df.loc[group.index[mask], 'TMIN'] = mean_tmin + corr['tmin_prcp'] * (group['PRCP'] - mean_prcp)
                    self.logger.debug(f"Imputed TMIN for {mask.sum()} rows in station {station_id} using PRCP")
                
                # Fill TMAX using PRCP
                mask = group['TMAX'].isna() & group['PRCP'].notna()
                if mask.any() and pd.notna(mean_tmax) and pd.notna(mean_prcp):
                    self.df.loc[group.index[mask], 'TMAX'] = mean_tmax + corr['tmax_prcp'] * (group['PRCP'] - mean_prcp)
                    self.logger.debug(f"Imputed TMAX for {mask.sum()} rows in station {station_id} using PRCP")
                
                # Fill PRCP using TMIN
                mask = group['PRCP'].isna() & group['TMIN'].notna()
                if mask.any() and pd.notna(mean_prcp) and pd.notna(mean_tmin):
                    self.df.loc[group.index[mask], 'PRCP'] = mean_prcp + corr['tmin_prcp'] * (group['TMIN'] - mean_tmin)
                    self.logger.debug(f"Imputed PRCP for {mask.sum()} rows in station {station_id} using TMIN")
                
                # Fill SNOW using PRCP, TMIN, and TMAX
                mask = group['SNOW'].isna() & group['PRCP'].notna() & group['TMIN'].notna() & group['TMAX'].notna()
                if mask.any():
                    snow_fraction = np.where(
                        group['TMIN'] < 0,
                        np.where(group['TMAX'] <= 0, 1.0, np.where(group['TMAX'] < 2, 0.5, 0.25)),
                        0.0
                    )
                    self.df.loc[group.index[mask], 'SNOW'] = group['PRCP'] * snow_fraction
                    self.logger.debug(f"Imputed SNOW for {mask.sum()} rows in station {station_id} using PRCP and temperature")
            self.logger.info("Completed filling missing values with relationships")
        except Exception as e:
            self.logger.error(f"Error in filling missing values with relationships: {e}")
            raise

    def adjust_snwd(self):
        self.logger.info("Adjusting SNWD based on SNOW and TMAX")
        try:
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
                self.logger.debug(f"Adjusted SNWD for station {station_id}")
            self.logger.info("Completed SNWD adjustment")
        except Exception as e:
            self.logger.error(f"Error adjusting SNWD: {e}")
            raise

    def fill_missing_with_same_date_means(self):
        self.logger.info("Filling missing values with same date means")
        try:
            for station_id, group in self.df.groupby('ID'):
                date_means = group.groupby(['Month', 'Day'])[['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']].mean()
                for col in ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']:
                    still_missing = self.df.loc[group.index, col].isna()
                    if still_missing.any():
                        for idx in group.index[still_missing]:
                            month, day = self.df.at[idx, 'Month'], self.df.at[idx, 'Day']
                            if (month, day) in date_means.index and pd.notna(date_means.loc[(month, day), col]):
                                self.df.at[idx, col] = date_means.loc[(month, day), col]
                                self.logger.debug(f"Imputed {col} for station {station_id} on {month}-{day}")
            self.logger.info("Completed filling missing values with same date means")
        except Exception as e:
            self.logger.error(f"Error in filling missing values with same date means: {e}")
            raise

    def fill_missing_with_seasonal_means(self):
        self.logger.info("Filling missing values with seasonal means")
        try:
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
                            self.logger.debug(f"Imputed {col} for station {station_id} in season {season}")
            self.logger.info("Completed filling missing values with seasonal means")
        except Exception as e:
            self.logger.error(f"Error in filling missing values with seasonal means: {e}")
            raise

    def clean_all(self):
        self.logger.info("Running full cleaning pipeline")
        try:
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
            self.logger.info(f"Completed cleaning pipeline, resulting in {len(self.df)} rows")
            return self.df
        except Exception as e:
            self.logger.error(f"Error in cleaning pipeline: {e}")
            raise