import os
import pandas as pd
import glob

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
    
