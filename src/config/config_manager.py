import json
import os

class ConfigManager:
    def __init__(self, config_directory="config"):
        self.config_directory = config_directory  
        self.config_data = {}  

    def load_config(self, config_file):
        try:
            file_path = os.path.join(self.config_directory, config_file)
            with open(file_path, 'r') as f:
                config = json.load(f)
                self.config_data[config_file] = config
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {config_file}.")

    def load_all_configs(self):
        for file in os.listdir(self.config_directory):
            if file.endswith(".json"):
                self.load_config(file)

    def get(self, config_file, key, default=None):
        config = self.config_data.get(config_file, {})
        keys = key.split(".")
        for k in keys:
            config = config.get(k, {})
            if not config:
                return default
        return config if config else default

    def list_loaded_configs(self):
        return list(self.config_data.keys())
