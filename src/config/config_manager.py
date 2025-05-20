import json
import os
import logging
from pathlib import Path

class ConfigManager:
    def __init__(self, config_directory="config"):

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Convert to Path object for robust path handling
        self.config_directory = Path(config_directory).resolve()
        self.config_data = {}
        
        # Create config directory if it doesn't exist
        try:
            self.config_directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Configuration directory set to: {self.config_directory}")
        except Exception as e:
            self.logger.error(f"Failed to create config directory: {e}")
            raise

    def load_config(self, config_file):

        try:
            file_path = self.config_directory / config_file
            self.logger.debug(f"Attempting to load config: {file_path}")
            
            if not file_path.exists():
                self.logger.warning(f"Configuration file {file_path} not found")
                return
            
            with file_path.open('r') as f:
                config = json.load(f)
                self.config_data[config_file] = config
                self.logger.info(f"Successfully loaded config: {config_file}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {config_file}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading {config_file}: {e}")

    def load_all_configs(self):

        try:
            self.logger.info(f"Loading all configs from: {self.config_directory}")
            for file_path in self.config_directory.glob("*.json"):
                self.load_config(file_path.name)
            self.logger.info(f"Finished loading {len(self.config_data)} config files")
        except Exception as e:
            self.logger.error(f"Error loading configs from directory: {e}")

    def get(self, config_file, key, default=None):

        self.logger.debug(f"Retrieving {key} from {config_file}")
        config = self.config_data.get(config_file, {})
        keys = key.split(".")
        
        try:
            for k in keys:
                config = config.get(k, {})
                if not isinstance(config, dict):
                    if config or config == 0 or config == False:
                        self.logger.debug(f"Found value for {key} in {config_file}")
                        return config
                    return default
            self.logger.debug(f"Found nested config for {key} in {config_file}")
            return config if config else default
        except Exception as e:
            self.logger.error(f"Error retrieving {key} from {config_file}: {e}")
            return default

    def list_loaded_configs(self):

        configs = list(self.config_data.keys())
        self.logger.info(f"Listing loaded configs: {configs}")
        return configs