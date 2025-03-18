Sure! Here is a simplified `README.md` file specifically for the `config` folder and its usage:

---

# `config` Folder - Configuration Management

This folder contains the `ConfigManager` class, which is responsible for managing and loading configuration files in JSON format. The configuration files can be used to store URLs or any other settings required for your project.

## Directory Structure

```
/config
    __init__.py            # Initialize the config module
    config.py              # ConfigManager class for loading and retrieving configurations
    settings.json          # Example config file with URLs and settings (you can add your own)
```

## Usage of `ConfigManager`

### 1. Initialize `ConfigManager`

Import the `ConfigManager` class and initialize it. Optionally, specify the directory where your configuration files are stored (defaults to `"config"`):

```python
from config.config import ConfigManager

# Initialize with default config directory
config_manager = ConfigManager(config_directory="config")
```

### 2. Load Configuration Files

Use the `load_all_configs()` method to load all `.json` configuration files in the specified directory:

```python
config_manager.load_all_configs()
```

### 3. Retrieve Configuration Values

You can retrieve configuration values from the loaded files using the `get()` method. The first parameter is the config file name (e.g., `"settings.json"`), and the second parameter is the key you want to access:

```python
# Retrieve URLs from the settings.json file
inventory_url = config_manager.get("settings.json", "inventory")
stations_url = config_manager.get("settings.json", "stations")
countries_url = config_manager.get("settings.json", "countries")
states_url = config_manager.get("settings.json", "states")
state_raw_url = config_manager.get("settings.json", "state_raw")
```

### 4. List Loaded Config Files

You can also list all the configuration files that have been loaded:

```python
loaded_configs = config_manager.list_loaded_configs()
print("Loaded Configurations:", loaded_configs)
```

### Example Usage

```python
from config.config import ConfigManager

# Initialize ConfigManager with the directory where config files are stored
config_manager = ConfigManager(config_directory="config")

# Load all configuration files
config_manager.load_all_configs()

# List all loaded configuration files
loaded_configs = config_manager.list_loaded_configs()
print("Loaded Configurations:", loaded_configs)

# Retrieve specific configuration values (URLs in this case)
inventory_url = config_manager.get("settings.json", "inventory")
stations_url = config_manager.get("settings.json", "stations")

print("Inventory URL:", inventory_url)
print("Stations URL:", stations_url)
```

## Configuration File Example

Here is an example of what the `settings.json` configuration file might look like:

```json
{
    "inventory": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt",
    "stations": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt",
    "countries": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-countries.txt",
    "states": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-states.txt",
    "state_raw": "https://raw.githubusercontent.com/georgique/world-geojson/develop/states/usa/new_hampshire.json"
}
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README focuses solely on explaining the usage of the `config` folder and its `ConfigManager` class. It guides users on how to load configuration files, retrieve values, and how to structure the `settings.json` file.