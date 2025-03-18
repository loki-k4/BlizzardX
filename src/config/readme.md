Here's the updated version of your README for the `config` folder, with a focus on clarity and accuracy:

---

# `config` Folder - Configuration Management

This folder contains the `ConfigManager` class, which is responsible for managing and loading configuration files in JSON format. These configuration files can store various settings, such as URLs or other project-specific values.

## Directory Structure

```
src/config
    __init__.py            # Initialize the config module
    configmanager.py              # Contains the ConfigManager class for loading and retrieving configurations
    urls.json              # Example config file containing URLs and other settings (you can add your own)
```

## Usage of `ConfigManager`

### 1. Initialize `ConfigManager`

Import the `ConfigManager` class and initialize it. Optionally, you can specify the directory where your configuration files are stored (the default is `"config"`):

```python
from src.config.configmanager import ConfigManager

# Initialize with the default config directory
config_manager = ConfigManager(config_directory="src/config")
```

### 2. Load Configuration Files

Use the `load_all_configs()` method to load all `.json` configuration files in the specified directory:

```python
config_manager.load_all_configs()
```

This method will automatically load any `.json` files found in the directory and store them for later access.

### 3. Retrieve Configuration Values

Once the configurations are loaded, use the `get()` method to retrieve specific values. Pass the configuration file name (e.g., `"urls.json"`) and the key you wish to access (e.g., `"inventory"`):

```python
# Retrieve URLs from the urls.json file
inventory_url = config_manager.get("urls.json", "inventory")
```

### 4. List Loaded Config Files

To see all the configuration files that have been successfully loaded, you can use the `list_loaded_configs()` method:

```python 
print("Loaded Configurations:", config_manager.list_loaded_configs())
```

---