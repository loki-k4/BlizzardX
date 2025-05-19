## ConfigManager Usage

The `ConfigManager` class allows you to load and manage configuration data from JSON files. It supports loading a single configuration file or all JSON files from a specified directory. The configuration data can be accessed using dot notation for nested keys.

### Setup

1. **Initialization**:
   - Create an instance of the `ConfigManager` by specifying the `config_directory` where your JSON files are stored (default is `"config"`).

   ```python
   config_manager = ConfigManager(config_directory="src/config")
   ```

2. **Loading a Configuration File**:
   - Use `load_config()` to load a specific configuration file (e.g., `settings.json`).
   
   ```python
   config_manager.load_config("settings.json")
   ```

3. **Loading All Configuration Files**:
   - Use `load_all_configs()` to automatically load all JSON files in the specified `config_directory`.

   ```python
   config_manager.load_all_configs()
   ```

4. **Accessing Configuration Data**:
   - Use the `get()` method to retrieve a specific value. You can use dot notation for nested keys (e.g., `"cpu_config.cpu_usage_limit"`).
   - Optionally, you can provide a default value if the key doesn't exist.
   
   ```python
   cpu_usage_limit = config_manager.get("settings.json", "cpu_config.cpu_usage_limit", default=85)
   print(cpu_usage_limit)
   ```

5. **List Loaded Config Files**:
   - Use `list_loaded_configs()` to see a list of all the configuration files that have been loaded.
   
   ```python
   print(config_manager.list_loaded_configs())
   ```
---