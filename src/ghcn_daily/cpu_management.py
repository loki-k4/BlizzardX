import psutil
import time
import threading
from src.config.config_manager import ConfigManager
import json

class CPUManager:
    def __init__(self, config):
        # Get the configuration values from the provided config dictionary
        self.cpu_usage_limit = config.get("cpu_usage_limit", 80)  # Default to 80% if not provided
        self.cpu_check_interval = config.get("cpu_check_interval", 2)  # Default to 2 seconds if not provided
        self.workers = config.get("max_concurrent_workers", 12)  # Default workers set to 12 if not provided
        self.max_workers = config.get("max_concurrent_workers", 12)  # Max concurrent workers (to prevent overload)
        self.dynamic_worker_adjustment = config.get("dynamic_worker_adjustment", True)  # Default to True if not provided
        self.adjusting_workers = False  # Flag to check if we're adjusting the workers dynamically

        # Start CPU monitoring in a separate thread
        self.cpu_monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
        self.cpu_monitor_thread.daemon = True
        self.cpu_monitor_thread.start()

    def monitor_cpu_usage(self):
        while True:
            cpu_usage = psutil.cpu_percent(interval=self.cpu_check_interval)
            print(f"CPU Usage: {cpu_usage}%")
            if cpu_usage > self.cpu_usage_limit:
                print(f"CPU usage exceeds {self.cpu_usage_limit}%, adjusting workers...")
                self.adjust_workers(cpu_usage)
            else:
                if self.adjusting_workers:
                    self.restore_workers()
            time.sleep(self.cpu_check_interval)

    def adjust_workers(self, cpu_usage):
        """Adjust the number of workers dynamically based on CPU usage."""
        if self.dynamic_worker_adjustment:
            if not self.adjusting_workers:
                self.adjusting_workers = True
                # Reduce workers based on CPU usage
                reduction_factor = max(1, int(cpu_usage / self.cpu_usage_limit))
                self.workers = max(1, self.max_workers // reduction_factor)
                print(f"Reducing workers to {self.workers} due to high CPU usage.")
            else:
                print(f"Already adjusting workers, current workers: {self.workers}")

    def restore_workers(self):
        """Restore the number of workers if CPU usage is below the limit."""
        if self.adjusting_workers:
            self.workers = self.max_workers
            self.adjusting_workers = False
            print(f"Restoring workers to {self.workers} as CPU usage is below the limit.")

# Using ConfigManager to load the config
if __name__ == "__main__":
    # Create an instance of ConfigManager
    config_manager = ConfigManager(config_directory="src/config")
    
    # Load the specific config.json file
    config_manager.load_config("config.json")
    
    # Get the CPU configuration from the loaded JSON
    cpu_config = config_manager.get("config.json", "cpu_config", {})
    
    # Initialize CPUManager with the values from the config.json
    cpu_manager = CPUManager(cpu_config)
