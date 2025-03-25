# cpu_monitor.py

import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from src.config.config_manager import ConfigManager  

class CPUMonitor:
    def __init__(self, config_file='settings.json', config_directory="/workspaces/BlizzardX/src/config"):
        # Initialize ConfigManager and load config
        self.config_manager = ConfigManager(config_directory)
        self.config_manager.load_config(config_file)
        cpu_config = self.config_manager.get(config_file, 'cpu_config', {})
        self.cpu_usage_limit = cpu_config.get('cpu_usage_limit', 85)
        self.max_workers = cpu_config.get('max_concurrent_workers', 12)
        self.workers = self.max_workers
        self.cpu_check_interval = cpu_config.get('cpu_check_interval', 2)
        self.executor = ThreadPoolExecutor(max_workers=self.workers)

    def monitor_cpu_usage(self):
        """Monitors CPU usage and dynamically adjusts the number of workers based on usage."""
        while True:
            cpu_usage = psutil.cpu_percent(interval=self.cpu_check_interval)
            if cpu_usage > self.cpu_usage_limit:
                self.decrease_workers()
            else:
                self.increase_workers()

    def decrease_workers(self):
        """Decrease the number of workers if CPU usage is too high."""
        if self.workers > 2:  # Ensure there's at least one worker
            self.workers -= 2
            self.executor._max_workers = self.workers
            print(f"CPU usage is high! Decreasing workers to {self.workers}")

    def increase_workers(self):
        """Increase the number of workers if CPU usage is low enough."""
        if self.workers < self.max_workers:
            self.workers += 2
            self.executor._max_workers = self.workers
            print(f"CPU usage is stable. Increasing workers to {self.workers}")

    def start_cpu_monitoring(self):
        """Start monitoring CPU usage in a separate thread."""
        cpu_monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
        cpu_monitor_thread.daemon = True
        cpu_monitor_thread.start()

    def get_executor(self):
        """Return the executor instance to be used in other parts of the code."""
        return self.executor
