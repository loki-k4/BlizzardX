import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class CPUManager:
    def __init__(self, cpu_config):
        # Load the CPU configuration from the passed config dictionary
        self.cpu_usage_limit = cpu_config.get("cpu_usage_limit", 80)
        self.cpu_check_interval = cpu_config.get("cpu_check_interval", 2)
        self.max_workers = cpu_config.get("max_concurrent_workers", 12)
        self.chunk_size = cpu_config.get("chunk_size", 100)
        self.dynamic_worker_adjustment = cpu_config.get("dynamic_worker_adjustment", True)

        self.workers = self.max_workers  # Starting number of workers
        self.adjusting_workers = False  # Flag for dynamically adjusting workers
        self.executor = ThreadPoolExecutor(max_workers=self.workers)  # Synchronous thread pool
        
        # Start CPU monitoring in a separate thread
        self.cpu_monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
        self.cpu_monitor_thread.daemon = True
        self.cpu_monitor_thread.start()

    def monitor_cpu_usage(self):
        """Monitors CPU usage and dynamically adjusts the number of workers based on usage."""
        while True:
            cpu_usage = psutil.cpu_percent(interval=self.cpu_check_interval)
            if cpu_usage > self.cpu_usage_limit:
                if not self.adjusting_workers:
                    self.adjusting_workers = True
                    self.decrease_workers()
            else:
                if self.adjusting_workers:
                    self.adjusting_workers = False
                    self.increase_workers()
            time.sleep(self.cpu_check_interval)

    def decrease_workers(self):
        """Decrease the number of workers if CPU usage is too high."""
        if self.workers > 2:  # Ensure there's at least one worker
            self.workers -= 2
            self.executor._max_workers = self.workers  # Adjust the max workers for the executor
            print(f"CPU usage is high! Decreasing workers to {self.workers}")

    def increase_workers(self):
        """Increase the number of workers if CPU usage is low enough."""
        if self.workers < self.max_workers:
            self.workers += 2
            self.executor._max_workers = self.workers  # Adjust the max workers for the executor
            print(f"CPU usage is stable. Increasing workers to {self.workers}")
