import psutil
import time
import threading

class CPUManager:
    def __init__(self, config):
        # Get values from the config
        self.cpu_usage_limit = config["cpu_usage_limit"]
        self.cpu_check_interval = config["cpu_check_interval"]
        self.workers = config["max_concurrent_workers"]
        self.dynamic_worker_adjustment = config["dynamic_worker_adjustment"]
        self.adjusting_workers = False

    def start_cpu_monitoring(self):
        """Start the CPU monitoring in a separate thread."""
        self.monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def monitor_cpu_usage(self):
        """Monitor the CPU usage and adjust workers dynamically."""
        while True:
            cpu_usage = psutil.cpu_percent(interval=self.cpu_check_interval)

            if cpu_usage > self.cpu_usage_limit and not self.adjusting_workers:
                print(f"CPU usage {cpu_usage}% exceeds the limit ({self.cpu_usage_limit}%). Adjusting workers.")
                self.adjust_workers(down=True)
            elif cpu_usage < self.cpu_usage_limit and self.adjusting_workers:
                print(f"CPU usage {cpu_usage}% is below the limit ({self.cpu_usage_limit}%). Restoring workers.")
                self.adjust_workers(down=False)

            time.sleep(self.cpu_check_interval)

    def adjust_workers(self, down=False):
        """Dynamically adjust the number of workers based on CPU usage."""
        if down:
            # Decrease workers if CPU usage is too high
            self.workers = max(1, self.workers - 2)  # Ensure at least 1 worker
        else:
            # Increase workers if CPU usage is under the limit
            self.workers = min(12, self.workers + 2)  # Max of 12 workers

        self.adjusting_workers = True
        print(f"Adjusted workers to {self.workers}.")

    def get_worker_count(self):
        """Get the current number of workers."""
        return self.workers
