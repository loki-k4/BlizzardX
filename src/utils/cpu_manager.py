import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from src.config.config_manager import ConfigManager

# Configure logging to console only
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class CPUMonitor:
    def __init__(self, config_file='settings.json', config_directory=None):
        self.logger = setup_logging()
        self.logger.info("Initializing CPUMonitor")
        
        # Determine config directory dynamically
        if config_directory is None:
            # Default to 'config' directory relative to the script
            config_directory = Path(__file__).parent.parent / "src" / "config"
        else:
            config_directory = Path(config_directory)
        
        # Ensure config directory exists
        try:
            config_directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Config directory set to: {config_directory}")
        except Exception as e:
            self.logger.error(f"Failed to create config directory: {e}")
            raise

        # Initialize ConfigManager and load config
        try:
            self.config_manager = ConfigManager(str(config_directory))
            self.config_manager.load_config(config_file)
            cpu_config = self.config_manager.get(config_file, 'cpu_config', {})
            self.cpu_usage_limit = cpu_config.get('cpu_usage_limit', 85)
            self.max_workers = cpu_config.get('max_concurrent_workers', 12)
            self.workers = self.max_workers
            self.cpu_check_interval = cpu_config.get('cpu_check_interval', 2)
            self.executor = ThreadPoolExecutor(max_workers=self.workers)
            self.logger.info(f"Loaded config: cpu_usage_limit={self.cpu_usage_limit}, max_workers={self.max_workers}, cpu_check_interval={self.cpu_check_interval}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def monitor_cpu_usage(self):
        """Monitors CPU usage and dynamically adjusts the number of workers based on usage."""
        self.logger.info("Starting CPU usage monitoring")
        while True:
            try:
                cpu_usage = psutil.cpu_percent(interval=self.cpu_check_interval)
                self.logger.info(f"Current CPU usage: {cpu_usage}%")
                if cpu_usage > self.cpu_usage_limit:
                    self.decrease_workers()
                else:
                    self.increase_workers()
            except Exception as e:
                self.logger.error(f"Error in CPU monitoring: {e}")

    def decrease_workers(self):
        """Decrease the number of workers if CPU usage is too high."""
        if self.workers > 2:  # Ensure there's at least one worker
            self.workers -= 2
            self.executor._max_workers = self.workers
            self.logger.info(f"CPU usage is high! Decreasing workers to {self.workers}")
        else:
            self.logger.info("Worker count already at minimum (2)")

    def increase_workers(self):
        """Increase the number of workers if CPU usage is low enough."""
        if self.workers < self.max_workers:
            self.workers += 2
            self.executor._max_workers = self.workers
            self.logger.info(f"CPU usage is stable. Increasing workers to {self.workers}")
        else:
            self.logger.info("Worker count already at maximum")

    def start_cpu_monitoring(self):
        """Start monitoring CPU usage in a separate thread."""
        self.logger.info("Starting CPU monitoring thread")
        cpu_monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
        cpu_monitor_thread.daemon = True
        cpu_monitor_thread.start()

    def get_executor(self):
        """Return the executor instance to be used in other parts of the code."""
        return self.executor