import time
import threading
import csv
import os
import logging

logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self, device_idx=0, interval=1.0, csv_path=None):
        """
        Monitors GPU metrics: Usage %, Memory used (MB), and Power Draw (Watts).

        :param device_idx: Physical index of the GPU to monitor.
        :param interval: Measurement interval in seconds (default: 1.0).
        :param csv_path: File path to save the historical CSV log.
        """
        self.device_idx = device_idx
        self.interval = interval
        self.csv_path = csv_path
        self.history = []
        self.running = False
        self.thread = None
        self._nvml_initialized = False

        # Attempt to import and initialize NVML
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info(f"Successfully initialized NVML for GPU monitoring (Device: {device_idx})")
        except Exception as e:
            logger.warning(f"Could not initialize NVML: {e}. GPU monitoring will run in dummy mode.")
            self.pynvml = None

    def start(self):
        """Starts the background monitoring thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the background monitoring thread and shuts down NVML."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self._nvml_initialized and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def _monitor_loop(self):
        if self.csv_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
                with open(self.csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'gpu_usage_pct', 'memory_used_mb', 'power_draw_watts'])
            except Exception as e:
                logger.warning(f"Failed to initialize CSV log: {e}")

        while self.running:
            t_start = time.time()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            gpu_usage = 0.0
            mem_used_mb = 0.0
            power_watts = 0.0

            if self._nvml_initialized and self.pynvml:
                try:
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
                    
                    # GPU Usage %
                    try:
                        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_usage = float(util.gpu)
                    except Exception:
                        pass
                    
                    # Memory MB
                    try:
                        mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used_mb = float(mem.used) / (1024.0 * 1024.0)
                    except Exception:
                        pass
                    
                    # Power Draw (Watts)
                    try:
                        power_watts = float(self.pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Error querying GPU metrics: {e}")
            else:
                # Fallback / Dummy values if no GPU/NVML is available
                gpu_usage = 0.0
                mem_used_mb = 0.0
                power_watts = 0.0

            row = [timestamp, gpu_usage, mem_used_mb, power_watts]
            self.history.append(row)

            if self.csv_path:
                try:
                    with open(self.csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                except Exception as e:
                    logger.warning(f"Failed to append to GPU log: {e}")

            # Sleep to maintain 1-second interval
            t_elapsed = time.time() - t_start
            sleep_time = max(0.0, self.interval - t_elapsed)
            time.sleep(sleep_time)

    def get_history(self):
        """Returns the accumulated history as list of rows."""
        return self.history
