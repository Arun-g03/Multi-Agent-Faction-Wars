from SHARED.core_imports import *
import os
import subprocess
import threading
import webbrowser
import time
import datetime
from collections import defaultdict
import UTILITIES.utils_config as utils_config
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    _instances = {}
    _default_run_name = None
    tensorboard_process = None

    def __new__(cls, log_dir="RUNTIME_LOGS/Tensorboard_logs", run_name=None):
        if not utils_config.ENABLE_TENSORBOARD:
            return None

        if run_name is None:
            run_name = cls.get_default_run_name()

        instance_key = f"{log_dir}_{run_name}"
        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            instance._init_writer(log_dir, run_name)
            cls._instances[instance_key] = instance

        return cls._instances[instance_key]

    @classmethod
    def set_default_run_name(cls, run_name=None):
        if run_name is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_DATE_{current_time[:8]}_TIME_{current_time[9:]}"
        cls._default_run_name = run_name
        print(f"[TensorBoard] Default run name set to: {run_name}")
        return run_name

    @classmethod
    def get_default_run_name(cls):
        if cls._default_run_name is None:
            cls.set_default_run_name()
        return cls._default_run_name

    @classmethod
    def reset(cls, new_run_name=None):
        for instance in cls._instances.values():
            try:
                instance.close()
            except Exception as e:
                print(f"[TensorBoard] Error closing logger: {e}")
        cls._instances.clear()
        cls.set_default_run_name(new_run_name)

    def _init_writer(self, log_dir, run_name):
        self.log_dir = log_dir
        self.run_name = run_name
        self.run_dir = os.path.join(self.log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        print(f"[TensorBoard] Logger initialized at: {self.run_dir}")

    def log_scalar(self, name, value, step):
        self._safe_log(lambda: self.writer.add_scalar(name, value, step))

    

    def log_distribution(self, name, values, step):
        """
        Log a distribution (e.g., weights, activations) to TensorBoard.
        Converts lists to numpy arrays automatically.
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            if isinstance(values, list):
                values = np.array(values)

            self.writer.add_histogram(name, values, step)
        except Exception as e:
            print(f"[TensorBoard] Error logging distribution '{name}': {e}")



    def log_image(self, name, image_tensor, step):
        """
        Log images to TensorBoard (e.g., visualizations, game frames).
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        if image_tensor is None:
            print(f"[TensorBoard WARNING] Skipping image log for '{name}' — image_tensor is None.")
            return
        try:
            self._safe_log(lambda: self.writer.add_image(name, image_tensor, step))
        except Exception as e:
            print(f"[TensorBoard] Error logging image '{name}': {e}")


    def log_text(self, name, text, step):
        self._safe_log(lambda: self.writer.add_text(name, text, step))

    

    def _safe_log(self, log_func):
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            log_func()
        except Exception as e:
            print(f"[TensorBoard] Logging error: {e}")

    def close(self):
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            self.writer.close()
        except Exception as e:
            print(f"[TensorBoard] Error closing writer: {e}")

    def run_tensorboard(self, log_dir="RUNTIME_LOGS/Tensorboard_logs", port=6006):
        if self.tensorboard_process:
            print("[TensorBoard] Already running.")
            return

        abs_log_path = os.path.abspath(log_dir)
        print(f"[TensorBoard] Launching at: {abs_log_path}")

        def _launch():
            try:
                self.tensorboard_process = subprocess.Popen(
                    ["tensorboard", f"--logdir={abs_log_path}", f"--port={port}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(2)
                webbrowser.open(f"http://localhost:6006/?darkMode=true#timeseries&reloadInterval=5")
            except Exception as e:
                print(f"[TensorBoard ERROR] Failed to launch: {e}")

        threading.Thread(target=_launch, daemon=True).start()

    def stop_tensorboard(self):
        if not self.tensorboard_process:
            print("\033[91m[TensorBoard] Stop called but was not running.\033[0m")
            return

        if self.tensorboard_process.poll() is None:
            print("\033[91m[TensorBoard] Shutting down...\033[0m")
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait(timeout=3)
            self.cleanup_event_files()
        else:
            print("\033[91m[TensorBoard] TensorBoard was already stopped.\033[0m")

    def cleanup_event_files(self, log_dir="RUNTIME_LOGS/Tensorboard_logs"):
        event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.")]
        for file in event_files:
            try:
                file_path = os.path.join(log_dir, file)
                if os.path.getsize(file_path) < 2048:
                    os.remove(file_path)
                    print(f"[TensorBoard] Deleted small event file: {file_path}")
            except Exception as e:
                print(f"[TensorBoard ERROR] Failed to delete {file_path}: {e}")
