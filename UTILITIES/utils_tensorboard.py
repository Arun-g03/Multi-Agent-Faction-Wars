from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config




class TensorBoardLogger:
    """
    TensorBoardLogger is a singleton class that initialises a TensorBoard SummaryWriter.
    It ensures that only one instance of the SummaryWriter is created and used across the application.
    """
    _instances = {}  # Dictionary to store multiple instances by run name
    tensorboard_process = None
    _default_run_name = None  # Store a default run name
    
    @classmethod
    def set_default_run_name(cls, run_name):
        """Set a default run name to be used when none is specified"""
        cls._default_run_name = run_name
        print(f"[TensorBoard] Default run name set to: {run_name}")
    
    @classmethod
    def get_default_run_name(cls):
        """Get the current default run name, creating one if needed"""
        if cls._default_run_name is None:
            # Create a timestamp-based default run name
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cls._default_run_name = f"run_{current_time}"
        return cls._default_run_name

    def __new__(cls, log_dir="RUNTIME_LOGS/Tensorboard_logs", run_name=None):
        if not utils_config.ENABLE_TENSORBOARD:
            return None
        
        # Use the default run name if none is provided
        if run_name is None:
            run_name = cls.get_default_run_name()
            
        # Create a unique key for this instance based on log_dir and run_name
        instance_key = f"{log_dir}_{run_name}"
        
        if instance_key not in cls._instances:
            try:
                instance = super(TensorBoardLogger, cls).__new__(cls)
                instance._init_writer(log_dir, run_name)
                cls._instances[instance_key] = instance
            except Exception as e:
                print(f"Error creating TensorBoardLogger instance: {str(e)}")
                return None
                
        return cls._instances[instance_key]

    def _init_writer(self, log_dir, run_name):
        try:
            self.log_dir = log_dir
            self.run_name = run_name
            
            # Create the run directory path
            self.run_dir = os.path.join(self.log_dir, run_name)
                
            # Create directory if it doesn't exist
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
                
            self.writer = SummaryWriter(log_dir=self.run_dir)
            
            print(f"[TensorBoard] Logger initialized for run: {run_name} at {self.run_dir}")
        except Exception as e:
            print(f"Error initialising TensorBoard writer: {str(e)}")



            

    def log_scalar(self, name, value, step):
        """
        Log scalar metrics (e.g., reward, loss).

        :param name: The name of the metric (e.g., "reward", "loss").
        :param value: The value of the metric.
        :param step: The step at which to log the metric.
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            self.writer.add_scalar(name, value, step)
        except Exception as e:
            print(f"Error logging scalar {name}: {str(e)}")

    def log_histogram(self, name, value, step):
        """
        Log histograms (e.g., weights, activations).

        :param name: The name of the metric (e.g., "weights").
        :param value: The value to log (usually a tensor or array).
        :param step: The step at which to log the metric.
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            self.writer.add_histogram(name, value, step)
        except Exception as e:
            print(f"Error logging histogram {name}: {str(e)}")

    def log_image(self, name, image_tensor, step):
        """
        Log images to TensorBoard (e.g., visualisations, game frames).

        :param name: The name of the image metric (e.g., "game_frame").
        :param image_tensor: The image tensor to log (typically in CHW format).
        :param step: The step at which to log the image.
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            self.writer.add_image(name, image_tensor, step)
        except Exception as e:
            print(f"Error logging image {name}: {str(e)}")

    def log_text(self, name, text, step):
        """
        Log text data to TensorBoard (e.g., logs, outputs).

        :param name: The name of the text metric (e.g., "episode_info").
        :param text: The text to log.
        :param step: The step at which to log the text.
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            self.writer.add_text(name, text, step)
        except Exception as e:
            print(f"Error logging text {name}: {str(e)}")

    def log_metrics(self, metrics_dict, step):
        """
        Log multiple metrics at once (e.g., reward, loss, accuracy).

        :param metrics_dict: A dictionary where the keys are metric names, and the values are the metric values.
        :param step: The step at which to log the metrics.
        """
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            for name, value in metrics_dict.items():
                self.log_scalar(name, value, step)
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

    def close(self):
        """Close the TensorBoard writer when you're done logging."""
        if not utils_config.ENABLE_TENSORBOARD:
            return
        try:
            self.writer.close()
        except Exception as e:
            print(f"Error closing TensorBoard writer: {str(e)}")

    def run_tensorboard(self, log_dir="RUNTIME_LOGS/Tensorboard_logs", port=6006):
        """
        Launch TensorBoard pointing at the specified log directory.
        Opens the browser automatically.
        """
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
                webbrowser.open(f"http://localhost:{port}")
            except Exception as e:
                print(f"[TensorBoard ERROR] Failed to launch: {e}")

        threading.Thread(target=_launch, daemon=True).start()

    def stop_tensorboard(self):
        """
        Stop the TensorBoard process and clean up event files smaller than 2KB.
        """
        if not self.tensorboard_process:
            print("\033[91m[TensorBoard] Stop called on TensorBoard but it was not started anyway.\033[0m")
            return

        if self.tensorboard_process.poll() is None:
            print("\033[91m[TensorBoard] Shutting down...\033[0m")
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait(timeout=3)

            # Clean up event files smaller than 2KB
            self.cleanup_event_files()
        else:
            print("\033[91m[TensorBoard] TensorBoard is not running.\033[0m")

    def cleanup_event_files(self, log_dir="RUNTIME_LOGS/Tensorboard_logs"):
        """
        Clean up event files smaller than 2KB.
        """
        event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.")]
        for file in event_files:
            file_path = os.path.join(log_dir, file)
            try:
                if os.path.getsize(file_path) < 2048:  # Check if the file is smaller than 2KB
                    os.remove(file_path)
                    print(f"[TensorBoard] Deleted small event file: {file_path}")
            except Exception as e:
                print(f"[TensorBoard ERROR] Failed to delete {file_path}: {e}")






































