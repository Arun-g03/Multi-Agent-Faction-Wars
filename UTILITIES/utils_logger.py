from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config


class Logger:
    def __init__(
        self, log_file, log_level=logging.INFO, log_to_console=False, clear_logs=True
    ):

        self.enabled = getattr(
            utils_config, "ENABLE_LOGGING"
        )  # Default safe fallback: True

        if not self.enabled:
            # Logging is disabled globally: create a dummy logger
            self.logger = None
            print("[Logger] Logging disabled by config.")
            return

        # === Real logger setup ===
        self.logger = logging.getLogger(log_file)

        # Clear old handlers if re-instantiating
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Ensure the LOG directories exist (separate folders for general and error logs)
        general_log_dir = "RUNTIME_LOGS/General_Logs"
        error_log_dir = "RUNTIME_LOGS/Error_Logs"
        os.makedirs(general_log_dir, exist_ok=True)
        os.makedirs(error_log_dir, exist_ok=True)

        self.log_file = os.path.join(general_log_dir, log_file)
        # Create error log file name in Error_Logs folder
        error_log_file = os.path.join(error_log_dir, log_file)
        self.log_level = log_level

        if clear_logs:
            open(self.log_file, "w").close()
            open(error_log_file, "w").close()
            print(
                f"Logger initialised. General logs will be saved to: {os.path.abspath(self.log_file)}"
            )
            print(f"Error logs will be saved to: {os.path.abspath(error_log_file)}")

        self.logger.setLevel(self.log_level)

        # Regular file handler for all logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Dedicated error file handler for ERROR and CRITICAL logs only
        error_file_handler = logging.FileHandler(error_log_file)
        error_file_handler.setLevel(logging.ERROR)  # Only capture ERROR and CRITICAL
        error_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        error_file_handler.setFormatter(error_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Logger initialised and log file cleared.")

    def log_msg(self, message, level=logging.INFO):
        """
        Log a message at a specified logging level.
        """
        try:
            if not utils_config.ENABLE_LOGGING:
                return

            if level == logging.DEBUG:
                self.logger.debug(message)
            elif level == logging.INFO:
                self.logger.info(message)
            elif level == logging.WARNING:
                self.logger.warning(message)
            elif level == logging.ERROR:
                self.logger.error(message)
            elif level == logging.CRITICAL:
                self.logger.critical(message)
        except Exception as e:
            raise Exception(f"Logging error: {str(e)}")
