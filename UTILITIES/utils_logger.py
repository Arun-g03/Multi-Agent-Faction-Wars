from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config



class Logger:
    """
    Custom logging class to handle logging to a file and optionally to the console throughout the system.
    
    Usage:
        # Create a logger instance
        logger = Logger("my_log.log", log_to_console=True)
        
        # Log messages at different levels
        logger.log_msg("Info message")  # Default INFO level
        logger.log_msg("Debug message", level=logging.DEBUG)
        logger.log_msg("Warning message", level=logging.WARNING)
        logger.log_msg("Error message", level=logging.ERROR)
        logger.log_msg("Critical message", level=logging.CRITICAL)
        
    Args:
        log_file (str): Name of the log file
        log_level (int): Logging level (default: logging.INFO)
        log_to_console (bool): Whether to also log to console (default: False)
        clear_logs (bool): Whether to clear existing logs on initialization (default: True)
    """
    def __init__(
            self,
            log_file,
            log_level=logging.INFO,
            log_to_console=False,
            clear_logs=True):
        """
        Initialise the Logger class with mandatory log file and optional logging level.
        """
        # Create a unique logger instance
        self.logger = logging.getLogger(log_file)

        # Ensure the LOG directory exists
        log_directory = "RUNTIME_LOGS/LOGS"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            print(f"Created directory: {log_directory}")

        # Full path for the log file
        self.log_file = os.path.join(log_directory, log_file)
        self.log_level = log_level

        # Overwrite the log file at initialisation if clear_logs is True
        if clear_logs:
            open(self.log_file, 'w').close()
            print(
                f"Logger initialised. Logs will be saved to: {os.path.abspath(self.log_file)}")

        self.logger.setLevel(self.log_level)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                "%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Clear existing handlers (prevents duplicate logs)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(file_handler)

        # Log initialisation
        self.logger.info("Logger initialised and log file cleared.")

    def log_msg(self, message, level=logging.INFO):
        """
        Log a message at a specified logging level.
        """
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

