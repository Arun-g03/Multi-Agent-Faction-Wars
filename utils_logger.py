import logging

from utils_config import LOGGING_ENABLED
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 # Enables/disables debug logging and visuals

class Logger:
    def __init__(self, log_file, log_level=logging.INFO, log_to_console=True, clear_logs=True):
        """
        Initialise the Logger class with mandatory log file and optional logging level.
        """
        # Create a unique logger instance
        self.logger = logging.getLogger(log_file)
    
        if not LOGGING_ENABLED:
            self.logger.addHandler(logging.NullHandler())
            print("DEBUG_MODE is disabled. Logger will not create log files.")
            return

        # Ensure the LOG directory exists
        log_directory = "LOG"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            print(f"Created directory: {log_directory}")

        # Full path for the log file
        self.log_file = os.path.join(log_directory, log_file)
        self.log_level = log_level

        # Overwrite the log file at initialisation if clear_logs is True
        if clear_logs:
            open(self.log_file, 'w').close()
            print(f"Logger initialised. Logs will be saved to: {os.path.abspath(self.log_file)}")

        self.logger.setLevel(self.log_level)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Clear existing handlers (prevents duplicate logs)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(file_handler)

        # Log initialisation
        self.logger.info("Logger initialised and log file cleared.")
        
    def log(self, message, level=logging.INFO):
        """
        Log a message at a specified logging level.
        """
        if not LOGGING_ENABLED:
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

    def debug_log(self, message, level=logging.DEBUG):
        """
        Log a debug message if DEBUG_MODE is enabled.
        """
        if LOGGING_ENABLED:
            self.log(message, level)

import torch
from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardLogger:
    def __init__(self, log_dir="log"):
        """
        Initialises the TensorBoard logger and sets up the directory for logs.
        
        :param log_dir: The directory to store the TensorBoard logs.
        """
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create a SummaryWriter to log data for TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, name, value, step):
        """
        Log scalar metrics (e.g., reward, loss).
        
        :param name: The name of the metric (e.g., "reward", "loss").
        :param value: The value of the metric.
        :param step: The step at which to log the metric.
        """
        self.writer.add_scalar(name, value, step)

    def log_histogram(self, name, value, step):
        """
        Log histograms (e.g., weights, activations).
        
        :param name: The name of the metric (e.g., "weights").
        :param value: The value to log (usually a tensor or array).
        :param step: The step at which to log the metric.
        """
        self.writer.add_histogram(name, value, step)

    def log_image(self, name, image_tensor, step):
        """
        Log images to TensorBoard (e.g., visualisations, game frames).
        
        :param name: The name of the image metric (e.g., "game_frame").
        :param image_tensor: The image tensor to log (typically in CHW format).
        :param step: The step at which to log the image.
        """
        self.writer.add_image(name, image_tensor, step)

    def log_text(self, name, text, step):
        """
        Log text data to TensorBoard (e.g., logs, outputs).
        
        :param name: The name of the text metric (e.g., "episode_info").
        :param text: The text to log.
        :param step: The step at which to log the text.
        """
        self.writer.add_text(name, text, step)

    def log_metrics(self, metrics_dict, step):
        """
        Log multiple metrics at once (e.g., reward, loss, accuracy).
        
        :param metrics_dict: A dictionary where the keys are metric names, and the values are the metric values.
        :param step: The step at which to log the metrics.
        """
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, step)

    def close(self):
        """Close the TensorBoard writer when you're done logging."""
        self.writer.close()




class MatplotlibPlotter:
    def __init__(self, file_path="simulation_metrics.csv"):
        """
        Initialises the MatplotlibPlotter class with the given file path for metrics.

        :param file_path: Path to the CSV file containing the metrics. Defaults to 'simulation_metrics.csv'.
        """
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            print(f"Metrics file '{self.file_path}' not found. Please run the simulation first.")

    def plot_metrics(self):
        """
        Reads the metrics from the CSV file and generates a plot.
        """
        data = pd.read_csv(self.file_path)
        plt.figure(figsize=(12, 6))

        plt.plot(data["time"], data["resources_collected"], label="Resources Collected", colour="green")
        plt.plot(data["time"], data["threats_eliminated"], label="Threats Eliminated", colour="red")
        plt.plot(data["time"], data["average_health"], label="Average Health", colour="blue")

        plt.xlabel("Time")
        plt.ylabel("Metrics")
        plt.title("Simulation Metrics Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_single_metric(self, metric_name):
        """
        Plot a single metric over time.

        :param metric_name: The name of the metric to plot (e.g., 'resources_collected').
        """
        if metric_name not in ['resources_collected', 'threats_eliminated', 'average_health']:
            print(f"Invalid metric name: {metric_name}. Valid options are 'resources_collected', 'threats_eliminated', or 'average_health'.")
            return

        data = pd.read_csv(self.file_path)
        plt.figure(figsize=(12, 6))
        plt.plot(data["time"], data[metric_name], label=metric_name.replace("_", " ").title())

        plt.xlabel("Time")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"{metric_name.replace('_', ' ').title()} Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_heatmap(self, matrix_data):
        """
        Plot a heatmap (e.g., confusion matrix, correlation matrix).

        :param matrix_data: A 2D array or DataFrame to plot as a heatmap.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5, cbar=True)
        plt.title("Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.show()

    def plot_correlation_matrix(self, data):
        """
        Plot a correlation matrix for the given dataset.

        :param data: The dataset as a pandas DataFrame.
        """
        corr = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=.5)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_boxplot(self, data, column):
        """
        Plot a box plot for a single column to detect outliers and distribution.

        :param data: The dataset as a pandas DataFrame.
        :param column: The column name to plot.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plt.show()

    def plot_histogram(self, data, column, bins=30):
        """
        Plot a histogram for a single column to show the distribution of values.

        :param data: The dataset as a pandas DataFrame.
        :param column: The column name to plot.
        :param bins: Number of bins to use in the histogram.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], bins=bins, kde=True, colour="skyblue")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_pairplot(self, data):
        """
        Plot a pairplot for a given dataset to visualise pairwise relationships.

        :param data: The dataset as a pandas DataFrame.
        """
        sns.pairplot(data)
        plt.suptitle("Pairplot of Features", y=1.02)
        plt.show()

    def plot_violinplot(self, data, column):
        """
        Plot a violin plot to show the distribution of a variable.

        :param data: The dataset as a pandas DataFrame.
        :param column: The column name to plot.
        """
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=data[column])
        plt.title(f"Violin Plot of {column}")
        plt.show()

    def save_plot_as_image(self, image_name="plot.png"):
        """
        Save the current plot as an image file.

        :param image_name: Name of the file to save the plot as (e.g., 'plot.png').
        """
        plt.savefig(image_name, format='png')
        print(f"Plot saved as {image_name}")

    def send_to_tensorboard(self, image_name, tensorboard_logger):
        """
        Send the saved plot to TensorBoard as an image.

        :param image_name: Name of the image file to send.
        :param tensorboard_logger: TensorBoardLogger instance.
        """
        image = plt.imread(image_name)
        tensorboard_logger.log_image(image_name, image, step=0)
        print(f"Sent {image_name} to TensorBoard.")