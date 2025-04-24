from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config


class MatplotlibPlotter:
    def __init__(self, file_path="simulation_metrics.csv"):
        """
        Initialises the MatplotlibPlotter class with the given file path for metrics.

        :param file_path: Path to the CSV file containing the metrics. Defaults to 'simulation_metrics.csv'.
        """
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            print(
                f"Metrics file '{self.file_path}' not found. Please run the simulation first.")

    def plot_metrics(self):
        """
        Reads the metrics from the CSV file and generates a plot.
        """
        data = pd.read_csv(self.file_path)
        plt.figure(figsize=(12, 6))

        plt.plot(data["time"], data["resources_collected"],
                 label="Resources Collected", colour="green")
        plt.plot(data["time"], data["threats_eliminated"],
                 label="Threats Eliminated", colour="red")
        plt.plot(data["time"], data["average_health"],
                 label="Average Health", colour="blue")

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
        if metric_name not in [
            'resources_collected',
            'threats_eliminated',
                'average_health']:
            print(
                f"Invalid metric name: {metric_name}. Valid options are 'resources_collected', 'threats_eliminated', or 'average_health'.")
            return

        data = pd.read_csv(self.file_path)
        plt.figure(figsize=(12, 6))
        plt.plot(data["time"], data[metric_name],
                 label=metric_name.replace("_", " ").title())

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
        sns.heatmap(matrix_data, annot=True, cmap="YlGnBu",
                    fmt=".2f", linewidths=.5, cbar=True)
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
        TensorBoardLogger().log_image(image_name, image, step=0)
        print(f"Sent {image_name} to TensorBoard.")
