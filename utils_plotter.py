import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

# File to store metrics
METRICS_FILE = "simulation_metrics.csv"

# Initialise the CSV file for metrics
def Initialise_metrics_file():
    """Creates a CSV file for logging metrics if it doesn't already exist."""
    if not os.path.exists(METRICS_FILE):  # Avoid overwriting if the file exists
        with open(METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "resources_collected", "threats_eliminated", "average_health"])  # Header

# Log metrics during the simulation
def log_metrics(time, resources_collected, threats_eliminated, average_health):
    """
    Appends a row of metrics to the CSV file.

    Args:
        time (int): The simulation time step.
        resources_collected (int): Total resources collected.
        threats_eliminated (int): Total threats eliminated.
        average_health (float): Average health of agents.
    """
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time, resources_collected, threats_eliminated, average_health])

# Plot metrics after simulation
def plot_metrics(file_path=METRICS_FILE):
    """
    Reads the metrics from the CSV file and generates plots.

    Args:
        file_path (str): Path to the CSV file containing the metrics.
    """
    if not os.path.exists(file_path):
        print(f"Metrics file '{file_path}' not found. Please run the simulation first.")
        return

    # Load data
    data = pd.read_csv(file_path)

    # Generate plots
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

# Standalone functionality
if __name__ == "__main__":
    plot_metrics()
