from SHARED.core_imports import *

matplotlib.use(
    "Agg"
)  # Disable the need for an interactive plots since system saves them as images
import matplotlib.pyplot as plt

import seaborn as sns
import io
from PIL import Image
import torchvision.transforms as transforms
import glob
import pandas as pd
import numpy as np
import os
import csv
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import UTILITIES.utils_config as utils_config


class MatplotlibPlotter:
    _instance = None

    def __new__(cls, image_dir=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if not hasattr(cls._instance, "image_dir"):
            cls._instance.image_dir = image_dir
        if not hasattr(cls._instance, "episode_data"):
            cls._instance.episode_data = defaultdict(list)
        if not hasattr(cls._instance, "plot_types"):
            cls._instance.plot_types = {}  # new: name -> "scalar" or "heatmap"
        return cls._instance

    # === Core Utilities ===

    def _plot_to_tensor(self, fig):
        if fig is None:
            print("[Plotter] WARNING: Received None figure.")
            return None
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            image = Image.open(buf).convert("RGB")
            image_tensor = transforms.ToTensor()(image)

            buf.close()
            plt.close(fig)
            return image_tensor
        except Exception as e:
            print(f"[Plotter] Failed to convert figure to tensor: {e}")
            return None

    def update_image_plot(self, name, fig=None, tensorboard_logger=None, step=0):
        # Skip if plots are disabled
        if not utils_config.ENABLE_PLOTS:
            return

        if fig is None:
            fig = plt.gcf()

        save_dir = self.image_dir
        if (
            tensorboard_logger
            and hasattr(tensorboard_logger, "run_dir")
            and utils_config.ENABLE_TENSORBOARD
        ):
            save_dir = tensorboard_logger.run_dir

        # Fallback to default directory if both are None
        if save_dir is None:
            save_dir = os.path.join("VISUALS", "PLOTS")

        os.makedirs(save_dir, exist_ok=True)

        image_path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(image_path)
        print(f"[Plotter] Saved plot to {image_path}")

        if tensorboard_logger and utils_config.ENABLE_TENSORBOARD:
            image_tensor = self._plot_to_tensor(fig)
            if image_tensor is not None:
                tensorboard_logger.log_image(name, image_tensor, step)
                print(f"[Plotter] Sent '{name}' to TensorBoard.")

    # === Data Collection ===

    def add_episode_matrix(
        self,
        name,
        matrix,
        step=0,
        episode=None,
        extra_data=None,
        plot_type="heatmap",
        keys=None,
    ):
        """
        Add matrix + optional metadata (extra_data) and custom keys (for X-axis or other uses) for this episode.
        """
        # If custom keys are provided, ensure they match the shape of the matrix
        if keys is not None and len(keys) != matrix.shape[1]:
            print(
                f"[ERROR] The number of keys does not match the number of columns in the matrix."
            )
            return

        entry = {
            "matrix": np.array(matrix),
            "step": step,
            "episode": episode,
            "extra_data": extra_data or {},
            "keys": keys if keys is not None else [],
        }

        # Store the matrix and its metadata
        self.episode_data[name].append(entry)
        self.plot_types[name] = plot_type  # track intended plot type

    def flush_episode_plots(
        self, tensorboard_logger=None, save_data=True, save_as="npy"
    ):
        for name, entries in self.episode_data.items():
            try:
                matrices = [e["matrix"] for e in entries]
                step = max(e["step"] for e in entries)
                episode = max(e["episode"] for e in entries if e["episode"] is not None)

                stacked = np.vstack(matrices)
                summary = np.sum(stacked, axis=0, keepdims=True)

                plot_type = self.plot_types.get(name, "heatmap")
                keys = entries[0].get(
                    "keys", None
                )  # Get keys from the first entry (if present)

                if plot_type == "scalar":
                    self.plot_scalar_summary(
                        value=summary[0][0],
                        name=name,
                        step=step,
                        episode=episode,
                        tensorboard_logger=tensorboard_logger,
                    )
                else:
                    role_name = name.replace("_actions", "").replace(
                        "_task_distribution", ""
                    )

                    # Determine title and labels based on the plot name
                    if "task_distribution" in name:
                        label_source = list(utils_config.TASK_TYPE_MAPPING.keys())
                        title = f"{role_name.title()} Task Summary"
                    else:
                        label_source = utils_config.ROLE_ACTIONS_MAP.get(
                            role_name, [str(i) for i in range(summary.shape[1])]
                        )
                        title = f"{role_name.title()} Action Summary"

                    action_labels = label_source[: summary.shape[1]]
                    df = pd.DataFrame(summary, columns=action_labels)

                    # Plot heatmap with custom keys for the x-axis (if available)
                    self.plot_heatmap(
                        data=df,
                        name=name,
                        title=title,
                        tensorboard_logger=tensorboard_logger,
                        step=step,
                        episode=episode,
                        save_data=save_data,
                        save_as=save_as,
                        xticks=keys,  # Pass keys for X-axis labels if available
                    )

            except Exception as e:
                print(f"[Plotter] Failed to summarize {name}: {e}")

        self.episode_data.clear()

    # === Plot Types ===

    def plot_heatmap(
        self,
        data,
        name="heatmap",
        title=None,
        tensorboard_logger=None,
        step=0,
        episode=None,
        save_data=True,
        save_as="csv",
        xticks=None,
    ):
        # Skip if plots are disabled
        if not utils_config.ENABLE_PLOTS:
            return
        """
        Plot a heatmap with custom X-axis labels (xticks).
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        fmt = ".2f" if data.values.dtype.kind == "f" else "d"

        if isinstance(data, pd.DataFrame):
            sns.heatmap(data, annot=True, fmt=fmt, cmap="viridis", ax=ax, cbar=True)
            # Set custom xticks if provided
            if xticks:
                ax.set_xticklabels(xticks, rotation=25, ha="right")
            else:
                ax.set_xticklabels(data.columns, rotation=25, ha="right")
        else:
            sns.heatmap(data, annot=True, fmt=fmt, cmap="viridis", ax=ax, cbar=True)
            # Set custom xticks if provided
            if xticks:
                ax.set_xticklabels(xticks, rotation=25, ha="right")

        ax.set_yticks([0])
        ax.set_yticklabels(["Summary"])
        ax.set_xlabel("Action")

        title = title or name.replace("_", " ").title()
        if episode is not None:
            title += f" (Episode {episode})"
        ax.set_title(title)

        file_name = f"{name}_ep{episode}" if episode is not None else name
        save_dir = self.image_dir
        if tensorboard_logger and hasattr(tensorboard_logger, "run_dir"):
            save_dir = (
                tensorboard_logger.run_dir
            )  # Use the directory managed by TensorBoard

        # Fallback to default directory if both are None
        if save_dir is None:
            save_dir = os.path.join("VISUALS", "PLOTS")

        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        self.update_image_plot(
            name=file_name, fig=fig, tensorboard_logger=tensorboard_logger, step=step
        )

        # Save data as CSV alongside the image plot
        if save_data:
            try:
                # Save CSV in the same directory as the plot
                csv_filename = os.path.join(save_dir, f"{name}_Episode_{episode}.csv")

                # Assuming `data` is a DataFrame, you can save it directly as CSV
                data.to_csv(csv_filename, index=False)
                print(f"[Plotter] CSV data saved to {csv_filename}")
            except Exception as e:
                print(f"[Plotter] Failed to write CSV for {name}: {e}")

    def plot_victory_timeline(
        self, episodes, winner_ids, victory_types, tensorboard_logger=None
    ):
        # Skip if plots are disabled
        if not utils_config.ENABLE_PLOTS:
            return
        """
        Plots a per-episode timeline showing which faction won and how (e.g., resource, elimination),
        and saves a corresponding CSV file.
        """
        if not episodes or not winner_ids or not victory_types:
            print("[Plotter] Victory data is empty — skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        color_map = {
            "resource": "green",
            "elimination": "blue",
            "timeout": "orange",
            "none": "grey",
        }

        labels = []
        colors = []
        for wid, vtype in zip(winner_ids, victory_types):
            if wid == -1:
                labels.append("No Winner")
            else:
                labels.append(f"Faction {wid} ({vtype})")
            colors.append(color_map.get(vtype, "black"))

        ax.bar(episodes, [1] * len(episodes), color=colors)

        for i, label in enumerate(labels):
            ax.text(
                episodes[i],
                0.5,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                weight="bold",
            )

        ax.set_title("Victory Timeline Per Episode")
        ax.set_xlabel("Episode")
        ax.set_yticks([])
        ax.set_xlim(left=0)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        self.update_image_plot(
            name="victory_timeline",
            fig=fig,
            tensorboard_logger=tensorboard_logger,
            step=episodes[-1],
        )

        # === Save CSV ===
        save_dir = self.image_dir
        if tensorboard_logger and hasattr(tensorboard_logger, "run_dir"):
            save_dir = tensorboard_logger.run_dir
        elif not save_dir:
            save_dir = os.path.join("VISUALS", "PLOTS")

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "victory_timeline.csv")

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Winner ID", "Victory Type", "Label"])
                for ep, wid, vtype, label in zip(
                    episodes, winner_ids, victory_types, labels
                ):
                    writer.writerow([ep, wid, vtype, label])
            print(f"[Plotter] Victory timeline CSV saved to {csv_path}")
        except Exception as e:
            print(f"[Plotter] Failed to write CSV for victory timeline: {e}")

    def plot_scalar(self, name, value, step=0, episode=None):
        matrix = np.array([[value]])
        self.add_episode_matrix(
            name, matrix, step=step, episode=episode, plot_type="scalar"
        )

    def plot_scalar_summary(self, value, name, step, episode, tensorboard_logger=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["value"], [value])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name} (Episode {episode})")
        ax.set_ylabel("Value")

        self.update_image_plot(
            name=name, fig=fig, tensorboard_logger=tensorboard_logger, step=step
        )

    def plot_clustered_stacked_bar_chart(
        self,
        task_types,
        success_counts,
        failure_counts,
        ongoing_counts,
        episodes,
        tensorboard_logger=None,
        step=0,
        name=None,
    ):
        """
        Plots a clustered stacked bar chart to represent task success, failure, and ongoing counts over a single episode for each faction.
        Each cluster represents one task type with bars for the outcomes: Success, Failure, and Ongoing.

        Parameters:
        - task_types: List of task types (e.g., 'gather', 'explore', etc.).
        - success_counts: List of success counts for each task type in the episode.
        - failure_counts: List of failure counts for each task type in the episode.
        - ongoing_counts: List of ongoing counts for each task type in the episode.
        - episodes: List of episode numbers.
        - tensorboard_logger: TensorBoard logger instance (optional).
        - step: Current step (for TensorBoard).
        - name: Name of the plot (e.g., "Faction_1_Task_Timeline").
        """

        num_task_types = len(task_types)

        # Prepare the x positions for each task type (each task type will have a single bar for each episode)
        x = np.arange(num_task_types)  # X locations for the task types
        width = 0.5  # Width of each bar

        fig, ax = plt.subplots(figsize=(12, 7))

        # Loop through each task and plot stacked bars for Success, Failure, and Ongoing
        for i, task_type in enumerate(task_types):
            ax.bar(
                x[i], success_counts[i], width, label="Success", color="#4CAF50"
            )  # Green for success
            ax.bar(
                x[i],
                failure_counts[i],
                width,
                label="Failure",
                color="#F44336",
                bottom=success_counts[i],
            )  # Red for failure
            ax.bar(
                x[i],
                ongoing_counts[i],
                width,
                label="Ongoing",
                color="#FFC107",
                bottom=np.array(success_counts[i]) + np.array(failure_counts[i]),
            )  # Yellow for ongoing

        # Labeling the x-axis and adding necessary plot details
        ax.set_xticks(x)
        ax.set_xticklabels(task_types, rotation=45, ha="right")
        ax.set_xlabel("Task Type")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Task Success vs Failure vs Ongoing Counts for {name} in Episode {episodes[0]}"
        )

        # Adjust the legend to display only color labels for Success, Failure, and Ongoing
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ["Success", "Failure", "Ongoing"]  # New legend labels
        ax.legend(
            handles=handles[:3],
            labels=new_labels,
            title="Task Status",
            loc="upper left",
            ncol=3,
        )

        # Display grid for clarity
        ax.grid(True, linestyle="--", alpha=0.7)

        # Update and save the plot with the provided name
        if name:
            self.update_image_plot(
                name=name, fig=fig, tensorboard_logger=tensorboard_logger, step=step
            )

        plt.tight_layout()

        # Save data as CSV alongside the image plot
        if name:
            # Use tensorboard_logger's run_dir if available, else fallback to self.image_dir
            save_dir = self.image_dir
            if tensorboard_logger and hasattr(tensorboard_logger, "run_dir"):
                save_dir = (
                    tensorboard_logger.run_dir
                )  # Use TensorBoard's directory if available

            # Fallback to default directory if both are None
            if save_dir is None:
                save_dir = os.path.join("VISUALS", "PLOTS")

            os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

            csv_filename = os.path.join(save_dir, f"{name}_Episode_{episodes[0]}.csv")

            try:
                with open(csv_filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Task Type", "Success", "Failure", "Ongoing"])
                    for i in range(len(task_types)):
                        writer.writerow(
                            [
                                task_types[i],
                                success_counts[i],
                                failure_counts[i],
                                ongoing_counts[i],
                            ]
                        )
                print(f"[Plotter] CSV data saved to {csv_filename}")
            except Exception as e:
                print(f"[Plotter] Failed to write CSV for {name}: {e}")

    def plot_scalar_over_time(
        self, names, values_list, episodes, tensorboard_logger=None
    ):
        # Skip if plots are disabled
        if not utils_config.ENABLE_PLOTS:
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        # Pad value lists to match length of episodes
        max_length = len(episodes)
        for idx in range(len(values_list)):
            if len(values_list[idx]) < max_length:
                values_list[idx] += [0] * (max_length - len(values_list[idx]))

        # Ensure values_list and names match in count
        if len(values_list) < len(names):
            values_list += [[] for _ in range(len(names) - len(values_list))]
        elif len(values_list) > len(names):
            values_list = values_list[: len(names)]

        # Plot each line
        for name, values in zip(names, values_list):
            if len(values) != len(episodes):
                print(
                    f"[Plotter] Skipping '{name}' — length mismatch (episodes: {len(episodes)}, values: {len(values)})"
                )
                continue  # skip broken/empty data

            ax.plot(episodes, values, marker="o", label=name)
            final_avg = sum(values) / len(values) if values else 0
            ax.text(
                episodes[-1],
                final_avg,
                f"Avg: {final_avg:.2f}",
                color="black",
                backgroundcolor="grey",
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

            final_avg = sum(values) / len(values) if values else 0
            ax.text(
                episodes[-1],
                final_avg,
                f"Avg: {final_avg:.2f}",
                color="black",
                backgroundcolor="grey",
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

        clean_names = [name.replace("_", " ") for name in names]
        ax.set_title(f"{names} over {episodes[-1]} episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(title="Metrics")

        self.update_image_plot(
            name="_".join(clean_names) + "_trend",
            fig=fig,
            tensorboard_logger=tensorboard_logger,
            step=episodes[-1],
        )

        # Save CSV
        save_dir = self.image_dir
        if tensorboard_logger and hasattr(tensorboard_logger, "run_dir"):
            save_dir = tensorboard_logger.run_dir

        # Fallback to default directory if both are None
        if save_dir is None:
            save_dir = os.path.join("VISUALS", "PLOTS")

        os.makedirs(save_dir, exist_ok=True)

        csv_name = "_".join(clean_names) + "_trend.csv"
        csv_path = os.path.join(save_dir, csv_name)

        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Episode"] + clean_names)

                for i, ep in enumerate(episodes):
                    row = [ep]
                    for j in range(len(names)):
                        if len(values_list[j]) > i:
                            row.append(values_list[j][i])
                        else:
                            row.append(0)
                    writer.writerow(row)

            print(f"[Plotter] Scalar trend data saved to {csv_path}")
        except Exception as e:
            print(f"[Plotter] Failed to write CSV for scalar trend: {e}")

    # === Saved Plots Utility ===

    def plot_saved_plots(
        self, run_dir=None, pattern="*_heatmap_ep*.npy", resend_to_tensorboard=True
    ):
        """
        Reload saved plot data (like heatmaps) and optionally re-log them to TensorBoard.
        """
        run_dir = run_dir or self.image_dir
        files = glob.glob(os.path.join(run_dir, pattern))

        if not files:
            print("[Plotter] No saved plots found.")
            return

        for path in files:
            try:
                name = os.path.splitext(os.path.basename(path))[0]
                if path.endswith(".npy"):
                    matrix = np.load(path)
                elif path.endswith(".csv"):
                    matrix = pd.read_csv(path).values
                else:
                    print(f"[Plotter] Unsupported file format: {path}")
                    continue

                print(f"[Plotter] Re-plotting from {path}")
                self.plot_heatmap(
                    data=matrix,
                    name=name,
                    title=name.replace("_", " ").title(),
                    tensorboard_logger=(
                        TensorBoardLogger() if resend_to_tensorboard else None
                    ),
                    step=0,
                    episode=None,
                )
            except Exception as e:
                print(f"[Plotter] Failed to reload plot from {path}: {e}")
