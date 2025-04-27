from SHARED.core_imports import *
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import torchvision.transforms as transforms
import glob
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import UTILITIES.utils_config as utils_config



class MatplotlibPlotter:
    _instance = None

    def __new__(cls, image_dir=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if not hasattr(cls._instance, "image_dir"):
            cls._instance.image_dir = image_dir or "RUNTIME_LOGS/plots"
        if not hasattr(cls._instance, "episode_data"):
            cls._instance.episode_data = defaultdict(list)
        return cls._instance

    # === Core Utilities ===

    def _plot_to_tensor(self, fig):
        if fig is None:
            print("[Plotter] WARNING: Received None figure.")
            return None
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
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
        if fig is None:
            fig = plt.gcf()

        save_dir = self.image_dir
        if tensorboard_logger and hasattr(tensorboard_logger, "run_dir") and utils_config.ENABLE_TENSORBOARD:
            save_dir = tensorboard_logger.run_dir
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

    def add_episode_matrix(self, name, matrix, step=0, episode=None, extra_data=None):
        """
        Add matrix + optional metadata (extra_data) for this episode.
        """
        entry = {
            "matrix": np.array(matrix),
            "step": step,
            "episode": episode,
            "extra_data": extra_data or {}
        }
        self.episode_data[name].append(entry)

    def flush_episode_plots(self, tensorboard_logger=None, save_data=True, save_as="npy"):
        for name, entries in self.episode_data.items():
            try:
                matrices = [e["matrix"] for e in entries]
                step = max(e["step"] for e in entries)
                episode = max(e["episode"] for e in entries if e["episode"] is not None)

                stacked = np.vstack(matrices)
                summary = np.sum(stacked, axis=0, keepdims=True)

                # Infer plot type
                role_name = name.replace("_actions", "").replace("_task_distribution", "")
                if "task_distribution" in name:
                    label_source = list(utils_config.TASK_TYPE_MAPPING.keys())
                    title = f"{role_name.title()} Task Summary"
                else:
                    label_source = utils_config.ROLE_ACTIONS_MAP.get(role_name, [str(i) for i in range(summary.shape[1])])
                    title = f"{role_name.title()} Action Summary"

                action_labels = label_source[:summary.shape[1]]
                df = pd.DataFrame(summary, columns=action_labels)

                self.plot_heatmap(
                    data=df,
                    name=name,
                    title=title,
                    tensorboard_logger=tensorboard_logger,
                    step=step,
                    episode=episode,
                    save_data=save_data,
                    save_as=save_as
                )

            except Exception as e:
                print(f"[Plotter] Failed to summarize {name}: {e}")

        self.episode_data.clear()

    # === Plot Types ===

    def plot_heatmap(self, data, name="heatmap", title=None,
                     tensorboard_logger=None, step=0, episode=None,
                     save_data=True, save_as="npy"):
        fig, ax = plt.subplots(figsize=(12, 6))

        if isinstance(data, pd.DataFrame):
            sns.heatmap(data, annot=True, fmt="d", cmap="viridis", ax=ax, cbar=True)
            ax.set_xticklabels(data.columns, rotation=25, ha='right')
        else:
            sns.heatmap(data, annot=True, fmt="d", cmap="viridis", ax=ax, cbar=True)

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
            save_dir = tensorboard_logger.run_dir
        os.makedirs(save_dir, exist_ok=True)

        self.update_image_plot(name=file_name, fig=fig, tensorboard_logger=tensorboard_logger, step=step)

        if save_data:
            try:
                data_path = os.path.join(save_dir, f"{file_name}.{save_as}")
                if save_as == "npy":
                    np.save(data_path, data)
                elif save_as == "csv":
                    pd.DataFrame(data).to_csv(data_path, index=False)
                print(f"[Plotter] Saved data to {data_path}")
            except Exception as e:
                print(f"[Plotter] Failed to save heatmap data: {e}")

    def plot_task_timeline(self, task_records, name="task_timeline", tensorboard_logger=None, step=0):
        """
        Summarized task timeline: success, failure, ongoing counts and average completion time per task type.
        """
        if not task_records:
            print("[Plotter] No task records provided.")
            return

        # Aggregation
        task_summary = {}

        for task_id, task_info in task_records.items():
            task_label = task_info.get("task_id", "unknown")

            if "Mine" in task_label:
                category = "mine"
            elif "Forage" in task_label:
                category = "forage"
            elif "Explore" in task_label:
                category = "explore"
            elif "DefendHQ" in task_label:
                category = "defend"
            else:
                category = task_info.get("type", "unknown")  # fallback to base type

            start = task_info.get("start_step", 0)
            end = task_info.get("end_step", step)

            if category not in task_summary:
                task_summary[category] = {"success": 0, "failure": 0, "ongoing": 0, "durations": []}

            for agent_id, result in task_info.get("agents", {}).items():
                if result.name.lower() == "success":
                    task_summary[category]["success"] += 1
                    if end is not None:
                        task_summary[category]["durations"].append(end - start)
                elif result.name.lower() == "failure":
                    task_summary[category]["failure"] += 1
                else:
                    task_summary[category]["ongoing"] += 1

        # Prepare data for plot
        types = list(task_summary.keys())
        success_counts = [task_summary[t]["success"] for t in types]
        failure_counts = [task_summary[t]["failure"] for t in types]
        ongoing_counts = [task_summary[t]["ongoing"] for t in types]
        avg_completion_times = [
            np.mean(task_summary[t]["durations"]) if task_summary[t]["durations"] else 0
            for t in types
        ]

        x = np.arange(len(types))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))

        # Bars for each category
        rects1 = ax.bar(x - width, success_counts, width, label='Success', color='#4CAF50')  # green
        rects2 = ax.bar(x, failure_counts, width, label='Failure', color='#F44336')  # red
        rects3 = ax.bar(x + width, ongoing_counts, width, label='Ongoing', color='#FFC107')  # yellow-orange


        # Labels and title
        ax.set_ylabel('Count')
        ax.set_title('Task Success vs Failure vs Ongoing Counts')
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=30, ha='right')
        ax.legend()

        # Annotate average completion time above success bars
        for idx, rect in enumerate(rects1):
            height = rect.get_height()
            if height > 0:  # Only annotate if non-zero
                ax.annotate(f'{avg_completion_times[idx]:.1f}s',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='blue')

        ax.grid(True, linestyle='--', alpha=0.5)

        # Save and send to TensorBoard
        self.update_image_plot(name=name, fig=fig, tensorboard_logger=tensorboard_logger, step=step)


    # === Saved Plots Utility ===

    def plot_saved_plots(self, run_dir=None, pattern="*_heatmap_ep*.npy", resend_to_tensorboard=True):
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
                    tensorboard_logger=TensorBoardLogger() if resend_to_tensorboard else None,
                    step=0,
                    episode=None
                )
            except Exception as e:
                print(f"[Plotter] Failed to reload plot from {path}: {e}")
