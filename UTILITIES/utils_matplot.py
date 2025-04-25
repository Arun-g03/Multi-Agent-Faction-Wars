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
from UTILITIES.utils_tensorboard import TensorBoardLogger


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

    def _plot_to_tensor(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        buf.close()
        plt.close(fig)
        return image_tensor

    def update_image_plot(self, name, fig=None, tensorboard_logger=None, step=0):
        save_dir = self.image_dir
        if tensorboard_logger and hasattr(tensorboard_logger, "run_dir"):
            save_dir = tensorboard_logger.run_dir
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"{name}.png")

        if fig is None:
            fig = plt.gcf()
        fig.savefig(image_path)
        print(f"[Plotter] Saved plot to {image_path}")

        if tensorboard_logger:
            image_tensor = self._plot_to_tensor(fig)
            tensorboard_logger.log_image(name, image_tensor, step)
            print(f"[Plotter] Sent '{name}' to TensorBoard.")

    def plot_heatmap(self, data, name="heatmap", title=None,
                     tensorboard_logger=None, step=0, episode=None,
                     save_data=True, save_as="npy"):
        fig, ax = plt.subplots(figsize=(10, 2.5))  # Wider and flatter

        if isinstance(data, pd.DataFrame):
            sns.heatmap(data, annot=True, fmt="d", cmap="viridis", ax=ax, cbar=True)
            ax.set_xticklabels(data.columns, rotation=45, ha='right')
        else:
            sns.heatmap(data, annot=True, fmt="d", cmap="viridis", ax=ax, cbar=True)

        ax.set_yticks([0])  # Single row: label as "Summary"
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

    def add_episode_matrix(self, name, matrix, step=0, episode=None):
        self.episode_data[name].append((np.array(matrix), step, episode))

    def flush_episode_heatmaps(self, tensorboard_logger=None, save_data=True, save_as="npy"):
        for name, entries in self.episode_data.items():
            try:
                matrices = [m for m, _, _ in entries]
                step = max(s for _, s, _ in entries)
                episode = max(e for _, _, e in entries if e is not None)

                stacked = np.vstack(matrices)
                summary = np.sum(stacked, axis=0, keepdims=True)

                action_labels = []
                try:
                    role_name = name.replace("_actions", "").replace("_task_distribution", "")
                    
                    if "task_distribution" in name:
                        title = f"{role_name.title()} Task Summary"
                    else:
                        title = f"{role_name.title()} Action Summary"

                    action_labels = label_source[:summary.shape[1]]

                except:
                    action_labels = [str(i) for i in range(summary.shape[1])]

                df = pd.DataFrame(summary, columns=action_labels)

                self.plot_heatmap(
                    data=df,
                    name=name,
                    title=f"{role.title()} Action Summary",
                    tensorboard_logger=tensorboard_logger,
                    step=step,
                    episode=episode,
                    save_data=save_data,
                    save_as=save_as
                )
            except Exception as e:
                print(f"[Plotter] Failed to plot {name}: {e}")

        self.episode_data.clear()

    def plot_saved_heatmaps(self, run_dir=None, pattern="*_heatmap_ep*.npy", resend_to_tensorboard=True):
        run_dir = run_dir or self.image_dir
        files = glob.glob(os.path.join(run_dir, pattern))

        if not files:
            print("[Plotter] No saved heatmaps found.")
            return

        logger = TensorBoardLogger()
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

                print(f"[Plotter] Re-plotting heatmap from {path}")
                self.plot_heatmap(
                    data=matrix,
                    name=name,
                    title=name.replace("_", " ").title(),
                    tensorboard_logger=logger if resend_to_tensorboard else None,
                    step=0,
                    episode=None
                )
            except Exception as e:
                print(f"[Plotter] Failed to reload plot from {path}: {e}")
