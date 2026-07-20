"""
visualizer.py
--------------
Handles scan visualization workflows: single-slice display, before/after
filter comparisons, histograms, and multi-slice montages of a series.
Saves figures to disk (headless-friendly) rather than requiring an
interactive display.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers/containers
import matplotlib.pyplot as plt


class MRIVisualizer:

    @staticmethod
    def show_image(pixel_array: np.ndarray, title: str, out_path: str) -> str:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(pixel_array, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out_path

    @staticmethod
    def show_comparison(images: dict, out_path: str, suptitle: str = "MRI Processing Comparison") -> str:
        """
        images: dict mapping {label: pixel_array}, e.g.
                {"Original": arr, "Denoised": arr2, "Edges": arr3}
        """
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (label, arr) in zip(axes, images.items()):
            ax.imshow(arr, cmap="gray")
            ax.set_title(label)
            ax.axis("off")
        fig.suptitle(suptitle)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out_path

    @staticmethod
    def show_histogram(pixel_array: np.ndarray, out_path: str, title: str = "Pixel Intensity Histogram") -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(pixel_array.flatten(), bins=256, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Pixel Count")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out_path

    @staticmethod
    def show_montage(volume: np.ndarray, out_path: str, cols: int = 4, title: str = "Series Montage") -> str:
        """Grid view of every slice in a volume - a quick way to review a whole series."""
        n_slices = volume.shape[0]
        rows = int(np.ceil(n_slices / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for i in range(len(axes)):
            axes[i].axis("off")
            if i < n_slices:
                axes[i].imshow(volume[i], cmap="gray")
                axes[i].set_title(f"Slice {i}", fontsize=8)
        fig.suptitle(title)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out_path
