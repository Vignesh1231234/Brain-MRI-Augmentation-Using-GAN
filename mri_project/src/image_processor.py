"""
image_processor.py
-------------------
Core image-processing operations applied to MRI pixel arrays after
they've been loaded by DicomLoader. Covers the "filtering, pixel-array
handling" work: windowing/normalization, denoising, edge/contrast
enhancement, and histogram equalization.

All functions operate on plain numpy arrays so they can be reused for
single slices (2D) or whole volumes (3D) with minimal changes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class MRIImageProcessor:
    """A small toolbox of pixel-level operations for MRI slices."""

    # ------------------------------------------------------------------ #
    # Normalization / windowing
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(pixel_array: np.ndarray) -> np.ndarray:
        """Rescale any pixel array to the 0-255 uint8 range for display/export."""
        arr = pixel_array.astype(np.float64)
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - min_val) / (max_val - min_val) * 255.0
        return scaled.astype(np.uint8)

    @staticmethod
    def apply_windowing(pixel_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """
        Apply radiology-style windowing (a.k.a. window/level).
        Maps [center - width/2, center + width/2] to [0, 255], clipping
        outside that range. This is how MRI/CT viewers let you emphasize
        soft tissue vs bone vs air.
        """
        arr = pixel_array.astype(np.float64)
        low = window_center - window_width / 2.0
        high = window_center + window_width / 2.0
        arr = np.clip(arr, low, high)
        arr = (arr - low) / (high - low) * 255.0
        return arr.astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    @staticmethod
    def gaussian_smooth(pixel_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian blur to reduce scan noise."""
        return ndimage.gaussian_filter(pixel_array.astype(np.float64), sigma=sigma)

    @staticmethod
    def median_denoise(pixel_array: np.ndarray, size: int = 3) -> np.ndarray:
        """Median filter, good for salt-and-pepper style noise without blurring edges as much."""
        return ndimage.median_filter(pixel_array.astype(np.float64), size=size)

    @staticmethod
    def sharpen(pixel_array: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """Unsharp-mask style sharpening to bring out tissue boundaries."""
        arr = pixel_array.astype(np.float64)
        blurred = ndimage.gaussian_filter(arr, sigma=1.0)
        sharpened = arr + amount * (arr - blurred)
        return np.clip(sharpened, arr.min(), arr.max())

    @staticmethod
    def edge_detect(pixel_array: np.ndarray) -> np.ndarray:
        """Sobel-based edge map, useful for spotting boundaries between structures."""
        arr = pixel_array.astype(np.float64)
        sx = ndimage.sobel(arr, axis=0)
        sy = ndimage.sobel(arr, axis=1)
        magnitude = np.hypot(sx, sy)
        return MRIImageProcessor.normalize(magnitude)

    # ------------------------------------------------------------------ #
    # Contrast enhancement
    # ------------------------------------------------------------------ #
    @staticmethod
    def histogram_equalize(pixel_array: np.ndarray) -> np.ndarray:
        """Classic histogram equalization to boost global contrast."""
        arr = MRIImageProcessor.normalize(pixel_array)
        hist, bins = np.histogram(arr.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype(np.uint8)
        return cdf_final[arr]

    # ------------------------------------------------------------------ #
    # Volume-level (3D) helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_to_volume(volume: np.ndarray, func, **kwargs) -> np.ndarray:
        """Apply any 2D slice function to every slice of a 3D volume."""
        return np.stack([func(volume[i], **kwargs) for i in range(volume.shape[0])], axis=0)
