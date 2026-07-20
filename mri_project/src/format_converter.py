"""
format_converter.py
--------------------
Handles file-format conversion workflows: turning DICOM slices/series
into common image formats (PNG, JPEG) for sharing, reporting, or
feeding into non-DICOM-aware tools, and batch-converting whole folders.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

from image_processor import MRIImageProcessor


class FormatConverter:
    """Converts processed MRI pixel arrays to standard image file formats."""

    def __init__(self, processor: MRIImageProcessor | None = None):
        self.processor = processor or MRIImageProcessor()

    def array_to_image(self, pixel_array: np.ndarray, auto_normalize: bool = True) -> Image.Image:
        """Convert a numpy pixel array into a Pillow Image object."""
        arr = pixel_array
        if auto_normalize and (arr.dtype != np.uint8 or arr.max() > 255):
            arr = self.processor.normalize(arr)
        return Image.fromarray(arr)

    def save_slice(self, pixel_array: np.ndarray, out_path: str, fmt: str = "PNG") -> str:
        """Save a single processed slice to disk in the requested format."""
        img = self.array_to_image(pixel_array)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        img.save(out_path, format=fmt)
        return out_path

    def convert_dicom_directory(self, dicom_arrays: list[np.ndarray], out_dir: str,
                                 fmt: str = "PNG", prefix: str = "slice") -> list[str]:
        """
        Batch-convert a list of already-loaded DICOM pixel arrays (e.g. an
        entire series) into individual image files.
        """
        os.makedirs(out_dir, exist_ok=True)
        ext = "png" if fmt.upper() == "PNG" else "jpg"
        saved_paths = []
        for i, arr in enumerate(dicom_arrays):
            out_path = os.path.join(out_dir, f"{prefix}_{i:03d}.{ext}")
            self.save_slice(arr, out_path, fmt=fmt)
            saved_paths.append(out_path)
        return saved_paths

    def volume_to_animated_gif(self, volume: np.ndarray, out_path: str, duration_ms: int = 100) -> str:
        """Convert a 3D MRI volume into an animated GIF - handy for reviewing a whole series at a glance."""
        frames = [self.array_to_image(volume[i]) for i in range(volume.shape[0])]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        frames[0].save(
            out_path, save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0
        )
        return out_path
