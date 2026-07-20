"""
dicom_loader.py
----------------
Core module for reading and integrating MRI scan data stored in the
DICOM format (.dcm). Wraps pydicom to provide a clean interface for:
  - loading single files
  - loading an entire series (a directory of slices) as a 3D volume
  - extracting patient / study / series metadata
  - pulling out the raw pixel array for downstream processing

This is the "integration" layer of the system: it normalizes the many
ways DICOM data can show up (single files, multi-file series, missing
tags, different bit depths) into one consistent object.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class DicomMetadata:
    """Flattened, human-friendly view of the DICOM tags we actually care about."""
    patient_id: str = "UNKNOWN"
    patient_name: str = "UNKNOWN"
    modality: str = "UNKNOWN"
    study_description: str = ""
    series_description: str = ""
    rows: int = 0
    columns: int = 0
    pixel_spacing: Optional[list] = None
    slice_thickness: Optional[float] = None
    bits_allocated: int = 16
    manufacturer: str = ""
    extra: dict = field(default_factory=dict)


class DicomLoadError(Exception):
    """Raised when a file/directory can't be read as valid DICOM."""


class DicomLoader:
    """Loads and integrates DICOM MRI data from disk."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Single-file loading
    # ------------------------------------------------------------------ #
    def load_file(self, filepath: str) -> tuple[np.ndarray, DicomMetadata]:
        """
        Read a single .dcm file.

        Returns
        -------
        pixel_array : np.ndarray
            The raw image data (rows x columns), in its native dtype.
        metadata : DicomMetadata
            Parsed header information.
        """
        if not os.path.isfile(filepath):
            raise DicomLoadError(f"File not found: {filepath}")

        try:
            ds = pydicom.dcmread(filepath)
        except InvalidDicomError as e:
            raise DicomLoadError(f"Not a valid DICOM file: {filepath}") from e

        if not hasattr(ds, "PixelData"):
            raise DicomLoadError(f"DICOM file has no pixel data: {filepath}")

        pixel_array = ds.pixel_array.astype(np.float64)
        metadata = self._extract_metadata(ds)

        if self.verbose:
            print(f"[DicomLoader] Loaded {os.path.basename(filepath)} "
                  f"({metadata.rows}x{metadata.columns}, modality={metadata.modality})")

        return pixel_array, metadata

    # ------------------------------------------------------------------ #
    # Series (directory) loading -> 3D volume
    # ------------------------------------------------------------------ #
    def load_series(self, directory: str, pattern: str = "*.dcm") -> tuple[np.ndarray, list[DicomMetadata]]:
        """
        Read every DICOM file in a directory and stack them into a 3D
        volume, ordered by slice position (InstanceNumber / SliceLocation
        when available, otherwise filename order).

        Returns
        -------
        volume : np.ndarray, shape (num_slices, rows, cols)
        metadata_list : list[DicomMetadata]
        """
        filepaths = sorted(glob.glob(os.path.join(directory, pattern)))
        if not filepaths:
            raise DicomLoadError(f"No files matching '{pattern}' found in {directory}")

        slices = []
        for fp in filepaths:
            try:
                ds = pydicom.dcmread(fp)
                sort_key = getattr(ds, "InstanceNumber", None)
                if sort_key is None:
                    sort_key = getattr(ds, "SliceLocation", filepaths.index(fp))
                slices.append((float(sort_key), ds))
            except InvalidDicomError:
                if self.verbose:
                    print(f"[DicomLoader] Skipping non-DICOM file: {fp}")
                continue

        if not slices:
            raise DicomLoadError(f"No valid DICOM slices found in {directory}")

        slices.sort(key=lambda x: x[0])

        volume_slices = []
        metadata_list = []
        for _, ds in slices:
            volume_slices.append(ds.pixel_array.astype(np.float64))
            metadata_list.append(self._extract_metadata(ds))

        volume = np.stack(volume_slices, axis=0)

        if self.verbose:
            print(f"[DicomLoader] Assembled volume {volume.shape} from {len(slices)} slices in {directory}")

        return volume, metadata_list

    # ------------------------------------------------------------------ #
    # Metadata extraction
    # ------------------------------------------------------------------ #
    def _extract_metadata(self, ds: pydicom.Dataset) -> DicomMetadata:
        return DicomMetadata(
            patient_id=str(getattr(ds, "PatientID", "UNKNOWN")),
            patient_name=str(getattr(ds, "PatientName", "UNKNOWN")),
            modality=str(getattr(ds, "Modality", "UNKNOWN")),
            study_description=str(getattr(ds, "StudyDescription", "")),
            series_description=str(getattr(ds, "SeriesDescription", "")),
            rows=int(getattr(ds, "Rows", 0)),
            columns=int(getattr(ds, "Columns", 0)),
            pixel_spacing=list(getattr(ds, "PixelSpacing", [])) or None,
            slice_thickness=float(getattr(ds, "SliceThickness", 0)) or None,
            bits_allocated=int(getattr(ds, "BitsAllocated", 16)),
            manufacturer=str(getattr(ds, "Manufacturer", "")),
        )
