"""
generate_sample_dicom.py
--------------------------
Generates a small synthetic MRI-like DICOM series (brain-slice-style
concentric structures + realistic noise) purely so the rest of the
system can be run and demoed end-to-end without needing access to
real patient data. Not used for anything clinical - just a stand-in
"scanner output" for development/testing.
"""

import os
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _make_synthetic_slice(size: int, slice_index: int, num_slices: int) -> np.ndarray:
    """Create a brain-scan-like image: concentric soft-tissue rings + noise."""
    y, x = np.ogrid[:size, :size]
    cx, cy = size / 2, size / 2

    # Vary the "skull radius" slightly per slice to fake a 3D head shape
    progress = slice_index / max(num_slices - 1, 1)
    taper = np.sin(progress * np.pi)  # 0 at ends, 1 in middle (like a head silhouette)
    skull_radius = size * 0.42 * (0.6 + 0.4 * taper)

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    img = np.zeros((size, size), dtype=np.float64)
    img[dist < skull_radius] = 900          # skull/CSF baseline
    img[dist < skull_radius * 0.85] = 1400   # gray matter
    img[dist < skull_radius * 0.65] = 1000   # white matter
    img[dist < skull_radius * 0.30] = 1700   # ventricle-like core

    # Add a couple of asymmetric "structures" so slices aren't perfectly radial
    blob1 = np.exp(-(((x - cx - size * 0.12) ** 2 + (y - cy + size * 0.08) ** 2) / (2 * (size * 0.06) ** 2)))
    img += blob1 * 500

    # Gaussian scanner noise
    noise = np.random.normal(0, 40, (size, size))
    img += noise
    img = np.clip(img, 0, 4095)  # 12-bit-ish MRI range
    return img.astype(np.uint16)


def generate_series(out_dir: str, num_slices: int = 12, size: int = 256, seed: int = 42) -> list:
    """Write a synthetic DICOM series to out_dir. Returns list of file paths."""
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    paths = []
    for i in range(num_slices):
        pixel_array = _make_synthetic_slice(size, i, num_slices)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.PatientName = "Synthetic^Demo"
        ds.PatientID = "DEMO001"
        ds.Modality = "MR"
        ds.StudyDescription = "Synthetic Brain MRI (demo data)"
        ds.SeriesDescription = "T1-weighted (simulated)"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.InstanceNumber = i + 1
        ds.SliceLocation = float(i * 3.0)
        ds.SliceThickness = 3.0
        ds.PixelSpacing = [0.9, 0.9]
        ds.Manufacturer = "SyntheticScanner"

        ds.Rows, ds.Columns = pixel_array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_array.tobytes()

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        out_path = os.path.join(out_dir, f"mri_slice_{i:03d}.dcm")
        ds.save_as(out_path, write_like_original=False)
        paths.append(out_path)

    print(f"[generate_sample_dicom] Wrote {len(paths)} synthetic MRI slices to {out_dir}")
    return paths


if __name__ == "__main__":
    generate_series(out_dir=os.path.join(os.path.dirname(__file__), "synthetic_series"))
