# MRI Image Integration & Processing System

A Python-based system to read, process, and integrate MRI scan data in
DICOM format. Built as an independent project to explore healthcare /
medical imaging applications of software engineering.

**Stack:** Python · DICOM (`pydicom`) · NumPy/SciPy · Pillow · Matplotlib

## What it does

1. **DICOM integration** (`src/dicom_loader.py`)
   Reads single `.dcm` files or an entire series from a directory,
   stitches slices into a 3D volume (ordered by instance number /
   slice location), and extracts key metadata (patient, modality,
   pixel spacing, slice thickness, etc.).

2. **Core image processing** (`src/image_processor.py`)
   - Normalization & radiology-style windowing (window center/width)
   - Gaussian and median denoising filters
   - Unsharp-mask sharpening
   - Sobel edge detection
   - Histogram equalization for contrast enhancement
   - All operations work on single slices or full 3D volumes

3. **File-format conversion** (`src/format_converter.py`)
   Converts DICOM pixel data to PNG/JPEG, batch-converts full series,
   and can compile a series into an animated GIF for quick review.

4. **Scan visualization** (`src/visualizer.py`)
   Saves single-slice views, before/after processing comparisons,
   intensity histograms, and full-series montages as image files
   (headless — no display required).

5. **Pipeline entry point** (`src/main.py`)
   Runs the full flow: load → process → convert → visualize.

## Setup

```bash
pip install -r requirements.txt
```

## Demo data

Real MRI DICOM data isn't included, so a synthetic (but structurally
realistic) brain-MRI series can be generated for testing/demos:

```bash
python sample_data/generate_sample_dicom.py
```

This writes 12 synthetic slices to `sample_data/synthetic_series/`.

## Running the pipeline

```bash
cd src
python main.py --input ../sample_data/synthetic_series --output ../output
```

Outputs written to `output/`:
- `png_slices/` — every slice converted to PNG
- `series_review.gif` — animated scroll through the series
- `01_original.png` — a representative slice
- `02_processing_comparison.png` — original vs. windowed / denoised /
  sharpened / edge-detected / equalized
- `03_histogram.png` — pixel intensity distribution
- `04_series_montage.png` — grid view of the whole series

## Project structure

```
mri_project/
├── README.md
├── requirements.txt
├── src/
│   ├── dicom_loader.py
│   ├── image_processor.py
│   ├── format_converter.py
│   ├── visualizer.py
│   └── main.py
└── sample_data/
    └── generate_sample_dicom.py
```

## Using your own DICOM data

Point `--input` at any directory of `.dcm` files (e.g. an exported
series from a PACS system or public dataset like TCIA). No code
changes needed — `DicomLoader.load_series` sorts and stacks whatever
valid DICOM files it finds.

## Notes

This is an educational/portfolio project, not a clinical or
diagnostic tool. The synthetic data generator exists purely so the
pipeline can be demoed without real patient data.
