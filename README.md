MRI Image Integration & Processing System
A Python-based system to read, process, and integrate MRI scan data in
DICOM format. Built as an independent project to explore healthcare /
medical imaging applications of software engineering.
Stack: Python · DICOM (`pydicom`) · NumPy/SciPy · Pillow · Matplotlib
What it does
DICOM integration (`src/dicom\_loader.py`)
Reads single `.dcm` files or an entire series from a directory,
stitches slices into a 3D volume (ordered by instance number /
slice location), and extracts key metadata (patient, modality,
pixel spacing, slice thickness, etc.).
Core image processing (`src/image\_processor.py`)
Normalization & radiology-style windowing (window center/width)
Gaussian and median denoising filters
Unsharp-mask sharpening
Sobel edge detection
Histogram equalization for contrast enhancement
All operations work on single slices or full 3D volumes
File-format conversion (`src/format\_converter.py`)
Converts DICOM pixel data to PNG/JPEG, batch-converts full series,
and can compile a series into an animated GIF for quick review.
Scan visualization (`src/visualizer.py`)
Saves single-slice views, before/after processing comparisons,
intensity histograms, and full-series montages as image files
(headless — no display required).
Pipeline entry point (`src/main.py`)
Runs the full flow: load → process → convert → visualize.
Setup
```bash
pip install -r requirements.txt
```
Demo data
Real MRI DICOM data isn't included, so a synthetic (but structurally
realistic) brain-MRI series can be generated for testing/demos:
```bash
python sample\_data/generate\_sample\_dicom.py
```
This writes 12 synthetic slices to `sample\_data/synthetic\_series/`.
Running the pipeline
```bash
cd src
python main.py --input ../sample\_data/synthetic\_series --output ../output
```
Outputs written to `output/`:
`png\_slices/` — every slice converted to PNG
`series\_review.gif` — animated scroll through the series
`01\_original.png` — a representative slice
`02\_processing\_comparison.png` — original vs. windowed / denoised /
sharpened / edge-detected / equalized
`03\_histogram.png` — pixel intensity distribution
`04\_series\_montage.png` — grid view of the whole series
Project structure
```
mri\_project/
├── README.md
├── requirements.txt
├── src/
│   ├── dicom\_loader.py
│   ├── image\_processor.py
│   ├── format\_converter.py
│   ├── visualizer.py
│   └── main.py
└── sample\_data/
    └── generate\_sample\_dicom.py
```
Using your own DICOM data
Point `--input` at any directory of `.dcm` files (e.g. an exported
series from a PACS system or public dataset like TCIA). No code
changes needed — `DicomLoader.load\_series` sorts and stacks whatever
valid DICOM files it finds.
Notes
This is an educational/portfolio project, not a clinical or
diagnostic tool. The synthetic data generator exists purely so the
pipeline can be demoed without real patient data.
