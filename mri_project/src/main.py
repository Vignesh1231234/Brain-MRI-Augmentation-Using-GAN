"""
main.py
-------
End-to-end pipeline for the MRI Image Integration & Processing System.

Usage:
    python main.py --input <dicom_directory> --output <output_directory>

Pipeline:
    1. Load a DICOM series from disk (dicom_loader)
    2. Apply core processing: normalization, denoising, windowing,
       edge detection, histogram equalization (image_processor)
    3. Convert processed slices to PNG for portability (format_converter)
    4. Generate visualization outputs: single slice, before/after
       comparison, histogram, and full-series montage (visualizer)
"""

import argparse
import os
import sys

from dicom_loader import DicomLoader
from image_processor import MRIImageProcessor
from format_converter import FormatConverter
from visualizer import MRIVisualizer


def run_pipeline(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Step 1: Loading DICOM series ===")
    loader = DicomLoader(verbose=True)
    volume, metadata_list = loader.load_series(input_dir)
    meta = metadata_list[0]
    print(f"Patient: {meta.patient_id} | Modality: {meta.modality} | "
          f"Series: {meta.series_description} | Volume shape: {volume.shape}")

    processor = MRIImageProcessor()
    converter = FormatConverter(processor)
    viz = MRIVisualizer()

    mid_idx = volume.shape[0] // 2
    original_slice = volume[mid_idx]

    print("\n=== Step 2: Processing (denoise, window, sharpen, edges, equalize) ===")
    denoised = processor.gaussian_smooth(original_slice, sigma=1.2)
    windowed = processor.apply_windowing(original_slice, window_center=1200, window_width=1600)
    sharpened = processor.sharpen(original_slice, amount=1.5)
    edges = processor.edge_detect(original_slice)
    equalized = processor.histogram_equalize(original_slice)
    print("Processing complete for representative mid-series slice.")

    print("\n=== Step 3: Format conversion (DICOM -> PNG) ===")
    png_dir = os.path.join(output_dir, "png_slices")
    saved_paths = converter.convert_dicom_directory(
        [volume[i] for i in range(volume.shape[0])], png_dir, fmt="PNG"
    )
    print(f"Converted {len(saved_paths)} slices to PNG in {png_dir}")

    gif_path = converter.volume_to_animated_gif(volume, os.path.join(output_dir, "series_review.gif"))
    print(f"Saved animated series review: {gif_path}")

    print("\n=== Step 4: Visualization ===")
    viz.show_image(processor.normalize(original_slice), "Original Mid-Slice",
                    os.path.join(output_dir, "01_original.png"))

    viz.show_comparison(
        {
            "Original": processor.normalize(original_slice),
            "Windowed": windowed,
            "Denoised": processor.normalize(denoised),
            "Sharpened": processor.normalize(sharpened),
            "Edges": edges,
            "Equalized": equalized,
        },
        os.path.join(output_dir, "02_processing_comparison.png"),
    )

    viz.show_histogram(original_slice, os.path.join(output_dir, "03_histogram.png"))
    viz.show_montage(volume, os.path.join(output_dir, "04_series_montage.png"))

    print(f"\nAll outputs written to: {output_dir}")
    print("Pipeline finished successfully.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="MRI Image Integration & Processing System")
    parser.add_argument("--input", "-i", required=True, help="Directory containing DICOM (.dcm) series")
    parser.add_argument("--output", "-o", default="output", help="Directory to write results to")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.input):
        print(f"Error: input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    run_pipeline(args.input, args.output)
