"""Download ASL Alphabet dataset from Kaggle and extract MediaPipe landmarks.

This script downloads the Kaggle ASL Alphabet image dataset (grassknoted/asl-alphabet),
and runs landmark extraction using `scripts/extract_asl_landmarks.py` to produce
30-frame repeated landmark files (.npy, shape 30x1629) saved to:
    data/asl_alphabet/landmarks/<LETTER>/<index:05d>.npy

Usage:
    python scripts/download_asl_alphabet.py
    python scripts/download_asl_alphabet.py --skip-download
    python scripts/download_asl_alphabet.py --max 150
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.extract_asl_landmarks import find_dataset_dir, process_letter, main as run_extractor


def download_kaggle_dataset(output_dir="data/asl_alphabet/raw"):
    """Downloads the ASL Alphabet image dataset from Kaggle using the Kaggle API."""
    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print("Downloading ASL Alphabet dataset from Kaggle (grassknoted/asl-alphabet)...")
    print("This may take a few minutes (approx 1GB)...")
    os.system(f'kaggle datasets download -d grassknoted/asl-alphabet -p "{output_dir}" --unzip')
    print(f"Download complete! Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract ASL Alphabet landmarks")
    parser.add_argument("--image-dir",     default=None,                              help="Directory of downloaded Kaggle images")
    parser.add_argument("--output-dir",    default="data/asl_alphabet/landmarks",     help="Output directory for landmark .npy files")
    parser.add_argument("--letter",        default=None,                              help="Process only a specific letter (e.g. A)")
    parser.add_argument("--max",           default=150, type=int,                     help="Maximum images to process per letter")
    parser.add_argument("--skip-download", action="store_true",                       help="Skip Kaggle download (images already present)")
    args = parser.parse_args()

    if not args.skip_download:
        # Check if already present locally before downloading
        existing_dir = find_dataset_dir(args.image_dir)
        if existing_dir:
            print(f"Found existing ASL Alphabet dataset at: {existing_dir}")
            print("Skipping Kaggle download.\n")
        else:
            download_kaggle_dataset(args.image_dir or "data/asl_alphabet/raw")

    # Run extraction
    run_extractor()


if __name__ == "__main__":
    main()
