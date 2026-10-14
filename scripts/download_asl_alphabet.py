"""Download ASL Alphabet dataset from Kaggle and extract MediaPipe landmarks.

This script downloads the Kaggle ASL Alphabet image dataset (grassknoted/asl-alphabet),
runs Google MediaPipe Holistic landmark extraction on each image for letters A-Z,
and saves 30-frame repeated landmark files (.npy, shape 30x1629) to:
    data/asl_alphabet/landmarks/<LETTER>/<index>.npy

Prerequisites:
    1. Install Kaggle API:   pip install kaggle
    2. Set up your API key: Place ~/.kaggle/kaggle.json  (from kaggle.com -> Account -> API)
    3. Install mediapipe:   pip install mediapipe
    4. Install opencv:      pip install opencv-python

Usage:
    python scripts/download_asl_alphabet.py
    python scripts/download_asl_alphabet.py --letter A   # Process only letter A
    python scripts/download_asl_alphabet.py --max 100     # Max 100 images per letter
"""

import os
import sys
import argparse
import zipfile
import shutil
import glob
import numpy as np

# ------------------------------------------------------------------ #
# Check dependencies before heavy imports                            #
# ------------------------------------------------------------------ #
try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)


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


def extract_holistic_landmarks(image_path: str, holistic) -> np.ndarray | None:
    """Extracts 1629-dimensional MediaPipe Holistic landmark vector from a single image.

    Returns shape (1629,) or None if MediaPipe finds no landmarks.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # MediaPipe expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    if not results.right_hand_landmarks and not results.left_hand_landmarks:
        return None

    vec = np.zeros(1629, dtype=np.float32)

    # ---- Pose (33 landmarks x 3 = 99) ----
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        shoulder_w = max(abs(r_sh.x - l_sh.x), 1e-6)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            vec[i*3]   = (lm.x - nose.x) / shoulder_w
            vec[i*3+1] = (lm.y - nose.y) / shoulder_w
            vec[i*3+2] = lm.z / shoulder_w

    # ---- Face mesh (468 landmarks x 3 = 1404): left as zeros (masked in training) ----
    # vec[99:1503] = 0.0  — already zero

    # ---- Left hand (21 landmarks x 3 = 63) at index 1503 ----
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            vec[1503 + i*3]   = lm.x
            vec[1503 + i*3+1] = lm.y
            vec[1503 + i*3+2] = lm.z

    # ---- Right hand (21 landmarks x 3 = 63) at index 1566 ----
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            vec[1566 + i*3]   = lm.x
            vec[1566 + i*3+1] = lm.y
            vec[1566 + i*3+2] = lm.z

    return vec


def process_letter(letter: str, image_dir: str, output_dir: str, holistic, max_images: int = 150):
    """Processes all images for one letter, saving 30-frame repeated landmark .npy files."""
    letter_upper = letter.upper()
    out_path = os.path.join(output_dir, letter_upper)
    os.makedirs(out_path, exist_ok=True)

    # Search for images in the downloaded dataset
    image_patterns = [
        os.path.join(image_dir, "asl_alphabet_train", letter_upper, "*.jpg"),
        os.path.join(image_dir, "asl_alphabet_train", letter_upper, "*.png"),
        os.path.join(image_dir, "asl_alphabet_train", letter_upper.lower(), "*.jpg"),
        os.path.join(image_dir, letter_upper, "*.jpg"),
        os.path.join(image_dir, letter_upper, "*.png"),
    ]
    images = []
    for pat in image_patterns:
        images.extend(glob.glob(pat))
    images = images[:max_images]

    if not images:
        print(f"  [{letter_upper}] WARNING: No images found in {image_dir}")
        return 0

    saved = 0
    for idx, img_path in enumerate(images):
        vec = extract_holistic_landmarks(img_path, holistic)
        if vec is None:
            continue

        # Repeat the single frame 30 times to create a (30, 1629) sequence
        # (static handshapes have no temporal variation — augmentation handles diversity)
        sequence = np.tile(vec, (30, 1)).astype(np.float32)  # shape: (30, 1629)

        # Add subtle per-frame jitter for slight temporal realism
        noise = np.random.normal(0, 0.004, sequence.shape).astype(np.float32)
        active_mask = (sequence != 0.0).astype(np.float32)
        sequence = (sequence + noise) * active_mask

        out_file = os.path.join(out_path, f"{idx:05d}.npy")
        np.save(out_file, sequence)
        saved += 1

    print(f"  [{letter_upper}] Processed {len(images)} images → {saved} landmark files saved.")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download and extract ASL Alphabet landmarks")
    parser.add_argument("--image-dir",  default="data/asl_alphabet/raw",         help="Directory of downloaded Kaggle images")
    parser.add_argument("--output-dir", default="data/asl_alphabet/landmarks",    help="Output directory for landmark .npy files")
    parser.add_argument("--letter",     default=None,                             help="Process only a specific letter (e.g. A)")
    parser.add_argument("--max",        default=150, type=int,                    help="Maximum images to process per letter")
    parser.add_argument("--skip-download", action="store_true",                   help="Skip Kaggle download (images already present)")
    args = parser.parse_args()

    if not args.skip_download:
        download_kaggle_dataset(args.image_dir)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4
    )

    letters = [args.letter.upper()] if args.letter else [chr(c) for c in range(ord('A'), ord('Z') + 1)]

    print(f"\nExtracting MediaPipe Holistic landmarks for {len(letters)} letter(s)...")
    print(f"Output: {args.output_dir}\n")

    total = 0
    for letter in letters:
        n = process_letter(letter, args.image_dir, args.output_dir, holistic, max_images=args.max)
        total += n

    holistic.close()
    print(f"\nDone! Total landmark files saved: {total}")
    print(f"\nNow update configs/vocab.json and re-run: python scripts/train_wlasl_weights.py")


if __name__ == "__main__":
    main()
