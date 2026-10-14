"""Extract MediaPipe Hand Landmarks from Kaggle ASL Alphabet Dataset.

This script scans the Kaggle ASL Alphabet dataset images (letters A-Z),
runs MediaPipe Hands keypoint extraction (21 landmarks x 3D),
normalizes them into the unified 1629-dimensional temporal format (30 frames x 1629D),
and saves them to:
    data/asl_alphabet/landmarks/<LETTER>/<index:05d>.npy

These extracted landmarks are directly consumed by `scripts/train_wlasl_weights.py`
to train the SpatioTemporalSequenceModel with authentic sign data for all 32 classes.

Usage:
    python scripts/extract_asl_landmarks.py
    python scripts/extract_asl_landmarks.py --max 200
    python scripts/extract_asl_landmarks.py --letter A
    python scripts/extract_asl_landmarks.py --image-dir data/asl_alphabet_train
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe is not installed. Run: pip install mediapipe")
    sys.exit(1)


def find_dataset_dir(custom_path=None) -> str | None:
    """Auto-detects the directory containing the Kaggle ASL Alphabet letters A-Z."""
    candidate_paths = []
    if custom_path:
        candidate_paths.append(custom_path)
        candidate_paths.append(os.path.join(custom_path, "asl_alphabet_train"))

    candidate_paths.extend([
        "data/asl_alphabet_train/asl_alphabet_train",
        "data/asl_alphabet_train",
        "data/asl_alphabet/raw/asl_alphabet_train",
        "data/asl_alphabet/raw",
        "data/asl_alphabet",
    ])

    for path in candidate_paths:
        if os.path.isdir(path):
            # Check if at least letter 'A' folder exists inside
            if os.path.isdir(os.path.join(path, "A")) or os.path.isdir(os.path.join(path, "a")):
                return os.path.abspath(path)

    return None


def extract_hand_sequence(image_path: str, mp_hands_instance) -> np.ndarray | None:
    """Extracts 21 hand keypoints and formats into (30, 1629) temporal sequence.

    Returns:
        np.ndarray of shape (30, 1629) float32, or None if no hand detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # MediaPipe expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands_instance.process(image_rgb)

    if not results or not results.multi_hand_landmarks:
        return None

    hand_lms = results.multi_hand_landmarks[0].landmark
    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

    # ------------------------------------------------------------- #
    # Normalize hand landmarks to align with model coordinate space #
    # ------------------------------------------------------------- #
    # Base wrist anchor (normalized shoulder-width units near chin/chest)
    wrist_base = np.array([0.02, 0.55, 3.0], dtype=np.float32)

    # Measure palm scale (wrist pt 0 to middle finger MCP pt 9)
    raw_wrist = raw_pts[0]
    palm_len = float(np.linalg.norm(raw_pts[9] - raw_wrist))
    # Target palm length in shoulder-width space is ~0.14
    scale_factor = 0.14 / max(palm_len, 0.05)

    # Center at wrist and scale
    centered_hand = (raw_pts - raw_wrist) * scale_factor

    # Invert Y if needed so fingers point upwards (MediaPipe image Y is downwards)
    # Fingers extended have lower Y in image coordinates
    centered_hand[:, 1] = -centered_hand[:, 1]
    norm_hand = centered_hand + wrist_base

    # ------------------------------------------------------------- #
    # Build 1629D feature vector:                                   #
    #   0..98    : Pose (99D) - nose + shoulders + wrist anchor     #
    #   99..1502 : Face (1404D) - zeros (masked)                    #
    #   1503..1565 : Left Hand (63D) - zeros                        #
    #   1566..1628 : Right Hand (63D) - normalized hand landmarks   #
    # ------------------------------------------------------------- #
    vec = np.zeros(1629, dtype=np.float32)

    # Anchor pose landmarks
    vec[0:3] = [0.0, 0.0, 0.0]       # Nose at origin
    vec[33:36] = [-0.5, 0.5, 0.0]    # Left shoulder (idx 11: 11*3=33)
    vec[36:39] = [0.5, 0.5, 0.0]     # Right shoulder (idx 12: 12*3=36)
    vec[48:51] = wrist_base          # Right wrist (idx 16: 16*3=48)

    # Set Right Hand (21 x 3 = 63D at index 1566)
    vec[1566:1629] = norm_hand.flatten()

    # Repeat across 30 temporal frames (static hand gesture)
    sequence = np.tile(vec, (30, 1)).astype(np.float32)

    # Add subtle temporal micro-jitter (0.003 std) on active keypoints
    noise = np.random.normal(0, 0.003, sequence.shape).astype(np.float32)
    active_mask = (sequence != 0.0).astype(np.float32)
    sequence = (sequence + noise) * active_mask

    return sequence


def process_letter(letter: str, dataset_dir: str, output_dir: str, mp_hands, max_images: int = 150) -> int:
    """Processes images for one letter and saves .npy landmark sequences with resume capability."""
    letter_upper = letter.upper()
    letter_dir = os.path.join(dataset_dir, letter_upper)
    if not os.path.isdir(letter_dir):
        letter_dir = os.path.join(dataset_dir, letter.lower())
    if not os.path.isdir(letter_dir):
        print(f"  [{letter_upper}] ⚠️  Directory not found: {letter_dir}")
        return 0

    out_folder = os.path.join(output_dir, letter_upper)
    os.makedirs(out_folder, exist_ok=True)

    existing_files = glob.glob(os.path.join(out_folder, "*.npy"))
    existing_count = len(existing_files)

    # Check if already completed
    if max_images > 0 and existing_count >= max_images:
        print(f"  [{letter_upper}] Already has {existing_count:4d} landmark sequences (>= target {max_images}) -> Complete.")
        return existing_count

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(letter_dir, ext)))

    if not image_files:
        print(f"  [{letter_upper}] ⚠️  No image files found in {letter_dir}")
        return existing_count

    target_count = max_images if max_images > 0 else len(image_files)
    saved_count = existing_count

    # Check disk space before writing
    import shutil
    try:
        drive_path = os.path.splitdrive(os.path.abspath(output_dir))[0] or output_dir
        _, _, free_bytes = shutil.disk_usage(drive_path)
        if free_bytes < 500 * 1024 * 1024:  # Less than 500 MB
            print(f"  [{letter_upper}] ⚠️ Disk space low ({free_bytes / (1024**2):.1f} MB free). Pausing extraction.")
            return saved_count
    except Exception:
        pass

    # Process remaining images up to target_count
    for idx, img_path in enumerate(image_files[existing_count:]):
        if saved_count >= target_count:
            break
        seq = extract_hand_sequence(img_path, mp_hands)
        if seq is not None:
            out_file = os.path.join(out_folder, f"{saved_count:05d}.npy")
            np.save(out_file, seq)
            saved_count += 1

    pct = (saved_count / target_count * 100) if target_count else 0
    print(f"  [{letter_upper}] Target: {target_count:4d} | Extracted: {saved_count:4d} ({pct:.1f}% ready)")
    return saved_count



def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Hand Landmarks from ASL Alphabet Images")
    parser.add_argument("--image-dir",  default=None,                          help="Path to folder containing A-Z subfolders")
    parser.add_argument("--output-dir", default="data/asl_alphabet/landmarks", help="Output path for landmark .npy files")
    parser.add_argument("--letter",     default=None,                          help="Process only a specific letter (e.g. A)")
    parser.add_argument("--max",        default=150, type=int,                 help="Maximum images per letter (default: 150)")
    args = parser.parse_args()

    print("=" * 72)
    print("  ASL Alphabet MediaPipe Hand Landmark Extractor")
    print("=" * 72)

    dataset_dir = find_dataset_dir(args.image_dir)
    if not dataset_dir:
        print("\n❌ ERROR: Could not locate ASL Alphabet dataset directory.")
        print("Please check that the dataset is unzipped in data/asl_alphabet_train/")
        print("or specify the path with --image-dir <path>.")
        sys.exit(1)

    print(f"Dataset root: {dataset_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Max images:   {args.max} per letter")

    letters = [args.letter.upper()] if args.letter else [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    print(f"Processing {len(letters)} letter(s): {', '.join(letters)}\n")

    # Initialize MediaPipe Hands in static image mode
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.25
    )

    total_extracted = 0
    letter_counts = {}

    try:
        for letter in letters:
            count = process_letter(letter, dataset_dir, args.output_dir, mp_hands, max_images=args.max)
            letter_counts[letter] = count
            total_extracted += count
    finally:
        mp_hands.close()

    print("\n" + "=" * 72)
    print(f"  EXTRACTION COMPLETE: {total_extracted} total landmark sequences saved!")
    print("=" * 72)
    print(f"Output files stored in: {os.path.abspath(args.output_dir)}")
    print("\nNext step: Retrain the 32-class model with real landmarks by running:")
    print("  python scripts/train_wlasl_weights.py\n")


if __name__ == "__main__":
    main()
