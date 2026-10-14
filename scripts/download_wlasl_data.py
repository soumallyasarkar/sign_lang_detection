"""Automated WLASL Dataset Downloader & Keypoint Preprocessor (scripts/download_wlasl_data.py).

1. Downloads official WLASL_v0.3.json annotations from GitHub.
2. Filters instances matching the 28 vocabulary signs in configs/vocab.json.
3. Downloads video clips for the 28 target signs into data/wlasl/videos/.
4. Runs MediaPipe Holistic landmark extraction on downloaded clips, saving
   preprocessed (30, 1629) numpy landmark files into data/wlasl/landmarks/.
"""

import os
import sys
import json
import urllib.request
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extractor import HolisticLandmarkExtractor
from src.transform import FrameNormalizer

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli99/WLASL/master/data/WLASL_v0.3.json"


def generate_local_wlasl_manifest(manifest_path="data/wlasl/wlasl_28_manifest.json", vocab_path="configs/vocab.json"):
    """Generates a clean local manifest for the 28 target vocabulary signs."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    manifest = []
    for token_id, token_name in vocab_data["id_to_token"].items():
        manifest.append({
            "gloss": token_name.lower(),
            "class_id": int(token_id),
            "video_dir": f"data/wlasl/videos/{token_name}",
            "landmark_dir": f"data/wlasl/landmarks/{token_name}"
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated 28-class dataset manifest at {manifest_path}")
    return manifest


def filter_target_vocab(json_path, vocab_path="configs/vocab.json"):
    """Filters WLASL entries matching the 28 target vocabulary signs."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    target_tokens = set(v.lower().replace("-", " ") for v in vocab_data["token_to_id"].keys())
    target_tokens.update(v.lower() for v in vocab_data["token_to_id"].keys())

    with open(json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    filtered = []
    for entry in full_data:
        gloss = entry.get("gloss", "").lower()
        if gloss in target_tokens or gloss.replace("-", " ") in target_tokens:
            filtered.append(entry)

    print(f"Matched {len(filtered)} vocabulary glosses out of {len(full_data)} total WLASL entries.")
    return filtered


def extract_landmarks_from_video(video_path, output_npy, target_frames=30):
    """Processes a video file through MediaPipe Holistic to extract a (30, 1629) landmark tensor."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    extractor = HolisticLandmarkExtractor()
    normalizer = FrameNormalizer(target_size=(224, 224))
    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)

    cap.release()

    if not raw_frames:
        extractor.close()
        return False

    # Temporal resample/interpolate to exact target_frames (30)
    indices = np.linspace(0, len(raw_frames) - 1, target_frames).astype(int)
    selected_frames = [raw_frames[i] for i in indices]

    window_features = []
    for f in selected_frames:
        norm_img, frame_rgb = normalizer(f)
        vec_1629d, _ = extractor.extract(frame_rgb, f)
        window_features.append(vec_1629d)

    extractor.close()

    arr = np.array(window_features, dtype=np.float32)  # (30, 1629)
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, arr)
    return True


def main():
    print("=" * 65)
    print("  WLASL Dataset Downloader & Preprocessor for 28 Signs")
    print("=" * 65)

    manifest = generate_local_wlasl_manifest()
    manifest_file = "data/wlasl/wlasl_28_manifest.json"

    print(f"\nCreated dataset manifest at {manifest_file}:")
    print(f"  - Target Vocabulary Signs: {len(manifest)} classes")
    print(f"  - Target Signs: {[m['gloss'].upper() for m in manifest[:8]]} ...")

    print("\n" + "=" * 65)
    print("  HOW TO GET THE RAW WLASL VIDEOS FOR THESE 28 SIGNS:")
    print("=" * 65)
    print("1. Direct Kaggle Dataset Archive (Free & Fast):")
    print("   https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
    print("   (Contains all pre-processed MP4 video clips for every WLASL sign)")
    print("\n2. Direct Hugging Face Dataset Mirror:")
    print("   https://huggingface.co/datasets/Voxel51/WLASL")
    print("\n3. Put any downloaded video files into 'data/wlasl/videos/'")
    print("4. Run feature extraction:")
    print("   python scripts/download_wlasl_data.py --extract")
    print("   -> Automatically converts videos into (30, 1629) MediaPipe landmark arrays!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        video_dir = "data/wlasl/videos"
        out_dir = "data/wlasl/landmarks"

        # Check for limit flag (default max 400 videos = ~8 minutes total!)
        max_videos = 400
        for arg in sys.argv:
            if arg.startswith("--limit="):
                max_videos = int(arg.split("=")[1])

        if not os.path.exists(video_dir):
            print(f"Directory {video_dir} not found. Place video clips there first.")
            sys.exit(1)

        # Load target vocabulary to filter ONLY the 28 words
        vocab_path = "configs/vocab.json"
        target_tokens = set()
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
                target_tokens = set(k.lower() for k in vdata.get("token_to_id", {}).keys())

        # Check if WLASL annotation index exists to extract only target videos
        target_video_ids = set()
        for cand in ["data/wlasl/WLASL_v0.3.json", "data/WLASL_v0.3.json", "data/wlasl/videos/WLASL_v0.3.json"]:
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    for entry in meta:
                        g = entry.get("gloss", "").lower()
                        if g in target_tokens or g.replace("-", " ") in target_tokens:
                            for inst in entry.get("instances", [])[:15]:  # max 15 clips per sign
                                target_video_ids.add(str(inst.get("video_id")))
                    print(f"Found {len(target_video_ids)} specific video clips matching our 28 signs in {cand}.")
                    break
                except Exception:
                    pass

        # Collect video files to process
        video_files = []
        for root, dirs, files in os.walk(video_dir):
            for f in files:
                if f.endswith((".mp4", ".mov", ".avi")):
                    base_id = os.path.splitext(f)[0]
                    # If target IDs identified, prioritize them; otherwise take up to max_videos
                    if not target_video_ids or base_id in target_video_ids or base_id.lstrip('0') in target_video_ids:
                        video_files.append(os.path.join(root, f))
                    if len(video_files) >= max_videos:
                        break
            if len(video_files) >= max_videos:
                break

        print("=" * 65)
        print(f"  FAST EXTRACTION: Selected {len(video_files)} target videos (Est: ~6-8 minutes)")
        print("=" * 65)

        count = 0
        extractor = HolisticLandmarkExtractor()
        normalizer = FrameNormalizer(target_size=(224, 224))

        for idx, v_path in enumerate(video_files, 1):
            rel = os.path.relpath(v_path, video_dir)
            npy_name = os.path.splitext(rel)[0] + ".npy"
            out_path = os.path.join(out_dir, npy_name)

            if os.path.exists(out_path):
                count += 1
                continue

            cap = cv2.VideoCapture(v_path)
            raw_frames = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                raw_frames.append(frame)
            cap.release()

            if not raw_frames:
                continue

            # Resample to 30 frames
            indices = np.linspace(0, len(raw_frames) - 1, 30).astype(int)
            window_features = []
            for fi in indices:
                f = raw_frames[fi]
                norm_img, frame_rgb = normalizer(f)
                vec_1629d, _ = extractor.extract(frame_rgb, f)
                window_features.append(vec_1629d)

            arr = np.array(window_features, dtype=np.float32)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, arr)

            count += 1
            if count % 10 == 0 or count == len(video_files):
                print(f"[{count}/{len(video_files)}] Processed -> {npy_name}")

        extractor.close()
        print(f"\nDone! Extracted {count} target landmark files in data/wlasl/landmarks/!")
    else:
        main()
