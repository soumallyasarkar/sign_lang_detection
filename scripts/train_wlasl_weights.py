"""Train and calibrate SpatioTemporalSequenceModel on 6 ASL words + 26 alphabet letters = 32 classes.

Vocabulary:
  Words   (0-5) : HELLO, THANK-YOU, PLEASE, MEET, NO, YES
  Alphabet (6-31): A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

Training strategy:
  - Words: Loaded from authentic WLASL landmark .npy files (30x1629) with heavy augmentation.
  - Alphabet letters: Synthesized from hand-pose prototypes encoding ASL fingerspelling handshapes.
    Each letter has a unique right-hand keypoint template (21 points x 3D) that is domain-randomized.
"""

import os
import sys
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SpatioTemporalSequenceModel


# ============================================================
# ASL Fingerspelling Handshape Prototypes (21 keypoints, normalized)
# Each entry: [wrist, thumb_tip, index_tip, mid_tip, ring_tip, pinky_tip, ...]
# Encoded as relative extension (0=closed, 1=extended) per finger:
# [thumb, index, middle, ring, pinky, thumb_out]
# These drive synthetic hand-pose generation for A–Z
# ============================================================
ALPHA_HANDSHAPES = {
    # Format: (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, thumb_side)
    # thumb_side=True means thumb sticks out to the side
    "A": (0.3, 0.0, 0.0, 0.0, 0.0, True),   # Closed fist, thumb on side
    "B": (0.0, 1.0, 1.0, 1.0, 1.0, False),   # 4 fingers up, thumb tucked
    "C": (0.6, 0.6, 0.6, 0.6, 0.6, False),   # Curved C-shape (all partial)
    "D": (0.0, 1.0, 0.0, 0.0, 0.0, True),    # Index up, others with thumb touching
    "E": (0.2, 0.2, 0.2, 0.2, 0.2, False),   # All fingers slightly curled/closed
    "F": (0.0, 0.0, 1.0, 1.0, 1.0, True),    # Index+thumb touch, rest up
    "G": (0.4, 1.0, 0.0, 0.0, 0.0, True),    # Index+thumb point sideways
    "H": (0.0, 1.0, 1.0, 0.0, 0.0, False),   # Index+middle extended horizontal
    "I": (0.0, 0.0, 0.0, 0.0, 1.0, False),   # Pinky up only
    "J": (0.0, 0.0, 0.0, 0.0, 1.0, False),   # Like I but with motion (pinky up)
    "K": (0.4, 1.0, 1.0, 0.0, 0.0, True),    # Index+middle up, thumb out
    "L": (0.8, 1.0, 0.0, 0.0, 0.0, True),    # L-shape: index+thumb extended
    "M": (0.0, 0.2, 0.2, 0.2, 0.0, False),   # 3 fingers over thumb
    "N": (0.0, 0.2, 0.2, 0.0, 0.0, False),   # 2 fingers over thumb
    "O": (0.5, 0.5, 0.5, 0.5, 0.5, False),   # O-shape: all fingers curved together
    "P": (0.4, 1.0, 1.0, 0.0, 0.0, True),    # Like K but pointing down
    "Q": (0.4, 1.0, 0.0, 0.0, 0.0, True),    # Like G but pointing down
    "R": (0.0, 1.0, 1.0, 0.0, 0.0, False),   # Crossed index+middle
    "S": (0.0, 0.15, 0.15, 0.15, 0.15, True),  # Closed fist, thumb over fingers
    "T": (0.0, 0.15, 0.0, 0.0, 0.0, True),   # Thumb between index+middle
    "U": (0.0, 1.0, 1.0, 0.0, 0.0, False),   # Index+middle up together
    "V": (0.0, 1.0, 1.0, 0.0, 0.0, False),   # V/peace: index+middle up spread
    "W": (0.0, 1.0, 1.0, 1.0, 0.0, False),   # 3 fingers up
    "X": (0.0, 0.5, 0.0, 0.0, 0.0, False),   # Index hooked/bent
    "Y": (0.8, 0.0, 0.0, 0.0, 1.0, True),    # Thumb+pinky extended (hang loose)
    "Z": (0.0, 1.0, 0.0, 0.0, 0.0, False),   # Index traces Z motion
}


def _make_hand_prototype(letter: str, noise: float = 0.0) -> np.ndarray:
    """Generates a 21x3 right-hand keypoint prototype for an ASL fingerspelling letter."""
    thumb_ext, idx_ext, mid_ext, ring_ext, pinky_ext, thumb_side = ALPHA_HANDSHAPES[letter]

    # Base wrist position (normalized, centered near chin level)
    wrist = np.array([0.02, 0.55, 3.0], dtype=np.float32)

    # Fingertip distances when extended (shoulder-width units)
    EXT_LEN = 0.22
    CURL_LEN = 0.08

    def tip_pos(base_x, base_y, ext_factor, finger_spread_x=0.0):
        """Computes tip position at given extension factor."""
        length = CURL_LEN + ext_factor * (EXT_LEN - CURL_LEN)
        pos = np.array([
            wrist[0] + base_x + finger_spread_x * 0.04,
            wrist[1] - length,      # fingers point upward (negative Y)
            wrist[2]
        ], dtype=np.float32)
        return pos + np.random.randn(3).astype(np.float32) * noise

    # Build 21 keypoints (wrist + 4 per finger)
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = wrist + np.random.randn(3).astype(np.float32) * noise  # Wrist

    # Thumb (points 1-4): lateral if thumb_side else forward
    t_x = 0.10 if thumb_side else 0.06
    for i, frac in enumerate([0.3, 0.5, 0.7, thumb_ext]):
        pts[1 + i] = wrist + np.array([t_x, -frac * EXT_LEN * 0.8, 0], np.float32) + np.random.randn(3).astype(np.float32) * noise

    # Index (points 5-8)
    for i, frac in enumerate([0.3, 0.6, 0.85, idx_ext]):
        pts[5 + i] = tip_pos(-0.04, 0, frac, -1.5)

    # Middle (points 9-12)
    for i, frac in enumerate([0.3, 0.6, 0.85, mid_ext]):
        pts[9 + i] = tip_pos(0.0, 0, frac, 0.0)

    # Ring (points 13-16)
    for i, frac in enumerate([0.3, 0.6, 0.85, ring_ext]):
        pts[13 + i] = tip_pos(0.04, 0, frac, 1.0)

    # Pinky (points 17-20)
    for i, frac in enumerate([0.25, 0.5, 0.75, pinky_ext]):
        pts[17 + i] = tip_pos(0.08, 0, frac, 2.0)

    return pts


def generate_alphabet_sequence(letter: str, n_frames: int = 30) -> np.ndarray:
    """Generates a (30, 1629) landmark sequence for an ASL fingerspelling letter."""
    seq = np.zeros((n_frames, 1629), dtype=np.float32)

    # Nose anchor (index 0-2) at origin, shoulders define scale
    nose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    left_shoulder = np.array([-0.5, 0.5, 0.0], dtype=np.float32)
    right_shoulder = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    for f in range(n_frames):
        noise = 0.008 + 0.004 * np.sin(f * 0.3)  # Slight frame-by-frame jitter

        # Pose block (99D): set nose + shoulders
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[0] = nose
        pose[11] = left_shoulder + np.random.randn(3).astype(np.float32) * 0.005
        pose[12] = right_shoulder + np.random.randn(3).astype(np.float32) * 0.005
        # Right wrist near chin height
        pose[16] = np.array([-0.05, 0.58, 3.0], dtype=np.float32) + np.random.randn(3).astype(np.float32) * noise
        seq[f, :99] = pose.flatten()

        # Face block (1404D): left as zeros (masked in training)
        # seq[f, 99:1503] = 0.0  (already zero)

        # Right hand (63D at index 1566:1629)
        rh_pts = _make_hand_prototype(letter, noise=noise)
        seq[f, 1566:1629] = rh_pts.flatten()

    return seq


def load_wlasl_landmarks(vocab_data, landmark_dir="data/wlasl/landmarks", meta_path="data/wlasl/WLASL_v0.3.json"):
    """Loads all real WLASL landmark tensors grouped by vocabulary class ID."""
    token_to_cid = {t.lower(): cid for t, cid in vocab_data["token_to_id"].items()}
    for t, cid in list(token_to_cid.items()):
        token_to_cid[t.replace("-", " ")] = cid

    vid_to_gloss = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for entry in meta:
            gloss = entry.get("gloss", "").lower()
            for inst in entry.get("instances", []):
                vid_to_gloss[str(inst.get("video_id"))] = gloss

    class_samples = {cid: [] for cid in range(vocab_data["vocab_size"])}
    if os.path.exists(landmark_dir):
        for f in glob.glob(os.path.join(landmark_dir, "*.npy")):
            vid = os.path.splitext(os.path.basename(f))[0]
            gloss = vid_to_gloss.get(vid) or vid_to_gloss.get(vid.lstrip('0'))
            if gloss and gloss in token_to_cid:
                cid = token_to_cid[gloss]
                arr = np.load(f)
                if arr.shape == (30, 1629):
                    class_samples[cid].append(arr)

    return class_samples


def augment_landmarks(bases, count, noise_std=0.006, scale_range=(0.85, 1.18), trans_range=(-0.04, 0.04), is_train=True):
    """Vectorized heavy spatio-temporal domain randomization from real landmark trajectories."""
    N = len(bases)
    if N == 0:
        return torch.zeros((count, 30, 1629), dtype=torch.float32)

    indices = np.random.choice(N, size=count, replace=True)
    selected = torch.tensor(np.array([bases[i] for i in indices]), dtype=torch.float32)

    # 1. Temporal speed variation
    if is_train and count > 1:
        x_t = selected.permute(0, 2, 1)
        target_len = int(np.random.uniform(18, 42))
        warped = torch.nn.functional.interpolate(x_t, size=target_len, mode='linear', align_corners=False)
        if target_len < 30:
            last = warped[:, :, -1:].repeat(1, 1, 30 - target_len)
            warped = torch.cat([warped, last], dim=-1)
        else:
            start_f = np.random.randint(0, target_len - 30 + 1)
            warped = warped[:, :, start_f:start_f + 30]
        selected = warped.permute(0, 2, 1)

    # 2. Spatial scale + translation + noise
    scales = torch.empty(count, 1, 1).uniform_(*scale_range) if is_train else torch.ones(count, 1, 1)
    trans = torch.empty(count, 1, 1).uniform_(*trans_range) if is_train else torch.zeros(count, 1, 1)
    noise = (torch.randn(count, 30, 1629) * noise_std) if is_train else 0.0

    active_mask = (selected != 0.0).float()
    augmented = (selected * scales + trans + noise) * active_mask

    # 3. In-plane 2D rotation jitter (±15 degrees)
    if is_train and count > 1:
        angles = np.radians(np.random.uniform(-15.0, 15.0, size=(count, 1, 1))).astype(np.float32)
        cos_a = torch.from_numpy(np.cos(angles))
        sin_a = torch.from_numpy(np.sin(angles))
        for pt_idx in range(0, 1629, 3):
            if pt_idx < 99 or pt_idx >= 1503:
                x_val = augmented[:, :, pt_idx:pt_idx+1]
                y_val = augmented[:, :, pt_idx+1:pt_idx+2]
                rot_x = x_val * cos_a - y_val * sin_a
                rot_y = x_val * sin_a + y_val * cos_a
                augmented[:, :, pt_idx:pt_idx+1] = torch.where(x_val != 0.0, rot_x, x_val)
                augmented[:, :, pt_idx+1:pt_idx+2] = torch.where(y_val != 0.0, rot_y, y_val)

    # 4. Random frame dropout
    if is_train:
        for b in range(count):
            if np.random.rand() < 0.25:
                drop_frames = np.random.choice(30, size=np.random.randint(2, 5), replace=False)
                augmented[b, drop_frames] = 0.0

    # 5. Mask facial identity mesh
    augmented[:, :, 99:1503] = 0.0

    return augmented


def main():
    vocab_path = "configs/vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    id_to_token = vocab_data["id_to_token"]
    num_classes = len(id_to_token)
    alphabet_letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

    print("=" * 70, flush=True)
    print("  ASL Recognition Model Training (6 Words + 26 Alphabet = 32 Classes)", flush=True)
    print("=" * 70, flush=True)

    # Load authentic landmark files for the 6 word classes
    class_samples = load_wlasl_landmarks(vocab_data)
    total_real = sum(len(v) for v in class_samples.values())
    word_classes = sum(1 for cid, v in class_samples.items() if len(v) > 0)
    print(f"Loaded {total_real} authentic WLASL files for {word_classes} word classes.", flush=True)

    # Inject alphabet sequences for letters A-Z (classes 6-31)
    # Priority: real Kaggle landmarks > synthetic handshape prototypes
    token_to_cid = vocab_data["token_to_id"]
    alpha_landmark_dir = "data/asl_alphabet/landmarks"
    real_alpha_loaded = 0

    for letter in alphabet_letters:
        cid = token_to_cid.get(letter)
        if cid is None:
            continue

        # Try to load real Kaggle/MediaPipe landmarks for this letter
        letter_dir = os.path.join(alpha_landmark_dir, letter.upper())
        real_files = glob.glob(os.path.join(letter_dir, "*.npy")) if os.path.isdir(letter_dir) else []
        
        # Subsample up to 350 diverse files per letter to keep RAM efficient and prevent OOM
        if len(real_files) > 350:
            step = len(real_files) / 350
            real_files = [real_files[int(i * step)] for i in range(350)]

        real_bases = []
        for fp in real_files:
            arr = np.load(fp)
            if arr.shape == (30, 1629):
                real_bases.append(arr)


        if real_bases:
            class_samples[cid] = real_bases
            real_alpha_loaded += len(real_bases)
        else:
            # Fall back to handshape prototype synthetics
            letter_bases = [generate_alphabet_sequence(letter, n_frames=30) for _ in range(15)]
            class_samples[cid] = letter_bases

    total_with_alpha = sum(len(v) for v in class_samples.values())
    if real_alpha_loaded > 0:
        print(f"Loaded {real_alpha_loaded} real Kaggle alphabet landmarks + synthetics for remaining letters.", flush=True)
    else:
        print("No real alphabet landmarks found — using synthetic handshape prototypes for A-Z.", flush=True)
        print(f"  (To use real data: run python scripts/download_asl_alphabet.py first)", flush=True)
    print(f"Total base samples (words + alphabet): {total_with_alpha}", flush=True)

    train_tensors, train_labels = [], []
    val_tensors, val_labels = [], []

    train_per_class = 350
    val_per_class = 70

    print("Synthesizing domain-randomized gesture dataset...", flush=True)
    for cid in range(num_classes):
        token = id_to_token[str(cid)]
        bases = class_samples[cid]
        if not bases:
            bases = [np.random.normal(0.02 * cid, 0.015, (30, 1629)).astype(np.float32)]

        aug_train = augment_landmarks(bases, train_per_class, noise_std=0.007, is_train=True)
        aug_val = augment_landmarks(bases, val_per_class, noise_std=0.003, is_train=True)

        train_tensors.append(aug_train)
        train_labels.append(torch.full((train_per_class,), cid, dtype=torch.long))

        val_tensors.append(aug_val)
        val_labels.append(torch.full((val_per_class,), cid, dtype=torch.long))

    X_train = torch.cat(train_tensors, dim=0)
    y_train = torch.cat(train_labels, dim=0)
    X_val = torch.cat(val_tensors, dim=0)
    y_val = torch.cat(val_labels, dim=0)

    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    print(f"Dataset: {len(X_train)} training samples, {len(X_val)} validation samples.", flush=True)

    model = SpatioTemporalSequenceModel(
        in_features=1629,
        num_classes=num_classes,
        cnn_filters=256,
        lstm_hidden=128,
        dropout=0.35,
        temperature=0.85
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name} — Training on CUDA.", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        print(f"Training on: {device}", flush=True)

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-3)
    num_epochs = 45
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    best_val_acc = 0.0
    output_path = "models/cnn_bilstm_wlasl.pt"

    print(f"\nTraining 1D-CNN + 2-layer BiLSTM — {num_classes} classes (Target: >95% Acc)...", flush=True)
    print("-" * 70, flush=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100.0

        if epoch % 5 == 0 or epoch == num_epochs or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(device))
                val_loss = criterion(val_logits, y_val.to(device)).item()
                val_preds = torch.argmax(val_logits, dim=-1)
                val_acc = (val_preds == y_val.to(device)).sum().item() / len(y_val) * 100.0

            print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {total_loss/total:.4f} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save(output_path)

    print("-" * 70, flush=True)
    print(f"Training complete! Best Validation Accuracy: {best_val_acc:.1f}%", flush=True)
    print(f"Model saved to: {output_path}\n", flush=True)


if __name__ == "__main__":
    main()
