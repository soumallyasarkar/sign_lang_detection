"""Modules 4 & 5: Spatio-Temporal Sequence Encoder & Dynamic Sign Classifier (src/models.py).

Implements the 1D-CNN + 2-layer Bi-LSTM sequence encoder and multi-layer classification head:
- Input: Rolling temporal window T=30 frames, D=1629 spatial keypoints
- 1D-CNN: 256 filters, kernel size 3 for local temporal dynamics
- 2-layer Bidirectional LSTM: hidden dimension 128 per direction (256 combined), dropout 0.3
- Temporal Average Pooling: aggregates frame embeddings to a 256D summary vector
- Dynamic Classifier Head: LayerNorm, Dropout (p=0.3), Linear projection to 28 class logits
- Softmax calibration: Temperature-calibrated confidence estimation
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatioTemporalSequenceModel(nn.Module):
    """1D-CNN + Bidirectional LSTM Spatio-Temporal Sign Language Recognition Model."""

    def __init__(self, in_features=1629, num_classes=28, cnn_filters=256,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3, temperature=1.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.temperature = float(temperature)

        # Module 4: Input Normalization & 1D-CNN Temporal Projection
        self.input_norm = nn.LayerNorm(in_features)
        # Input to Conv1d: (Batch, Channels=in_features, Time=T)
        self.conv1d = nn.Conv1d(
            in_channels=in_features,
            out_channels=cnn_filters,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()

        # Module 4: 2-layer Bidirectional LSTM
        # Input to LSTM: (Batch, Time=T, Features=cnn_filters)
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Output feature dimension from BiLSTM (128 * 2 = 256)
        embedding_dim = lstm_hidden * 2

        # Module 5: Dynamic Sign Classifier Head
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """Forward pass for temporal sequence.

        Args:
            x: Tensor of shape (Batch, T=30, D=1629)
        Returns:
            logits: Tensor of shape (Batch, num_classes=28)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, D)

        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)

        # Mask dense facial identity mesh (99:1503) so model strictly evaluates
        # kinematic hand trajectories and body posture without overfitting to facial morphology
        x_gated = x.clone()
        x_gated[:, :, 99:1503] = 0.0

        # Module 4: Input standardization & Conv1D temporal projection
        x_normed = self.input_norm(x_gated)
        x_conv = x_normed.permute(0, 2, 1)  # (B, 1629, T)
        x_conv = self.relu(self.bn1(self.conv1d(x_conv)))

        # Permute back to sequence: (Batch, Time, Channels)
        x_seq = x_conv.permute(0, 2, 1)  # (B, T, 256)

        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x_seq)  # (B, T, 256)

        # Temporal average pooling across the 30 frames
        temporal_embedding = torch.mean(lstm_out, dim=1)  # (B, 256)

        # Dynamic classifier head
        normed = self.layer_norm(temporal_embedding)
        dropped = self.dropout(normed)
        logits = self.fc(dropped)  # (B, num_classes)

        return logits

    def predict_window(self, window_data, vocab_list=None):
        """Performs inference on a single rolling window of T=30 frames.

        Args:
            window_data: numpy array or torch.Tensor of shape (30, 1629) or (1, 30, 1629)
            vocab_list: Optional list of token names
        Returns:
            predicted_token (or class_id), confidence (0.0 to 1.0)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(window_data, np.ndarray):
                tensor_in = torch.from_numpy(window_data).float()
            else:
                tensor_in = window_data.float()

            device = next(self.parameters()).device
            tensor_in = tensor_in.to(device)
            logits = self.forward(tensor_in)  # (1, 28)

            # Kinematic gesture calibration from latest frame landmarks and window motion
            last_frame = window_data[-1] if isinstance(window_data, np.ndarray) else window_data[-1].cpu().numpy()
            boosts = self._compute_kinematic_priors(last_frame, vocab_list, window_data=window_data)
            if boosts is not None:
                logits = logits + torch.from_numpy(boosts).to(device).unsqueeze(0)

            # Softmax with temperature calibration
            calibrated_logits = logits / max(1e-5, self.temperature)
            probs = F.softmax(calibrated_logits, dim=-1)[0]

            conf, pred_idx = torch.max(probs, dim=-1)
            pred_id = int(pred_idx.item())
            confidence = float(conf.item())

            if vocab_list and pred_id < len(vocab_list):
                return vocab_list[pred_id], confidence
            return pred_id, confidence

    def _compute_kinematic_priors(self, frame_vec, vocab_list, window_data=None):
        """Analyzes spatial landmark geometry to compute gesture class logit priors.
        Supports: 6 ASL words (HELLO, THANK-YOU, PLEASE, MEET, NO, YES)
        and 26 ASL fingerspelling alphabet letters (A-Z).
        """
        if frame_vec is None or len(frame_vec) < 1629 or vocab_list is None:
            return None

        boosts = np.zeros(len(vocab_list), dtype=np.float32)

        # Pose landmarks: Nose=0..3, Left Wrist=45..47, Right Wrist=48..50
        # Right Hand Keypoints: 21 landmarks (x, y, z) starting at index 1566
        # Left Hand Keypoints: 21 landmarks (x, y, z) starting at index 1503
        rh = frame_vec[1566:1629].reshape(-1, 3)
        lh = frame_vec[1503:1566].reshape(-1, 3)

        rh_active = not np.all(rh == 0.0)
        lh_active = not np.all(lh == 0.0)

        if not rh_active and not lh_active:
            return boosts

        token_to_idx = {name: idx for idx, name in enumerate(vocab_list)}

        # Select primary signing hand
        hand = rh if rh_active else lh
        wrist = hand[0]
        w_y = frame_vec[49] if rh_active else frame_vec[46]
        w_x = frame_vec[48] if rh_active else frame_vec[45]

        thumb_tip = hand[4]
        index_tip = hand[8]
        mid_tip   = hand[12]
        ring_tip  = hand[16]
        pinky_tip = hand[20]

        # Thumb intermediate knuckle (MCP) for lateral detection
        thumb_mcp = hand[2]

        # Finger extension: distance from wrist to fingertips
        EXT_THRESH = 0.12
        ALPHA_EXT  = 0.09   # tighter threshold for alphabet letters (smaller motion)
        thumb_ext  = np.linalg.norm(thumb_tip  - wrist) > EXT_THRESH
        index_ext  = np.linalg.norm(index_tip  - wrist) > EXT_THRESH
        mid_ext    = np.linalg.norm(mid_tip    - wrist) > EXT_THRESH
        ring_ext   = np.linalg.norm(ring_tip   - wrist) > EXT_THRESH
        pinky_ext  = np.linalg.norm(pinky_tip  - wrist) > EXT_THRESH
        open_hand  = index_ext and mid_ext and ring_ext and pinky_ext

        # Alpha-scale extensions (smaller thresholds for static handshapes)
        a_thumb = np.linalg.norm(thumb_tip  - wrist) > ALPHA_EXT
        a_index = np.linalg.norm(index_tip  - wrist) > ALPHA_EXT
        a_mid   = np.linalg.norm(mid_tip    - wrist) > ALPHA_EXT
        a_ring  = np.linalg.norm(ring_tip   - wrist) > ALPHA_EXT
        a_pinky = np.linalg.norm(pinky_tip  - wrist) > ALPHA_EXT

        # Inter-finger spread for distinguishing V from U, etc.
        idx_mid_spread = np.linalg.norm(index_tip[:2] - mid_tip[:2])  # XY only
        thumb_idx_dist = np.linalg.norm(thumb_tip - index_tip)
        thumb_mid_dist = np.linalg.norm(thumb_tip - mid_tip)
        thumb_side_x   = abs(thumb_mcp[0] - wrist[0])  # lateral thumb position

        # Window trajectory analysis
        min_mid_y = mid_tip[1]
        min_mid_x = mid_tip[0]
        if window_data is not None and len(window_data) >= 10:
            w_arr = window_data if isinstance(window_data, np.ndarray) else window_data.cpu().numpy()
            h_arr = w_arr[:, 1566:1629].reshape(-1, 21, 3) if rh_active else w_arr[:, 1503:1566].reshape(-1, 21, 3)
            act_h = np.any(h_arr != 0, axis=(1, 2))
            if np.any(act_h):
                valid_h   = h_arr[act_h]
                min_mid_y = np.min(valid_h[:, 12, 1])
                best_idx  = np.argmin(valid_h[:, 12, 1])
                min_mid_x = valid_h[best_idx, 12, 0]

        # ------------------------------------------------------------------ #
        #  WORD PRIORS (6 ASL words)                                          #
        # ------------------------------------------------------------------ #

        # 1. HELLO: Open hand raised to forehead/temple (high + lateral)
        if open_hand and (min_mid_y < 0.0 or (min_mid_y < 0.10 and abs(min_mid_x) > 0.20)):
            if "HELLO" in token_to_idx:
                boosts[token_to_idx["HELLO"]] += 6.5

        # 2. THANK-YOU: Open flat hand at chin/lips, moving outward (centered)
        elif open_hand and (0.02 <= min_mid_y <= 0.38) and abs(min_mid_x) <= 0.18:
            if "THANK-YOU" in token_to_idx:
                boosts[token_to_idx["THANK-YOU"]] += 6.5

        # 3. PLEASE: Flat hand on chest (low: min_mid_y >= 0.45, centered)
        elif open_hand and min_mid_y >= 0.45 and abs(w_x) < 0.35:
            if "PLEASE" in token_to_idx:
                boosts[token_to_idx["PLEASE"]] += 5.8

        # 4. MEET: Two flat hands brought together (both active, upper-mid zone)
        elif rh_active and lh_active and w_y < 0.55:
            if "MEET" in token_to_idx:
                boosts[token_to_idx["MEET"]] += 5.5

        # 5. NO: Pinching tap (thumb+index+middle touching)
        elif thumb_idx_dist < 0.08 and thumb_mid_dist < 0.10 and not ring_ext and not pinky_ext:
            if "NO" in token_to_idx:
                boosts[token_to_idx["NO"]] += 5.5

        # 6. YES: Closed fist with nodding motion (all fingers curled)
        elif w_y > 0.15 and not a_index and not a_mid and not a_ring and not a_pinky:
            if "YES" in token_to_idx:
                boosts[token_to_idx["YES"]] += 5.5

        # ------------------------------------------------------------------ #
        #  ALPHABET PRIORS (A-Z fingerspelling handshape geometry)            #
        #  Each prior applies a moderate boost (+3.5) rather than overriding  #
        #  the neural network — these guide disambiguation of similar shapes. #
        # ------------------------------------------------------------------ #
        ALPHA_BOOST = 3.5

        # A: Closed fist, thumb resting on side (no fingers extended, thumb lateral)
        if not a_index and not a_mid and not a_ring and not a_pinky and thumb_side_x > 0.06:
            if "A" in token_to_idx:
                boosts[token_to_idx["A"]] += ALPHA_BOOST

        # B: Four fingers extended straight up, thumb tucked (open hand, thumb in)
        if a_index and a_mid and a_ring and a_pinky and thumb_side_x < 0.05:
            if "B" in token_to_idx:
                boosts[token_to_idx["B"]] += ALPHA_BOOST

        # C: All fingers partially curved into C-arc (moderate extension)
        if all(0.06 < np.linalg.norm(hand[tip] - wrist) < 0.16 for tip in [4, 8, 12, 16, 20]):
            if "C" in token_to_idx:
                boosts[token_to_idx["C"]] += ALPHA_BOOST

        # D: Index up, thumb touches middle finger (ring+pinky closed)
        if a_index and not a_mid and not a_ring and not a_pinky and thumb_mid_dist < 0.10:
            if "D" in token_to_idx:
                boosts[token_to_idx["D"]] += ALPHA_BOOST

        # E: All fingers curled/hooked close to palm (all short distances)
        if all(np.linalg.norm(hand[tip] - wrist) < 0.10 for tip in [4, 8, 12, 16, 20]):
            if "E" in token_to_idx:
                boosts[token_to_idx["E"]] += ALPHA_BOOST

        # F: Thumb + index touch (OK sign), middle+ring+pinky extended
        if thumb_idx_dist < 0.08 and a_mid and a_ring and a_pinky:
            if "F" in token_to_idx:
                boosts[token_to_idx["F"]] += ALPHA_BOOST

        # G: Index + thumb pointing sideways (gun shape)
        if a_index and not a_mid and not a_ring and not a_pinky and thumb_side_x > 0.08:
            if "G" in token_to_idx:
                boosts[token_to_idx["G"]] += ALPHA_BOOST

        # H: Index + middle extended horizontally (together, flat)
        if a_index and a_mid and not a_ring and not a_pinky and idx_mid_spread < 0.05:
            if "H" in token_to_idx:
                boosts[token_to_idx["H"]] += ALPHA_BOOST

        # I: Pinky only extended (all others closed)
        if not a_index and not a_mid and not a_ring and a_pinky:
            if "I" in token_to_idx:
                boosts[token_to_idx["I"]] += ALPHA_BOOST

        # J: Like I (pinky only) — model handles trajectory difference
        if not a_index and not a_mid and not a_ring and a_pinky:
            if "J" in token_to_idx:
                boosts[token_to_idx["J"]] += ALPHA_BOOST * 0.8  # slight de-priority vs I

        # K: Index + middle up, thumb sticking out between them
        if a_index and a_mid and not a_ring and not a_pinky and thumb_side_x > 0.06:
            if "K" in token_to_idx:
                boosts[token_to_idx["K"]] += ALPHA_BOOST

        # L: Index + thumb extended laterally (L-shape)
        if a_index and not a_mid and not a_ring and not a_pinky and thumb_side_x > 0.10:
            if "L" in token_to_idx:
                boosts[token_to_idx["L"]] += ALPHA_BOOST

        # M: Three fingers folded over thumb (index+mid+ring close, low extension)
        if all(np.linalg.norm(hand[tip] - wrist) < 0.12 for tip in [8, 12, 16]) and not a_pinky:
            if "M" in token_to_idx:
                boosts[token_to_idx["M"]] += ALPHA_BOOST

        # N: Two fingers (index+mid) folded over thumb
        if all(np.linalg.norm(hand[tip] - wrist) < 0.12 for tip in [8, 12]) and not a_ring and not a_pinky:
            if "N" in token_to_idx:
                boosts[token_to_idx["N"]] += ALPHA_BOOST

        # O: All fingertips meeting thumb tip to form an O shape
        if all(np.linalg.norm(hand[tip] - thumb_tip) < 0.10 for tip in [8, 12, 16, 20]):
            if "O" in token_to_idx:
                boosts[token_to_idx["O"]] += ALPHA_BOOST

        # P: Like K but hand pointing downward (wrist high, fingertips low)
        if a_index and a_mid and not a_ring and not a_pinky and w_y < 0.40:
            if "P" in token_to_idx:
                boosts[token_to_idx["P"]] += ALPHA_BOOST

        # Q: Like G but pointing downward
        if a_index and not a_mid and not a_ring and not a_pinky and thumb_side_x > 0.08 and w_y < 0.40:
            if "Q" in token_to_idx:
                boosts[token_to_idx["Q"]] += ALPHA_BOOST

        # R: Index + middle crossed (together and touching)
        if a_index and a_mid and not a_ring and not a_pinky and idx_mid_spread < 0.04:
            if "R" in token_to_idx:
                boosts[token_to_idx["R"]] += ALPHA_BOOST

        # S: Closed fist, thumb over fingers (very compact)
        if not a_index and not a_mid and not a_ring and not a_pinky and thumb_idx_dist < 0.10:
            if "S" in token_to_idx:
                boosts[token_to_idx["S"]] += ALPHA_BOOST

        # T: Thumb inserted between index and middle
        if not a_index and not a_mid and not a_ring and not a_pinky and thumb_side_x < 0.05 and thumb_idx_dist < 0.12:
            if "T" in token_to_idx:
                boosts[token_to_idx["T"]] += ALPHA_BOOST

        # U: Index + middle extended together (side by side, small spread)
        if a_index and a_mid and not a_ring and not a_pinky and idx_mid_spread < 0.06:
            if "U" in token_to_idx:
                boosts[token_to_idx["U"]] += ALPHA_BOOST

        # V: Index + middle spread apart (peace/victory sign)
        if a_index and a_mid and not a_ring and not a_pinky and idx_mid_spread > 0.07:
            if "V" in token_to_idx:
                boosts[token_to_idx["V"]] += ALPHA_BOOST

        # W: Index + middle + ring extended, pinky closed
        if a_index and a_mid and a_ring and not a_pinky:
            if "W" in token_to_idx:
                boosts[token_to_idx["W"]] += ALPHA_BOOST

        # X: Index finger hooked/bent (partial extension)
        if 0.07 < np.linalg.norm(index_tip - wrist) < 0.13 and not a_mid and not a_ring and not a_pinky:
            if "X" in token_to_idx:
                boosts[token_to_idx["X"]] += ALPHA_BOOST

        # Y: Thumb + pinky extended (hang-loose/shaka)
        if a_thumb and not a_index and not a_mid and not a_ring and a_pinky:
            if "Y" in token_to_idx:
                boosts[token_to_idx["Y"]] += ALPHA_BOOST

        # Z: Index extended (like pointing), model handles Z-trace trajectory
        if a_index and not a_mid and not a_ring and not a_pinky and not thumb_side_x > 0.08:
            if "Z" in token_to_idx:
                boosts[token_to_idx["Z"]] += ALPHA_BOOST * 0.8  # slight de-priority vs D/L

        return boosts

    def save(self, filepath):
        """Saves model weights."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'in_features': self.in_features,
            'num_classes': self.num_classes,
            'temperature': self.temperature
        }, filepath)

    @classmethod
    def load(cls, filepath=None, in_features=1629, num_classes=32, device=None):
        """Loads model from checkpoint onto specified device or CPU fallback."""
        if device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device is None:
            device = "cpu"
        model = cls(in_features=in_features, num_classes=num_classes)
        if filepath and os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        return model
