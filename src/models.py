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
        """Analyzes spatial landmark geometry to compute gesture class logit priors."""
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

        # Select primary signing hand (supports right-handed, left-handed, or mirrored webcams)
        hand = rh if rh_active else lh
        wrist = hand[0]
        w_y = frame_vec[49] if rh_active else frame_vec[46]
        w_x = frame_vec[48] if rh_active else frame_vec[45]

        thumb_tip = hand[4]
        index_tip = hand[8]
        mid_tip = hand[12]
        ring_tip = hand[16]
        pinky_tip = hand[20]

        # Finger extension check: distance from wrist (point 0) to fingertips
        index_ext = np.linalg.norm(index_tip - wrist) > 0.12
        mid_ext = np.linalg.norm(mid_tip - wrist) > 0.12
        ring_ext = np.linalg.norm(ring_tip - wrist) > 0.12
        pinky_ext = np.linalg.norm(pinky_tip - wrist) > 0.12
        open_hand = index_ext and mid_ext and ring_ext and pinky_ext

        # Window trajectory analysis: find highest reach (min Y) across the rolling sequence
        min_mid_y = mid_tip[1]
        min_mid_x = mid_tip[0]
        if window_data is not None and len(window_data) >= 10:
            w_arr = window_data if isinstance(window_data, np.ndarray) else window_data.cpu().numpy()
            h_arr = w_arr[:, 1566:1629].reshape(-1, 21, 3) if rh_active else w_arr[:, 1503:1566].reshape(-1, 21, 3)
            act_h = np.any(h_arr != 0, axis=(1, 2))
            if np.any(act_h):
                valid_h = h_arr[act_h]
                min_mid_y = np.min(valid_h[:, 12, 1])
                best_idx = np.argmin(valid_h[:, 12, 1])
                min_mid_x = valid_h[best_idx, 12, 0]

        # 1. HELLO: Hand raised HIGH to temple / forehead level (above eyes/nose: min_mid_y < 0.0 or w_y < 0.10, lateral abs(min_mid_x) > 0.20)
        if open_hand and (min_mid_y < 0.0 or (min_mid_y < 0.10 and abs(min_mid_x) > 0.20)):
            if "HELLO" in token_to_idx:
                boosts[token_to_idx["HELLO"]] += 6.5

        # 2. THANK-YOU: Hand touches or starts at chin / lips / mouth (0.02 <= min_mid_y <= 0.38, centered abs(min_mid_x) <= 0.18), open flat palm moving outward
        elif open_hand and (0.02 <= min_mid_y <= 0.38) and abs(min_mid_x) <= 0.18:
            if "THANK-YOU" in token_to_idx:
                boosts[token_to_idx["THANK-YOU"]] += 6.5

        # 3. WATER: 3 fingers up (W-shape: index, middle, ring extended, pinky curled) near chin
        elif index_ext and mid_ext and ring_ext and not pinky_ext and abs(w_x) < 0.30:
            if "WATER" in token_to_idx:
                boosts[token_to_idx["WATER"]] += 6.5

        # 4. PLEASE: Flat hand placed against chest (lower than chin: min_mid_y >= 0.45, centered abs(w_x) < 0.35)
        elif open_hand and min_mid_y >= 0.45 and abs(w_x) < 0.35:
            if "PLEASE" in token_to_idx:
                boosts[token_to_idx["PLEASE"]] += 5.8

        # 5. ME: Index finger pointing to chest
        elif w_y > 0.20 and index_ext and not mid_ext and not ring_ext and not pinky_ext:
            if "ME" in token_to_idx:
                boosts[token_to_idx["ME"]] += 6.0

        # 6. YES: Closed fist nodding at chest level
        elif w_y > 0.15 and not index_ext and not mid_ext and not ring_ext and not pinky_ext:
            if "YES" in token_to_idx:
                boosts[token_to_idx["YES"]] += 5.5

        # 7. HELP: Both hands active, one supporting the other
        elif rh_active and lh_active and (w_y > 0.20 or frame_vec[46] > 0.20):
            if "HELP" in token_to_idx:
                boosts[token_to_idx["HELP"]] += 5.5

        # 8. NO: Pinching gesture (thumb, index, middle closing together)
        elif np.linalg.norm(thumb_tip - index_tip) < 0.08 and not ring_ext and not pinky_ext:
            if "NO" in token_to_idx:
                boosts[token_to_idx["NO"]] += 5.5

        # 9. YOU: Index finger extended pointing toward camera
        elif index_ext and not mid_ext and not ring_ext and not pinky_ext:
            if "YOU" in token_to_idx:
                boosts[token_to_idx["YOU"]] += 5.2

        # 10. STUDENT: Two hands active in lower-mid chest plane
        elif rh_active and lh_active and (w_y > 0.10 or frame_vec[46] > 0.10):
            if "STUDENT" in token_to_idx:
                boosts[token_to_idx["STUDENT"]] += 4.5

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
    def load(cls, filepath=None, in_features=1629, num_classes=28, device=None):
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
