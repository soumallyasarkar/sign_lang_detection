"""Module 2: Preprocessing & Frame Normalization (src/transform.py).

Standardizes input frames to 224x224 resolution using isotropic scaling with padding,
converts BGR color space to RGB, and normalizes pixel intensities using ImageNet
statistics (mu=[0.485, 0.456, 0.406], sigma=[0.229, 0.224, 0.225]).
"""

import cv2
import numpy as np
import torch


class FrameNormalizer:
    """Preprocesses and normalizes video frames for vision and feature extractors."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, target_size=(224, 224), apply_imagenet_norm=True):
        self.target_size = target_size  # (width, height)
        self.apply_imagenet_norm = apply_imagenet_norm

    def letterbox_resize(self, frame_bgr):
        """Isotropically scales image to target_size with symmetric black padding."""
        h, w = frame_bgr.shape[:2]
        target_w, target_h = self.target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded

    def to_rgb(self, frame_bgr):
        """Converts BGR image to RGB."""
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def normalize(self, frame_rgb):
        """Normalizes RGB uint8 array [0, 255] to float32 using ImageNet statistics."""
        img_float = frame_rgb.astype(np.float32) / 255.0
        if self.apply_imagenet_norm:
            img_norm = (img_float - self.IMAGENET_MEAN) / self.IMAGENET_STD
            return img_norm
        return img_float

    def denormalize(self, norm_img):
        """Inverts ImageNet normalization back to RGB uint8."""
        if self.apply_imagenet_norm:
            img = (norm_img * self.IMAGENET_STD) + self.IMAGENET_MEAN
        else:
            img = norm_img
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def __call__(self, frame_bgr, return_tensor=False):
        """Executes full preprocessing pipeline: Letterbox -> BGR-to-RGB -> Normalization."""
        padded_bgr = self.letterbox_resize(frame_bgr)
        frame_rgb = self.to_rgb(padded_bgr)
        norm_img = self.normalize(frame_rgb)

        if return_tensor:
            # (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
            return tensor, frame_rgb

        return norm_img, frame_rgb
