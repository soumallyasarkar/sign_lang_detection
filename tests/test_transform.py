"""Unit tests for Module 2 (src/transform.py).

Verifies 6 target behaviors:
1. 224x224 output resolution scaling for square inputs
2. Isotropic letterbox scaling preserving aspect ratio with black padding
3. BGR to RGB color conversion
4. ImageNet normalization mean and standard deviation
5. PyTorch tensor output conversion and channel order (C, H, W)
6. Denormalization back to uint8 RGB image
"""

import numpy as np
import torch
import pytest
from src.transform import FrameNormalizer


def test_transform_resolution_scaling():
    """Test 1: Output frame matches standard 224x224 target size."""
    normalizer = FrameNormalizer(target_size=(224, 224))
    raw_bgr = np.zeros((480, 480, 3), dtype=np.uint8)
    norm_img, rgb_img = normalizer(raw_bgr)
    assert norm_img.shape == (224, 224, 3)
    assert rgb_img.shape == (224, 224, 3)


def test_transform_isotropic_letterbox_padding():
    """Test 2: Isotropic letterbox scaling preserves aspect ratio with padding."""
    normalizer = FrameNormalizer(target_size=(224, 224))
    # Wide 16:9 frame
    raw_bgr = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    padded = normalizer.letterbox_resize(raw_bgr)
    assert padded.shape == (224, 224, 3)
    # Check top/bottom black borders
    assert np.all(padded[0, :] == 0)
    assert np.all(padded[-1, :] == 0)


def test_transform_bgr_to_rgb():
    """Test 3: Color space is accurately converted from BGR to RGB."""
    normalizer = FrameNormalizer(target_size=(224, 224))
    bgr_pixel = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_pixel[:, :, 0] = 255  # Blue channel in BGR
    rgb_pixel = normalizer.to_rgb(bgr_pixel)
    assert rgb_pixel[0, 0, 2] == 255  # Blue channel in RGB
    assert rgb_pixel[0, 0, 0] == 0


def test_transform_imagenet_normalization():
    """Test 4: Image intensities normalized with ImageNet mu and sigma."""
    normalizer = FrameNormalizer(target_size=(224, 224))
    white_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 255
    norm_img = normalizer.normalize(white_rgb)

    expected_r = (1.0 - 0.485) / 0.229
    expected_g = (1.0 - 0.456) / 0.224
    expected_b = (1.0 - 0.406) / 0.225

    assert norm_img[0, 0, 0] == pytest.approx(expected_r, rel=1e-3)
    assert norm_img[0, 0, 1] == pytest.approx(expected_g, rel=1e-3)
    assert norm_img[0, 0, 2] == pytest.approx(expected_b, rel=1e-3)


def test_transform_pytorch_tensor_output():
    """Test 5: Return tensor option converts to PyTorch (C, H, W) format."""
    normalizer = FrameNormalizer(target_size=(224, 224))
    raw_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, rgb = normalizer(raw_bgr, return_tensor=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_transform_denormalize_reversibility():
    """Test 6: Denormalization reconstructs valid uint8 image."""
    normalizer = FrameNormalizer(target_size=(224, 224))
    original_rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    norm = normalizer.normalize(original_rgb)
    reconstructed = normalizer.denormalize(norm)
    assert reconstructed.shape == original_rgb.shape
    assert reconstructed.dtype == np.uint8
    assert np.allclose(original_rgb, reconstructed, atol=2)
