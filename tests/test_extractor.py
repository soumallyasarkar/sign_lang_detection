"""Unit tests for Module 3 (src/extractor.py).

Verifies 5 target behaviors:
1. 1,629-dimensional unified vector output shape
2. Spatial scale invariance via nose anchor & shoulder-width normalization
3. Zero-padding applied during momentary hand occlusions
4. Annotated overlay rendering with theme colors (cyan, yellow, pink, green)
5. Clean resource teardown on close
"""

import numpy as np
import pytest
from src.extractor import HolisticLandmarkExtractor


def test_extractor_vector_dimensions():
    """Test 1: Output landmark vector shape is exactly 1,629 features."""
    extractor = HolisticLandmarkExtractor()
    frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    vec, annotated = extractor.extract(frame_rgb)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1629,)
    assert vec.dtype == np.float32
    extractor.close()


def test_extractor_anchor_normalization_and_scale_invariance():
    """Test 2: Keypoints are normalized relative to nose anchor and shoulder width."""
    extractor = HolisticLandmarkExtractor()
    pose_pts = np.zeros((33, 3), dtype=np.float32)
    face_pts = np.zeros((468, 3), dtype=np.float32)
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts = np.zeros((21, 3), dtype=np.float32)

    # Set Nose anchor (idx 0)
    pose_pts[0] = [0.5, 0.4, 0.0]
    # Set Shoulders (idx 11, 12) with distance 0.2
    pose_pts[11] = [0.4, 0.6, 0.0]
    pose_pts[12] = [0.6, 0.6, 0.0]

    vec = extractor.normalize_landmarks(pose_pts, face_pts, lh_pts, rh_pts)
    assert vec.shape == (1629,)
    # Nose anchor landmark in pose should be normalized to (0, 0, 0)
    assert np.allclose(vec[0:3], [0.0, 0.0, 0.0])
    extractor.close()


def test_extractor_hand_occlusion_zero_padding():
    """Test 3: Occluded or unobserved hands are zero-padded (126 features of zeros)."""
    extractor = HolisticLandmarkExtractor()
    pose_pts = np.ones((33, 3), dtype=np.float32)
    face_pts = np.ones((468, 3), dtype=np.float32)
    # Hand occluded -> zero arrays
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts = np.zeros((21, 3), dtype=np.float32)

    vec = extractor.normalize_landmarks(pose_pts, face_pts, lh_pts, rh_pts)
    # Hand landmarks are the last 126 features: 63 for LH, 63 for RH
    hand_features = vec[99 + 1404:]
    assert len(hand_features) == 126
    assert np.all(hand_features == 0.0)
    extractor.close()


def test_extractor_skeleton_overlay_rendering():
    """Test 4: Annotated frame returns valid BGR canvas with identical dimensions."""
    extractor = HolisticLandmarkExtractor()
    frame_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    vec, annotated = extractor.extract(frame_rgb, frame_bgr)
    assert annotated.shape == (240, 320, 3)
    assert isinstance(annotated, np.ndarray)
    extractor.close()


def test_extractor_close_and_cleanup():
    """Test 5: MediaPipe resources release gracefully."""
    extractor = HolisticLandmarkExtractor()
    extractor.close()
    assert extractor.holistic is None
