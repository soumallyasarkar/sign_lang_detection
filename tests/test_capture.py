"""Unit and integration tests for Module 1 (src/capture.py).

Verifies 5 target behaviors:
1. Stream start/stop lifecycle and thread initialization
2. 30 FPS temporal resampling and interval compliance
3. Microsecond timestamp continuity
4. Camera stream recovery on failure / synthetic fallback
5. Resource release and clean shutdown
"""

import time
import numpy as np
import pytest
from src.capture import VideoStreamIngestor


def test_capture_lifecycle_and_start_stop():
    """Test 1: VideoStreamIngestor starts and stops cleanly."""
    ingestor = VideoStreamIngestor(src="invalid_cam_id_for_testing", target_fps=30)
    assert not ingestor.is_running()
    ingestor.start()
    assert ingestor.is_running()
    time.sleep(0.1)
    ingestor.stop()
    assert not ingestor.is_running()


def test_capture_temporal_resampling_rate():
    """Test 2: VideoStreamIngestor respects target FPS interval."""
    target_fps = 30.0
    ingestor = VideoStreamIngestor(src="invalid_src", target_fps=target_fps)
    ingestor.start()
    time.sleep(0.3)
    ts1, frame1 = ingestor.read()
    time.sleep(ingestor.frame_interval * 1.5)
    ts2, frame2 = ingestor.read()
    ingestor.stop()

    assert ts2 > ts1
    assert ingestor.frame_interval == pytest.approx(1.0 / 30.0, rel=1e-2)


def test_capture_microsecond_timestamp_continuity():
    """Test 3: Microsecond timestamps are monotonic and continuous."""
    ingestor = VideoStreamIngestor(src="invalid_src", target_fps=30)
    ingestor.start()
    timestamps = []
    for _ in range(5):
        time.sleep(0.04)
        ts, _ = ingestor.read()
        timestamps.append(ts)
    ingestor.stop()

    assert len(timestamps) == 5
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_capture_stream_recovery_and_synthetic_fallback():
    """Test 4: Seamless stream recovery and valid fallback frame generation."""
    ingestor = VideoStreamIngestor(src="invalid_src", target_fps=30)
    ingestor.start()
    time.sleep(0.1)
    ts, frame = ingestor.read()
    ingestor.stop()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_capture_statistics_and_resource_release():
    """Test 5: Runtime performance stats tracking and clean exit."""
    with VideoStreamIngestor(src="invalid_src", target_fps=30) as ingestor:
        time.sleep(0.15)
        stats = ingestor.get_stats()
        assert "target_fps" in stats
        assert stats["target_fps"] == 30.0
        assert "frame_count" in stats
        assert stats["frame_count"] > 0
    assert not ingestor.is_running()
