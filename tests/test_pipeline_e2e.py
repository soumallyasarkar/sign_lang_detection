"""End-to-end integration tests for Continuous Interpretation Pipeline (src/pipeline.py).

Verifies 3 target behaviors:
1. End-to-end per-frame latency complies with real-time target (<250 ms)
2. Synchronized outputs across vision canvas, confidence HUD, and language translation
3. Graceful pipeline initialization, execution, and resource teardown
"""

import time
import numpy as np
import pytest
from src.pipeline import ContinuousInterpretationPipeline


def test_pipeline_lifecycle_and_teardown():
    """Test 1: Full pipeline initializes and stops cleanly without resource leaks."""
    pipeline = ContinuousInterpretationPipeline()
    pipeline.start()
    assert pipeline.capture.is_running()
    pipeline.stop()
    assert not pipeline.capture.is_running()


def test_pipeline_step_synchronization():
    """Test 2: Pipeline step returns synchronized annotated frame, sign, conf, sentence."""
    pipeline = ContinuousInterpretationPipeline()
    pipeline.start()
    try:
        annotated_frame, emitted_sign, conf, sentence = pipeline.step()
        assert annotated_frame is not None
        assert isinstance(annotated_frame, np.ndarray)
        assert isinstance(emitted_sign, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
        assert isinstance(sentence, str)
    finally:
        pipeline.stop()


def test_pipeline_e2e_per_frame_latency():
    """Test 3: Per-frame processing latency meets real-time budget (< 250 ms)."""
    pipeline = ContinuousInterpretationPipeline()
    pipeline.start()
    try:
        # Warmup frame
        pipeline.step()

        t0 = time.perf_counter()
        annotated_frame, emitted_sign, conf, sentence = pipeline.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Must execute within real-time budget (< 250 ms target on CPU)
        assert elapsed_ms < 250.0, f"Latency {elapsed_ms:.1f}ms exceeds 250ms target"
    finally:
        pipeline.stop()
