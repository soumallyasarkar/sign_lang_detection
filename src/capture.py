"""Module 1: Video Ingestion & Stream Capture (src/capture.py).

Implements a multi-threaded OpenCV video capture pipeline with microsecond
timestamp tracking. Resamples raw camera frames to a uniform 30 FPS,
preventing frame drift and buffering delays.
"""

import time
import threading
import cv2
import numpy as np


class VideoStreamIngestor:
    """Multi-threaded OpenCV video stream ingestor with 30 FPS resampling."""

    def __init__(self, src=0, target_fps=30.0, max_reconnect_attempts=3, mirror=True):
        self.src = src
        self.target_fps = float(target_fps)
        self.frame_interval = 1.0 / self.target_fps
        self.max_reconnect_attempts = max_reconnect_attempts
        self.mirror = bool(mirror)

        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.latest_frame = None
        self.latest_timestamp_us = 0
        self.fps_actual = 0.0
        self.frame_count = 0
        self.dropped_frames = 0
        self.is_synthetic = False

    def _open_capture(self):
        """Attempts to open OpenCV VideoCapture, falls back to synthetic test stream if unavailable."""
        if isinstance(self.src, int) or (isinstance(self.src, str) and (self.src.isdigit() or len(self.src) > 0)):
            try:
                cap = cv2.VideoCapture(int(self.src) if str(self.src).isdigit() else self.src)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    self.is_synthetic = False
                    return cap
            except Exception:
                pass

        # Fallback for headless environments or absent webcams: generate synthetic stream
        self.is_synthetic = True
        return None

    def start(self):
        """Starts the multi-threaded frame acquisition thread."""
        with self.lock:
            if self.running:
                return self
            self.running = True
            self.cap = self._open_capture()
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
            return self

    def _generate_synthetic_frame(self):
        """Generates a synthetic frame for testing or when camera is offline."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :] = (30, 30, 35)
        # Mock silhouette
        cv2.circle(frame, (320, 160), 50, (180, 180, 180), -1)
        cv2.ellipse(frame, (320, 380), (120, 140), 0, 0, 180, (140, 140, 140), -1)
        cv2.putText(frame, f"TEST_STREAM_{self.frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def _capture_worker(self):
        """Background thread grabbing frames and maintaining uniform 30 FPS resampling."""
        last_emit_time = time.perf_counter()
        fps_timer = time.perf_counter()
        counted_frames = 0

        while self.running:
            loop_start = time.perf_counter()
            ret = False
            frame = None

            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Attempt stream recovery
                    for _ in range(self.max_reconnect_attempts):
                        time.sleep(0.05)
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            break
                if ret and frame is not None and self.mirror:
                    frame = cv2.flip(frame, 1)

            if not ret or frame is None:
                frame = self._generate_synthetic_frame()

            now = time.perf_counter()
            elapsed_since_last_emit = now - last_emit_time

            # 30 FPS temporal resampling & drift prevention
            if elapsed_since_last_emit >= self.frame_interval * 0.90:
                with self.lock:
                    self.latest_frame = frame
                    self.latest_timestamp_us = int(now * 1_000_000)
                    self.frame_count += 1
                    counted_frames += 1

                last_emit_time = now
            else:
                with self.lock:
                    self.dropped_frames += 1

            if now - fps_timer >= 1.0:
                with self.lock:
                    self.fps_actual = counted_frames / (now - fps_timer)
                counted_frames = 0
                fps_timer = now

            # Throttle thread loop to prevent 100% CPU utilization
            compute_time = time.perf_counter() - loop_start
            sleep_time = max(0.001, (self.frame_interval / 2.0) - compute_time)
            time.sleep(sleep_time)

    def read(self):
        """Returns the most recent frame and timestamp (timestamp_us, frame)."""
        with self.lock:
            if self.latest_frame is None:
                synth = self._generate_synthetic_frame()
                ts = int(time.perf_counter() * 1_000_000)
                return ts, synth
            return self.latest_timestamp_us, self.latest_frame.copy()

    def stop(self):
        """Stops the ingestion thread and releases camera resources."""
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None and hasattr(self.cap, 'release'):
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def is_running(self):
        """Checks if ingestor is currently active."""
        with self.lock:
            return self.running

    def get_stats(self):
        """Returns runtime performance statistics."""
        with self.lock:
            return {
                "target_fps": self.target_fps,
                "fps_actual": round(self.fps_actual, 2),
                "frame_count": self.frame_count,
                "dropped_frames": self.dropped_frames,
                "is_synthetic": self.is_synthetic,
                "latest_timestamp_us": self.latest_timestamp_us
            }

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
