"""Module 3: MediaPipe Holistic Landmark Extractor (src/extractor.py).

Extracts a unified 1,629-dimensional spatial vector per frame:
- 33 body pose landmarks (shoulders, elbows, wrists - 99D)
- 468 facial mesh coordinates (lip motion, head pose, eye gaze - 1,404D)
- Dual-hand keypoints (21 per hand - 126D)
Total: 99 + 1404 + 126 = 1629D.

Features are normalized relative to shoulder-width and nose anchors for spatial
scale invariance, with zero-padding applied during momentary hand occlusions.
"""

import math
import cv2
import numpy as np


class HolisticLandmarkExtractor:
    """Extracts and normalizes 1629D holistic spatial landmarks from RGB frames."""

    NUM_POSE = 33      # 33 * 3 = 99
    NUM_FACE = 468     # 468 * 3 = 1404
    NUM_HAND = 21      # 21 * 3 = 63 (per hand, total 126)
    TOTAL_DIMS = 1629

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.holistic = None
        self._init_holistic()

    def _init_holistic(self):
        """Initializes MediaPipe Holistic solution."""
        try:
            import mediapipe as mp
            solutions = getattr(mp, 'solutions', None)
            if solutions is None:
                try:
                    import mediapipe.python.solutions as solutions
                except ImportError:
                    pass

            if solutions and hasattr(solutions, 'holistic'):
                self.mp_holistic = solutions.holistic
                self.mp_drawing = getattr(solutions, 'drawing_utils', None)
                self.mp_drawing_styles = getattr(solutions, 'drawing_styles', None)
                self.holistic = self.mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
            else:
                self.holistic = None
        except Exception:
            self.holistic = None

    def _extract_raw_points(self, landmark_list, count):
        """Extracts (x, y, z) tuples from landmark list or returns zero array if None."""
        if not landmark_list or not hasattr(landmark_list, 'landmark'):
            return np.zeros((count, 3), dtype=np.float32)

        pts = []
        for i in range(min(count, len(landmark_list.landmark))):
            lm = landmark_list.landmark[i]
            pts.append([lm.x, lm.y, lm.z])

        while len(pts) < count:
            pts.append([0.0, 0.0, 0.0])

        return np.array(pts, dtype=np.float32)

    def normalize_landmarks(self, pose_pts, face_pts, left_hand_pts, right_hand_pts):
        """Normalizes spatial landmarks relative to nose anchor and shoulder-width scale.

        Scale invariance calculation:
        - Anchor: Nose (pose index 0)
        - Scale: Euclidean distance between Left Shoulder (idx 11) and Right Shoulder (idx 12)
        """
        # Landmark 0: Nose, 11: Left Shoulder, 12: Right Shoulder
        nose = pose_pts[0].copy()
        left_shoulder = pose_pts[11]
        right_shoulder = pose_pts[12]

        shoulder_dist = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        scale = shoulder_dist if shoulder_dist > 0.05 else 1.0

        def norm_group(pts):
            # Zero-padded points remain zero
            is_zero = np.all(pts == 0.0, axis=-1, keepdims=True)
            normed = (pts - nose) / scale
            return np.where(is_zero, 0.0, normed)

        norm_pose = norm_group(pose_pts)
        norm_face = norm_group(face_pts)
        norm_lh = norm_group(left_hand_pts)
        norm_rh = norm_group(right_hand_pts)

        # Flatten into 1629D vector
        vec = np.concatenate([
            norm_pose.flatten(),       # 99D
            norm_face.flatten(),       # 1404D
            norm_lh.flatten(),         # 63D
            norm_rh.flatten()          # 63D
        ], axis=0).astype(np.float32)

        return vec

    def draw_skeleton_overlay(self, frame_bgr, results):
        """Renders skeletal landmarks onto frame with theme colors:

        - Pose bones: Cyan (255, 255, 0)
        - Face mesh: Yellow (0, 255, 255)
        - Left hand: Pink (203, 192, 255)
        - Right hand: Light Green (0, 255, 127)
        """
        annotated = frame_bgr.copy()
        if results is None:
            return annotated

        h, w = annotated.shape[:2]

        # Draw Face Mesh contours (Yellow)
        if getattr(results, 'face_landmarks', None):
            for lm in results.face_landmarks.landmark[::12]:  # sparse for efficiency
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 255), -1)

        # Draw Upper Body Pose (Cyan)
        if getattr(results, 'pose_landmarks', None):
            pose_points = {}
            for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24]:
                if idx < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[idx]
                    pt = (int(lm.x * w), int(lm.y * h))
                    pose_points[idx] = pt
                    cv2.circle(annotated, pt, 4, (255, 255, 0), -1)

            # Connections: shoulders, arms
            connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24)]
            for p1, p2 in connections:
                if p1 in pose_points and p2 in pose_points:
                    cv2.line(annotated, pose_points[p1], pose_points[p2], (255, 255, 0), 2)

        # Draw Left Hand (Pink)
        if getattr(results, 'left_hand_landmarks', None):
            for lm in results.left_hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 3, (203, 192, 255), -1)

        # Draw Right Hand (Light Green)
        if getattr(results, 'right_hand_landmarks', None):
            for lm in results.right_hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 3, (0, 255, 127), -1)

        return annotated

    def extract(self, frame_rgb, frame_bgr_for_vis=None):
        """Processes RGB frame and returns (vec_1629d, annotated_bgr)."""
        if frame_bgr_for_vis is None:
            frame_bgr_for_vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        results = None
        if self.holistic is not None:
            try:
                results = self.holistic.process(frame_rgb)
            except Exception:
                results = None

        if results is not None:
            pose_pts = self._extract_raw_points(getattr(results, 'pose_landmarks', None), self.NUM_POSE)
            face_pts = self._extract_raw_points(getattr(results, 'face_landmarks', None), self.NUM_FACE)
            lh_pts = self._extract_raw_points(getattr(results, 'left_hand_landmarks', None), self.NUM_HAND)
            rh_pts = self._extract_raw_points(getattr(results, 'right_hand_landmarks', None), self.NUM_HAND)
            annotated_frame = self.draw_skeleton_overlay(frame_bgr_for_vis, results)
        else:
            pose_pts = np.zeros((self.NUM_POSE, 3), dtype=np.float32)
            face_pts = np.zeros((self.NUM_FACE, 3), dtype=np.float32)
            lh_pts = np.zeros((self.NUM_HAND, 3), dtype=np.float32)
            rh_pts = np.zeros((self.NUM_HAND, 3), dtype=np.float32)
            annotated_frame = frame_bgr_for_vis.copy()

        vec_1629d = self.normalize_landmarks(pose_pts, face_pts, lh_pts, rh_pts)
        return vec_1629d, annotated_frame

    def close(self):
        """Releases MediaPipe resources."""
        if self.holistic is not None and hasattr(self.holistic, 'close'):
            try:
                self.holistic.close()
            except Exception:
                pass
            self.holistic = None
