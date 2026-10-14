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

    def __init__(self, min_detection_confidence=0.35, min_tracking_confidence=0.35, smoothing_alpha=0.65):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_alpha = smoothing_alpha
        self.holistic = None
        self.hands = None
        
        # Temporal smoothing & anti-flicker persistence states
        self.prev_lh = None
        self.prev_rh = None
        self.lh_lost_count = 0
        self.rh_lost_count = 0
        self.max_lost_persistence = 3

        # Standard 21-keypoint hand bone connections
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17)                              # Palm base
        ]
        self._init_holistic()

    def _init_holistic(self):
        """Initializes MediaPipe Holistic solution and dual-mode Hands fallback."""
        try:
            import mediapipe as mp
            solutions = getattr(mp, 'solutions', None)
            if solutions is None:
                try:
                    import mediapipe.python.solutions as solutions
                except ImportError:
                    pass

            if solutions:
                if hasattr(solutions, 'holistic'):
                    self.mp_holistic = solutions.holistic
                    self.holistic = self.mp_holistic.Holistic(
                        static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=self.min_detection_confidence,
                        min_tracking_confidence=self.min_tracking_confidence
                    )
                if hasattr(solutions, 'hands'):
                    self.mp_hands = solutions.hands
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=self.min_detection_confidence,
                        min_tracking_confidence=self.min_tracking_confidence
                    )
        except Exception:
            self.holistic = None
            self.hands = None

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
        nose = pose_pts[0].copy()
        left_shoulder = pose_pts[11]
        right_shoulder = pose_pts[12]

        shoulder_dist = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        scale = shoulder_dist if shoulder_dist > 0.05 else 1.0

        def norm_group(pts):
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

    def _smooth_hand(self, raw_pts, prev_pts, lost_count):
        """Applies Exponential Moving Average smoothing and persistence to hand keypoints."""
        is_active = not np.all(raw_pts == 0.0)
        if is_active:
            if prev_pts is not None:
                smoothed = (self.smoothing_alpha * raw_pts + (1.0 - self.smoothing_alpha) * prev_pts).astype(np.float32)
            else:
                smoothed = raw_pts.copy()
            return smoothed, smoothed.copy(), 0
        else:
            if prev_pts is not None and lost_count < self.max_lost_persistence:
                # Retain previous position with slight decay during momentary tracking dropout
                decayed = (prev_pts * 0.98).astype(np.float32)
                return decayed, prev_pts, lost_count + 1
            else:
                return np.zeros((self.NUM_HAND, 3), dtype=np.float32), None, lost_count + 1

    def draw_skeleton_overlay(self, frame_bgr, results, lh_pts=None, rh_pts=None):
        """Renders skeletal landmarks and bones onto frame with theme colors:

        - Pose bones: Cyan (255, 255, 0)
        - Face mesh: Yellow (0, 255, 255)
        - Left hand: Pink (203, 192, 255) with pink bones
        - Right hand: Light Green (0, 255, 127) with green bones
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw Face Mesh contours (Yellow)
        if results and getattr(results, 'face_landmarks', None):
            for lm in results.face_landmarks.landmark[::12]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 255), -1)

        # Draw Upper Body Pose (Cyan)
        if results and getattr(results, 'pose_landmarks', None):
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

        def draw_hand_mesh(pts, joint_color, bone_color):
            if pts is None or np.all(pts == 0.0):
                return
            px_pts = []
            for pt in pts:
                px_pts.append((int(pt[0] * w), int(pt[1] * h)))

            # Draw bones
            for p1, p2 in self.HAND_CONNECTIONS:
                if p1 < len(px_pts) and p2 < len(px_pts):
                    cv2.line(annotated, px_pts[p1], px_pts[p2], bone_color, 2, cv2.LINE_AA)

            # Draw joints
            for p in px_pts:
                cv2.circle(annotated, p, 3, joint_color, -1, cv2.LINE_AA)

        # Draw Left Hand (Pink joints + magenta bones)
        draw_hand_mesh(lh_pts, (203, 192, 255), (180, 105, 255))

        # Draw Right Hand (Light Green joints + emerald bones)
        draw_hand_mesh(rh_pts, (0, 255, 127), (0, 200, 100))

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
        else:
            pose_pts = np.zeros((self.NUM_POSE, 3), dtype=np.float32)
            face_pts = np.zeros((self.NUM_FACE, 3), dtype=np.float32)
            lh_pts = np.zeros((self.NUM_HAND, 3), dtype=np.float32)
            rh_pts = np.zeros((self.NUM_HAND, 3), dtype=np.float32)

        # Dual-mode Fallback: If Holistic lost hands, run dedicated Hands detector
        rh_missing = np.all(rh_pts == 0.0)
        lh_missing = np.all(lh_pts == 0.0)
        if (rh_missing or lh_missing) and self.hands is not None:
            try:
                hands_res = self.hands.process(frame_rgb)
                if hands_res and hands_res.multi_hand_landmarks:
                    for idx, hand_lms in enumerate(hands_res.multi_hand_landmarks):
                        handedness_label = "Right"
                        if hasattr(hands_res, 'multi_handedness') and idx < len(hands_res.multi_handedness):
                            classification = hands_res.multi_handedness[idx].classification
                            if classification:
                                handedness_label = classification[0].label

                        hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                        # In mirrored webcam mode, 'Right' classification corresponds to right hand
                        if handedness_label == "Right" and rh_missing:
                            rh_pts = hand_coords
                            rh_missing = False
                        elif handedness_label == "Left" and lh_missing:
                            lh_pts = hand_coords
                            lh_missing = False
                        elif rh_missing:
                            rh_pts = hand_coords
                            rh_missing = False
            except Exception:
                pass

        # Apply Temporal EMA Smoothing & Anti-flicker Persistence
        lh_pts, self.prev_lh, self.lh_lost_count = self._smooth_hand(lh_pts, self.prev_lh, self.lh_lost_count)
        rh_pts, self.prev_rh, self.rh_lost_count = self._smooth_hand(rh_pts, self.prev_rh, self.rh_lost_count)

        annotated_frame = self.draw_skeleton_overlay(frame_bgr_for_vis, results, lh_pts=lh_pts, rh_pts=rh_pts)
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

        if self.hands is not None and hasattr(self.hands, 'close'):
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None

