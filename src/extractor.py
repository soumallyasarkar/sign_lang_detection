"""Module 3: MediaPipe Holistic Landmark Extractor (src/extractor.py).

Extracts a unified 1,629-dimensional spatial vector per frame:
- 33 body pose landmarks (shoulders, elbows, wrists - 99D)
- 468 facial mesh coordinates (lip motion, head pose, eye gaze - 1,404D)
- Dual-hand keypoints (21 per hand - 126D)
Total: 99 + 1404 + 126 = 1629D.

Features are normalized relative to shoulder-width and nose anchors for spatial
scale invariance, with zero-padding applied during momentary hand occlusions.
Includes cooperative multi-hand tracking, anti-swap hysteresis, and false-positive
rejection to prevent confusion between hands, face, and background textures.
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

    def __init__(self, min_detection_confidence=0.50, min_tracking_confidence=0.50, smoothing_alpha=0.45):
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
        self.max_lost_persistence = 3  # Smoothly bridge momentary 1-3 frame tracking drops

        # Stable anchor tracking
        self.stable_nose = None
        self.stable_scale = None

        # Standard 21-keypoint hand bone connections
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
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
                        model_complexity=1,
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

    def _is_valid_hand(self, hand_pts):
        """Checks if detected hand points represent a plausible physical hand.

        Filters out false detections (noise on ears, face, shirt collars).
        """
        if hand_pts is None or np.all(hand_pts == 0.0):
            return False
        # Distance from wrist (0) to middle finger MCP (9) represents palm size
        palm_len = float(np.linalg.norm(hand_pts[0, :2] - hand_pts[9, :2]))
        # Hand must have reasonable palm scale in normalized image coordinates (0.025 to 0.55)
        if palm_len < 0.025 or palm_len > 0.55:
            return False
        # Check that finger points are not collapsed to a single point
        std_dev = float(np.std(hand_pts[:, :2]))
        if std_dev < 0.008:
            return False
        return True

    def _assign_hands(self, candidate_hands, pose_pts, is_mirrored=True):
        """Assigns detected candidate hands to Left Hand and Right Hand slots.

        Prevents hand-swapping by fusing:
        1. Spatial body midline (derived from shoulders or nose)
        2. Pose wrist keypoint proximity (pose idx 15 left wrist, idx 16 right wrist)
        3. Temporal continuity with previously tracked hand positions
        4. Detector classification priors
        """
        # Determine body midline X
        midline_x = 0.5
        if not np.all(pose_pts[11] == 0.0) and not np.all(pose_pts[12] == 0.0):
            midline_x = float((pose_pts[11, 0] + pose_pts[12, 0]) / 2.0)
        elif not np.all(pose_pts[0] == 0.0):
            midline_x = float(pose_pts[0, 0])

        # Pose wrists in 2D
        lw_pose = pose_pts[15, :2] if not np.all(pose_pts[15] == 0.0) else None
        rw_pose = pose_pts[16, :2] if not np.all(pose_pts[16] == 0.0) else None

        # Filter valid candidates
        valid_candidates = []
        for pts, label, score in candidate_hands:
            if self._is_valid_hand(pts):
                valid_candidates.append((pts, label, score))

        if not valid_candidates:
            return (
                np.zeros((self.NUM_HAND, 3), dtype=np.float32),
                np.zeros((self.NUM_HAND, 3), dtype=np.float32)
            )

        def score_hand_as_right(pts, label):
            """Computes score (>0 prefers Right Hand, <0 prefers Left Hand)."""
            w_pt = pts[0, :2]
            score = 0.0

            # 1. Lateral position relative to body midline
            # In mirrored/selfie view: right hand is on right side of screen (x > midline)
            # In unmirrored view: right hand is on left side of screen (x < midline)
            lateral_offset = (w_pt[0] - midline_x) if is_mirrored else (midline_x - w_pt[0])
            score += lateral_offset * 4.0

            # 2. Pose wrist proximity
            if rw_pose is not None and lw_pose is not None:
                d_rw = float(np.linalg.norm(w_pt - rw_pose))
                d_lw = float(np.linalg.norm(w_pt - lw_pose))
                score += (d_lw - d_rw) * 5.0
            elif rw_pose is not None:
                d_rw = float(np.linalg.norm(w_pt - rw_pose))
                if d_rw < 0.25:
                    score += 2.0
            elif lw_pose is not None:
                d_lw = float(np.linalg.norm(w_pt - lw_pose))
                if d_lw < 0.25:
                    score -= 2.0

            # 3. Temporal tracking continuity with previous frames
            if self.prev_rh is not None and not np.all(self.prev_rh == 0.0):
                d_prev_rh = float(np.linalg.norm(w_pt - self.prev_rh[0, :2]))
                if d_prev_rh < 0.20:
                    score += 2.5
            if self.prev_lh is not None and not np.all(self.prev_lh == 0.0):
                d_prev_lh = float(np.linalg.norm(w_pt - self.prev_lh[0, :2]))
                if d_prev_lh < 0.20:
                    score -= 2.5

            # 4. Detector classification prior
            effective_label = label
            if not is_mirrored and label in ("Right", "Left"):
                effective_label = "Left" if label == "Right" else "Right"
            if effective_label == "Right":
                score += 1.0
            elif effective_label == "Left":
                score -= 1.0

            return score

        if len(valid_candidates) == 1:
            pts, label, _ = valid_candidates[0]
            s = score_hand_as_right(pts, label)
            # If positive, assign to Right Hand; if negative, assign to Left Hand
            if s >= -0.2:
                return np.zeros((self.NUM_HAND, 3), dtype=np.float32), pts
            else:
                return pts, np.zeros((self.NUM_HAND, 3), dtype=np.float32)

        # 2 or more hands: Assign the one with highest score to RH, lowest to LH
        sorted_hands = sorted(valid_candidates, key=lambda c: score_hand_as_right(c[0], c[1]))
        lh_assigned = sorted_hands[0][0]
        rh_assigned = sorted_hands[-1][0]
        return lh_assigned, rh_assigned

    def normalize_landmarks(self, pose_pts, face_pts, left_hand_pts, right_hand_pts):
        """Normalizes spatial landmarks relative to nose anchor and shoulder-width scale.

        Scale invariance calculation:
        - Anchor: Nose (pose index 0)
        - Scale: Euclidean distance between Left Shoulder (idx 11) and Right Shoulder (idx 12)
        Also provides single-hand signing routing to ensure single-hand gestures (A-Z, HELLO,
        THANK-YOU, PLEASE, NO, YES) are reliably mapped to the primary model channel.
        """
        has_pose = not np.all(pose_pts == 0.0)
        nose_raw = pose_pts[0].copy()
        left_shoulder = pose_pts[11]
        right_shoulder = pose_pts[12]

        shoulder_dist = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))

        if has_pose and not np.all(nose_raw == 0.0):
            nose = nose_raw
            self.stable_nose = nose_raw.copy()
        elif self.stable_nose is not None and has_pose:
            nose = self.stable_nose.copy()
        else:
            nose = np.zeros(3, dtype=np.float32)

        if has_pose and shoulder_dist > 0.05:
            scale = shoulder_dist
            self.stable_scale = shoulder_dist
        elif self.stable_scale is not None and has_pose:
            scale = self.stable_scale
        else:
            scale = 1.0

        def norm_group(pts):
            is_zero = np.all(pts == 0.0, axis=-1, keepdims=True)
            normed = (pts - nose) / scale
            return np.where(is_zero, 0.0, normed)

        norm_pose = norm_group(pose_pts)
        norm_face = norm_group(face_pts)
        norm_lh = norm_group(left_hand_pts)
        norm_rh = norm_group(right_hand_pts)

        # Single-hand routing: If only one hand is active (e.g. left hand) and right hand is empty,
        # route the single active hand to the right hand slot (and mirror X relative to wrist)
        # so all single-hand ASL signs and alphabet letters (A-Z) are recognized accurately.
        rh_is_active = not np.all(norm_rh == 0.0)
        lh_is_active = not np.all(norm_lh == 0.0)
        if lh_is_active and not rh_is_active:
            routed_rh = norm_lh.copy()
            wrist_x = routed_rh[0, 0]
            routed_rh[:, 0] = wrist_x - (routed_rh[:, 0] - wrist_x)
            norm_rh = routed_rh
            norm_lh = np.zeros_like(norm_lh)

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
        is_active = self._is_valid_hand(raw_pts)
        if is_active:
            if prev_pts is not None and not np.all(prev_pts == 0.0):
                smoothed = (self.smoothing_alpha * raw_pts + (1.0 - self.smoothing_alpha) * prev_pts).astype(np.float32)
            else:
                smoothed = raw_pts.copy()
            return smoothed, smoothed.copy(), 0
        else:
            if prev_pts is not None and lost_count < self.max_lost_persistence:
                # Retain previous position smoothly without abrupt zeroing
                return prev_pts.copy(), prev_pts, lost_count + 1
            else:
                return np.zeros((self.NUM_HAND, 3), dtype=np.float32), None, lost_count + 1

    def draw_skeleton_overlay(self, frame_bgr, results, lh_pts=None, rh_pts=None):
        """Renders skeletal landmarks and bones onto frame with theme colors:

        - Pose bones: Cyan (255, 255, 0)
        - Face mesh: Yellow (0, 255, 255)
        - Left hand: Pink (203, 192, 255) with magenta bones
        - Right hand: Light Green (0, 255, 127) with emerald bones
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

    def extract(self, frame_rgb, frame_bgr_for_vis=None, is_mirrored=True):
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
            lh_holistic = self._extract_raw_points(getattr(results, 'left_hand_landmarks', None), self.NUM_HAND)
            rh_holistic = self._extract_raw_points(getattr(results, 'right_hand_landmarks', None), self.NUM_HAND)
        else:
            pose_pts = np.zeros((self.NUM_POSE, 3), dtype=np.float32)
            face_pts = np.zeros((self.NUM_FACE, 3), dtype=np.float32)
            lh_holistic = np.zeros((self.NUM_HAND, 3), dtype=np.float32)
            rh_holistic = np.zeros((self.NUM_HAND, 3), dtype=np.float32)

        # Collect candidate hands from Holistic and dedicated Hands detector
        candidate_hands = []

        if self._is_valid_hand(rh_holistic):
            candidate_hands.append((rh_holistic, "Right", 0.90))
        if self._is_valid_hand(lh_holistic):
            candidate_hands.append((lh_holistic, "Left", 0.90))

        # If Holistic missed either hand, or to ensure robust tracking, run dedicated Hands detector
        need_hands_fallback = len(candidate_hands) < 2 and self.hands is not None
        if need_hands_fallback:
            try:
                hands_res = self.hands.process(frame_rgb)
                if hands_res and hands_res.multi_hand_landmarks:
                    for idx, hand_lms in enumerate(hands_res.multi_hand_landmarks):
                        h_label = "Right"
                        score = 0.80
                        if hasattr(hands_res, 'multi_handedness') and idx < len(hands_res.multi_handedness):
                            c_info = hands_res.multi_handedness[idx].classification
                            if c_info:
                                h_label = c_info[0].label
                                score = float(c_info[0].score)

                        hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                        # Avoid duplicate if close to an existing holistic candidate
                        is_duplicate = False
                        for exist_pts, _, _ in candidate_hands:
                            if np.linalg.norm(hand_coords[0, :2] - exist_pts[0, :2]) < 0.08:
                                is_duplicate = True
                                break

                        if not is_duplicate and self._is_valid_hand(hand_coords):
                            candidate_hands.append((hand_coords, h_label, score))
            except Exception:
                pass

        # Intelligent Hand Assignment preventing hand-swapping
        lh_pts, rh_pts = self._assign_hands(candidate_hands, pose_pts, is_mirrored=is_mirrored)

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
