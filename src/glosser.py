"""Module 6: Temporal Gloss Filtering & Hysteresis Smoothing (src/glosser.py).

Suppresses repeated tokens and transitional jitter over continuous video frames:
- Implements a sliding-window peak detector with refractory hysteresis smoothing
- Emits a sign token only if predicted confidence exceeds threshold tau=0.85
  for at least k=5 consecutive frames
- Followed by a refractory cooldown period (R=8 frames) to suppress duplicate token bursts
"""

import collections


class TemporalHysteresisGlosser:
    """Sliding-window peak detector and hysteresis smoother for continuous sign streams."""

    def __init__(self, thresh=0.85, min_consecutive=5, refractory_period=8):
        self.thresh = float(thresh)
        self.k = int(min_consecutive)
        self.refractory = int(refractory_period)

        self.history = collections.deque(maxlen=30)
        self.last_token = None
        self.cooldown = 0
        self.idle_frames = 0
        self.emitted_tokens = []

    def update(self, predicted_sign, confidence):
        """Processes current frame's prediction and returns emitted sign token if stabilized.

        Args:
            predicted_sign: str or int identifier of the candidate sign
            confidence: float confidence (0.0 to 1.0)
        Returns:
            emitted sign token (str/int) or None
        """
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if confidence >= self.thresh and predicted_sign is not None:
            self.idle_frames = 0
            self.history.append(predicted_sign)
            if len(self.history) >= self.k:
                # Check if all last k frames are identical to candidate sign
                recent_streak = list(self.history)[-self.k:]
                if all(s == predicted_sign for s in recent_streak):
                    if predicted_sign != self.last_token:
                        self.last_token = predicted_sign
                        self.cooldown = self.refractory
                        self.emitted_tokens.append(predicted_sign)
                        return predicted_sign
        else:
            self.history.clear()
            self.idle_frames += 1
            if self.idle_frames >= self.refractory:
                self.last_token = None

        return None

    def get_tokens(self):
        """Returns the chronological list of accumulated emitted gloss tokens."""
        return list(self.emitted_tokens)

    def clear_tokens(self):
        """Clears the accumulated gloss tokens buffer."""
        self.emitted_tokens.clear()

    def reset(self):
        """Resets all internal states, history, and cooldowns."""
        self.history.clear()
        self.last_token = None
        self.cooldown = 0
        self.idle_frames = 0
        self.emitted_tokens.clear()
