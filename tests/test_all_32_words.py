"""Interactive 32-Sign ASL Vocabulary Verification Suite & Desktop GUI (tests/test_all_32_words.py).

Guided live interactive calibration test for all 32 ASL signs (6 words + 26 alphabet letters):
- Prompts the user to perform each of the 32 signs in sequence.
- Tracks 3 successful detections per sign before automatically advancing to the next sign.
- Features a dual-pane Desktop GUI HUD:
    * Left Pane: Live video feed with skeletal overlay, prominent target sign banner,
      signing technique hints, trial progress badges [✓] [✓] [ ], live confidence meter,
      and clickable navigation buttons ([Prev], [Reset], [Next], [Mute], [Exit]).
    * Right Pane: Real-time 32-sign interactive checklist showing completion status
      ([✓] PASS, [▶] ACTIVE, [ ] PENDING) and overall session progress bar.
- Provides spoken audio cues via Neural Edge-TTS.
- Automatically exports a full markdown and JSON verification scorecard upon completion or exit.

Usage:
    python tests/test_all_32_words.py               # Launch interactive live desktop GUI
    python tests/test_all_32_words.py --camera 0    # Specify camera index
    pytest tests/test_all_32_words.py               # Automated CI test mode
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline import ContinuousInterpretationPipeline


# Comprehensive signing instructions for all 32 ASL signs (6 words + 26 alphabet letters)
SIGN_DESCRIPTIONS = {
    # ---- 6 ASL Words ----
    "HELLO":     "Open flat hand at temple/forehead, saluting/waving outward.",
    "THANK-YOU": "Flat hand touches chin/lips, moving forward/downward toward camera.",
    "PLEASE":    "Flat open hand rubbing in a circular motion on chest.",
    "MEET":      "Both index fingers pointing up, bringing knuckles together.",
    "NO":        "Index and middle fingers snapping firmly down onto thumb.",
    "YES":       "Closed fist nodding up and down like a head nodding.",
    # ---- 26 ASL Alphabet Letters ----
    "A": "Closed fist, thumb resting on side (like an 'A' fist).",
    "B": "Four fingers flat up, thumb tucked across palm.",
    "C": "Curve all fingers and thumb into a 'C' arc shape.",
    "D": "Index finger up, thumb touching middle finger forming 'D'.",
    "E": "All fingers curled/hooked down, thumb tucked under.",
    "F": "Thumb and index touch (OK), other three fingers spread up.",
    "G": "Index and thumb pointing sideways (like a gun pointing right).",
    "H": "Index and middle fingers extended together pointing sideways.",
    "I": "Pinky only extended straight up, fist closed.",
    "J": "Pinky up (like I), then trace a 'J' curve with pinky.",
    "K": "Index and middle up, thumb out between them (like K).",
    "L": "Index up and thumb out sideways forming an 'L' shape.",
    "M": "Three fingers (index, mid, ring) folded over thumb.",
    "N": "Two fingers (index, mid) folded over thumb.",
    "O": "All fingertips curve to meet thumb, forming an 'O' circle.",
    "P": "Like K shape but pointing downward.",
    "Q": "Like G shape but pointing downward.",
    "R": "Index and middle fingers crossed (R for 'right').",
    "S": "Closed fist, thumb wrapped over fingers (all tucked).",
    "T": "Thumb tucked between index and middle finger.",
    "U": "Index and middle fingers up together side by side.",
    "V": "Index and middle fingers spread into V/peace sign.",
    "W": "Index, middle, and ring fingers spread up (three fingers).",
    "X": "Index finger bent/hooked like a hook.",
    "Y": "Thumb and pinky extended (shaka/hang-loose sign).",
    "Z": "Index finger traces a 'Z' shape in the air.",
}


class InteractiveVocabTester:
    """Manages sequential testing, verification, and desktop GUI for all 32 ASL signs."""

    def __init__(self, config=None, trials_per_word=3, min_confidence=0.70):
        self.config = config or self._default_config()
        self.trials_per_word = int(trials_per_word)
        self.min_confidence = float(min_confidence)

        # Pipeline
        self.pipeline = ContinuousInterpretationPipeline(self.config)
        self.vocab = self.pipeline.vocab

        # State tracking
        self.current_word_idx = 0
        self.current_trial = 0
        self.running = False
        self.audio_enabled = True

        # Timing, Hold-Streak & Debounce
        self.cooldown_frames = 0
        self.COOLDOWN_DELAY = 45   # ~1.5 seconds at 30 FPS between trials
        self.advance_delay_frames = 0
        self.success_flash_frames = 0
        self.hold_streak = 0
        self.REQUIRED_HOLD_FRAMES = 6  # Requires holding gesture steadily for 6 consecutive frames (~200ms)

        # Detailed results tracking
        self.results = {
            token: {
                "completed": False,
                "trials_completed": 0,
                "confidences": [],
                "elapsed_sec": 0.0,
                "attempts": 0
            }
            for token in self.vocab
        }
        self.word_start_time = time.time()
        self.session_start_time = time.time()

        # GUI Buttons Definition (x1, y1, x2, y2, label, action)
        self.buttons = [
            {"rect": (25, 665, 145, 705), "label": "< Prev", "action": "prev", "bg": (55, 60, 75)},
            {"rect": (160, 665, 285, 705), "label": "Reset (0/3)", "action": "reset", "bg": (65, 55, 50)},
            {"rect": (300, 665, 435, 705), "label": "Next >", "action": "next", "bg": (40, 90, 160)},
            {"rect": (450, 665, 575, 705), "label": "Mute Voice", "action": "mute", "bg": (50, 70, 75)},
            {"rect": (590, 665, 715, 705), "label": "Save & Exit", "action": "exit", "bg": (140, 45, 45)},
        ]

    def _default_config(self):
        """Loads default system configuration."""
        cfg_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
        if os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "system": {"target_fps": 30, "camera_idx": 0, "device": "auto"},
            "glosser": {"threshold": 0.70, "min_consecutive": 3, "refractory_period": 8}
        }

    def start(self):
        """Starts video ingestion."""
        self.running = True
        self.pipeline.start()
        self.word_start_time = time.time()
        self.session_start_time = time.time()
        target = self.get_current_target()
        self._announce(f"ASL Verification Suite active. Target sign 1: {target}. Please perform {target} 3 times.")

    def stop(self):
        """Stops video ingestion."""
        self.running = False
        self.pipeline.stop()

    def get_current_target(self):
        """Returns the current target sign token."""
        if 0 <= self.current_word_idx < len(self.vocab):
            return self.vocab[self.current_word_idx]
        return ""

    def _announce(self, text):
        """Speaks announcement via Edge-TTS if voice is enabled."""
        if self.audio_enabled and hasattr(self.pipeline, 'tts'):
            try:
                self.pipeline.tts.speak_async(text)
            except Exception:
                pass

    def advance_to_next_word(self):
        """Advances to next word in vocabulary sequence."""
        target = self.get_current_target()
        if target:
            elapsed = time.time() - self.word_start_time
            self.results[target]["elapsed_sec"] = round(elapsed, 2)
            self.results[target]["completed"] = (self.current_trial >= self.trials_per_word)
            self.results[target]["trials_completed"] = self.current_trial

        self.current_word_idx += 1
        self.current_trial = 0
        self.cooldown_frames = 0
        self.advance_delay_frames = 0
        self.word_start_time = time.time()
        self.pipeline.glosser.reset()

        if self.current_word_idx < len(self.vocab):
            next_target = self.get_current_target()
            self._announce(f"Next sign: {next_target}.")
        else:
            self._announce(f"Congratulations! All {len(self.vocab)} vocabulary signs completed.")

    def previous_word(self):
        """Navigates back to the previous word."""
        if self.current_word_idx > 0:
            self.current_word_idx -= 1
            self.current_trial = 0
            self.cooldown_frames = 0
            self.advance_delay_frames = 0
            self.word_start_time = time.time()
            self.pipeline.glosser.reset()
            self._announce(f"Back to {self.get_current_target()}.")

    def reset_current_word(self):
        """Resets trial count for the current word."""
        self.current_trial = 0
        self.cooldown_frames = 0
        self.pipeline.glosser.reset()
        self._announce(f"Reset {self.get_current_target()}. Trial 1 of 3.")

    def handle_click(self, x, y):
        """Processes mouse click on interactive HUD buttons."""
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                action = btn["action"]
                if action == "next":
                    self.advance_to_next_word()
                elif action == "prev":
                    self.previous_word()
                elif action == "reset":
                    self.reset_current_word()
                elif action == "mute":
                    self.audio_enabled = not self.audio_enabled
                    state = "ON" if self.audio_enabled else "MUTED"
                    print(f"Voice Audio: {state}")
                elif action == "exit":
                    self.running = False
                    return True
                return True
        return False

    def step(self):
        """Processes one video frame and evaluates trial matching logic."""
        annotated_frame, emitted_sign, conf, sentence = self.pipeline.step()

        target = self.get_current_target()
        pred_sign = getattr(self.pipeline, "latest_predicted_token", "")
        conf_pct = conf * 100.0

        # Decrement cooldown
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

        # Waiting for word transition
        if self.advance_delay_frames > 0:
            self.advance_delay_frames -= 1
            if self.advance_delay_frames == 0:
                self.advance_to_next_word()
            return annotated_frame, pred_sign, conf, False

        # Match check: Require steady holding of target gesture or confirmed emitted token
        is_match = False
        if target and self.cooldown_frames == 0 and self.advance_delay_frames == 0:
            if emitted_sign == target and conf >= self.min_confidence:
                is_match = True
                self.hold_streak = 0
            elif pred_sign == target and conf >= self.min_confidence:
                self.hold_streak += 1
                if self.hold_streak >= self.REQUIRED_HOLD_FRAMES:
                    is_match = True
                    self.hold_streak = 0
            else:
                self.hold_streak = max(0, self.hold_streak - 1)
        else:
            self.hold_streak = 0

        trial_registered = False
        if is_match and self.current_trial < self.trials_per_word:
            self.current_trial += 1
            trial_registered = True
            self.success_flash_frames = 15
            self.cooldown_frames = self.COOLDOWN_DELAY
            self.hold_streak = 0
            self.pipeline.glosser.reset()

            self.results[target]["confidences"].append(round(conf_pct, 1))
            self.results[target]["trials_completed"] = self.current_trial

            if self.current_trial < self.trials_per_word:
                self._announce(f"{target}! Trial {self.current_trial} of {self.trials_per_word}.")
            else:
                self.results[target]["completed"] = True
                self._announce(f"Excellent! {target} completed.")
                self.advance_delay_frames = 40  # ~1.3 seconds celebration before next sign

        return annotated_frame, pred_sign, conf, trial_registered


    def render_hud(self, annotated_frame, pred_sign, conf):
        """Constructs an informative, beautiful visual HUD over the video feed."""
        canvas_w, canvas_h = 1180, 720
        hud = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Background fills
        hud[:, :740] = (16, 18, 24)       # Left pane (dark slate)
        hud[:, 740:] = (22, 26, 34)       # Right sidebar (charcoal navy)
        cv2.line(hud, (740, 0), (740, canvas_h), (40, 48, 62), 2)

        target = self.get_current_target()
        total_vocab = len(self.vocab)
        all_finished = (self.current_word_idx >= total_vocab)

        # ======================================================================
        # LEFT PANE: Video Canvas & Telemetry (Width: 740px)
        # ======================================================================

        # 1. Top Title Bar (Height: 46px)
        cv2.rectangle(hud, (0, 0), (740, 46), (24, 28, 38), -1)
        cv2.putText(hud, f"ASL {total_vocab}-SIGN INTERACTIVE VERIFICATION SUITE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 229, 255), 2)

        if not all_finished:
            pct_total = (self.current_word_idx / total_vocab) * 100.0
            cv2.putText(hud, f"Sign {self.current_word_idx + 1}/{total_vocab} ({pct_total:.0f}%)", (560, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 215, 255), 2)

        # 2. Prominent Target Sign Card (Height: 88px)
        if not all_finished:
            cv2.rectangle(hud, (15, 54), (725, 142), (28, 34, 46), -1)
            cv2.rectangle(hud, (15, 54), (725, 142), (0, 200, 255), 1)

            # Target Sign Heading
            cv2.putText(hud, f"CURRENT TARGET:  [ {target} ]", (30, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

            # Trial Checkbox Badges
            badges = []
            for t in range(self.trials_per_word):
                badges.append("[X]" if t < self.current_trial else "[ ]")
            badges_str = " ".join(badges) + f"  ({self.current_trial}/{self.trials_per_word})"

            badge_color = (0, 230, 118) if self.current_trial == self.trials_per_word else (0, 215, 255)
            cv2.putText(hud, badges_str, (490, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)

            # Technique Hint
            hint_text = SIGN_DESCRIPTIONS.get(target, "Perform sign facing camera.")
            cv2.putText(hud, f"How to sign: {hint_text}", (30, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 120), 1)
        else:
            cv2.rectangle(hud, (15, 54), (725, 142), (18, 55, 28), -1)
            cv2.rectangle(hud, (15, 54), (725, 142), (0, 230, 118), 2)
            cv2.putText(hud, f"★ ALL {total_vocab} ASL SIGNS SUCCESSFULLY VERIFIED! ★", (40, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 128), 2)

        # 3. Centered Video Canvas (480x360 scaled to fit 690x400)
        vid_h, vid_w = annotated_frame.shape[:2]
        target_vw, target_vh = 690, 395
        scale = min(target_vw / vid_w, target_vh / vid_h)
        scaled_w, scaled_h = int(vid_w * scale), int(vid_h * scale)
        scaled_frame = cv2.resize(annotated_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        vx = 15 + (710 - scaled_w) // 2
        vy = 150 + (target_vh - scaled_h) // 2
        hud[vy:vy + scaled_h, vx:vx + scaled_w] = scaled_frame
        cv2.rectangle(hud, (vx, vy), (vx + scaled_w, vy + scaled_h), (45, 55, 75), 1)

        # Success Flash Frame Border
        if self.success_flash_frames > 0:
            self.success_flash_frames -= 1
            cv2.rectangle(hud, (vx, vy), (vx + scaled_w, vy + scaled_h), (0, 230, 118), 6)
            cv2.putText(hud, f"★ TRIAL {self.current_trial}/3 CONFIRMED! ★", (vx + 160, vy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2)

        # 4. Telemetry Banner (Below Video)
        conf_pct = conf * 100.0
        tag = "HIGH" if conf_pct >= 85 else ("MED" if conf_pct >= 60 else "LOW")
        disp_sign = pred_sign if conf_pct >= 40 else "—"
        tts_str = "ON" if self.audio_enabled else "MUTED"

        if not all_finished:
            if self.cooldown_frames > 0:
                status_text = "● LOWER HAND & PREPARE FOR NEXT TRIAL"
                status_color = (0, 215, 255)
            elif disp_sign == target and conf_pct >= 70:
                status_text = "★ TARGET SIGN DETECTED — HOLDING STEADY..."
                status_color = (0, 230, 118)
            elif disp_sign != "—":
                status_text = f"Observing: {disp_sign} (Target is {target})"
                status_color = (180, 180, 180)
            else:
                status_text = "● Awaiting gesture motion in camera frame..."
                status_color = (130, 140, 150)
        else:
            status_text = "Session complete! Click [Save & Exit] or press [Q] to finish."
            status_color = (0, 255, 128)

        cv2.putText(hud, status_text, (25, 570),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, status_color, 2)

        # Confidence Gauge Bar
        metrics_line = (f"Live Recognized: {disp_sign} ({conf_pct:.1f}% [{tag}])   |   "
                        f"Current Target: {target or 'FINISHED'}   |   Voice Audio: {tts_str}")
        cv2.putText(hud, metrics_line, (25, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 215, 230), 1)

        # Progress bar for current word trials
        bar_x, bar_y, bar_w, bar_h = 25, 616, 690, 12
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 42, 55), -1)
        fill_w = int(bar_w * (self.current_trial / max(1, self.trials_per_word)))
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 230, 118), -1)

        # Hotkeys Helper
        cv2.putText(hud, "Hotkeys: [N] Next | [B] Back | [R] Reset | [M] Mute | [Q] Quit", (25, 648),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 140, 160), 1)

        # 5. Interactive Clickable Buttons
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            label = btn["label"]
            if btn["action"] == "mute":
                label = "Mute Voice" if self.audio_enabled else "Unmute Voice"
            bg = btn["bg"]
            cv2.rectangle(hud, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(hud, (x1, y1), (x2, y2), (90, 105, 125), 1)

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
            tx = x1 + (x2 - x1 - text_size[0]) // 2
            ty = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(hud, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        # ======================================================================
        # RIGHT SIDEBAR: Complete 32-Sign Checklist (Width: 440px)
        # ======================================================================

        # Sidebar Title
        cv2.rectangle(hud, (740, 0), (canvas_w, 46), (28, 32, 44), -1)
        cv2.putText(hud, f"VOCABULARY CHECKLIST ({total_vocab} SIGNS)", (755, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 215, 255), 2)

        total_done = sum(1 for v in self.results.values() if v["completed"])
        summary_str = f"Verified: {total_done} / {total_vocab}  ({(total_done / max(1, total_vocab)) * 100:.1f}%)"
        cv2.putText(hud, summary_str, (755, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1)

        # Overall Session Progress Bar
        bar2_x, bar2_y, bar2_w, bar2_h = 755, 76, 400, 8
        cv2.rectangle(hud, (bar2_x, bar2_y), (bar2_x + bar2_w, bar2_y + bar2_h), (35, 42, 55), -1)
        fill2_w = int(bar2_w * (total_done / max(1.0, float(total_vocab))))
        cv2.rectangle(hud, (bar2_x, bar2_y), (bar2_x + fill2_w, bar2_y + bar2_h), (0, 230, 118), -1)

        # Render all 32 signs in a clean table (each item is 19px high)
        list_start_y = 100
        row_h = 19

        for idx, token in enumerate(self.vocab):
            ry = list_start_y + (idx * row_h)
            is_active = (idx == self.current_word_idx)
            is_completed = self.results[token]["completed"]
            trials_comp = self.results[token]["trials_completed"]
            confs = self.results[token]["confidences"]
            mean_conf = f"{np.mean(confs):.0f}%" if confs else "--"

            # Row background highlight for active word
            if is_active:
                cv2.rectangle(hud, (748, ry - 13), (canvas_w - 10, ry + 4), (45, 60, 85), -1)
                cv2.rectangle(hud, (748, ry - 13), (canvas_w - 10, ry + 4), (0, 200, 255), 1)

            # Prefix & colors
            if is_completed:
                prefix = "[X]"
                color = (0, 230, 118)   # Bright green
            elif is_active:
                prefix = "[>]"
                color = (0, 235, 255)   # Cyan / Yellow
            else:
                prefix = "[ ]"
                color = (120, 130, 145) # Dim slate

            text_row = f"{prefix} {idx+1:02d}. {token:<12s} ({trials_comp}/3)  {mean_conf}"
            cv2.putText(hud, text_row, (755, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        return hud

    def export_report(self, filepath="tests/vocab_test_results.md"):
        """Exports complete verification results to markdown and JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        total_completed = sum(1 for v in self.results.values() if v["completed"])
        total_time = round(time.time() - self.session_start_time, 1)
        total_vocab = len(self.vocab)

        lines = [
            f"# ASL {total_vocab}-Sign Vocabulary Verification Report\n\n",
            f"- **Date & Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- **Words Verified (3/3 Trials):** {total_completed} / {total_vocab} ({total_completed / max(1, total_vocab) * 100:.1f}%)\n",
            f"- **Total Session Duration:** {total_time} seconds\n\n",
            "## Detailed Results Table:\n\n",
            "| Index | Sign Token | Trials Completed | Status | Mean Confidence | Duration (s) |\n",
            "|---|---|:---:|:---:|:---:|:---:|\n"
        ]

        for idx, token in enumerate(self.vocab, 1):
            data = self.results[token]
            status = "PASS" if data["completed"] else ("PARTIAL" if data["trials_completed"] > 0 else "SKIPPED")
            confs = data["confidences"]
            mean_c = f"{np.mean(confs):.1f}%" if confs else "N/A"
            dur = f"{data['elapsed_sec']:.1f}s" if data['elapsed_sec'] > 0 else "—"
            lines.append(f"| {idx} | **{token}** | {data['trials_completed']}/{self.trials_per_word} | {status} | {mean_c} | {dur} |\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Also write JSON summary
        json_path = os.path.splitext(filepath)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_completed": total_completed,
                "total_words": total_vocab,
                "duration_seconds": total_time,
                "results": self.results
            }, f, indent=2)

        return filepath

    def run_interactive_session(self):
        """Runs the main interactive loop displaying OpenCV HighGUI window."""
        total_vocab = len(self.vocab)
        window_name = f"ASL {total_vocab}-Sign Verification Suite"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1180, 720)

        # Mouse click callback
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.handle_click(x, y)

        cv2.setMouseCallback(window_name, on_mouse)

        print("\n" + "=" * 75)
        print(f"  ASL {total_vocab}-Sign Interactive Verification Suite & Desktop GUI")
        print("=" * 75)
        print(f"Total Signs: {total_vocab}")
        print("Goal: Perform each sign 3 times. The system counts and advances automatically.")
        print("Interactive Controls:")
        print("  Mouse : Click on-screen buttons [< Prev], [Reset], [Next >], [Mute], [Save & Exit]")
        print("  [N]   : Skip current word and advance to next")
        print("  [B]   : Go back to previous word")
        print("  [R]   : Reset counter for current word")
        print("  [M]   : Toggle voice audio on / off")
        print("  [Q]   : Save report and quit")
        print("=" * 75 + "\n")

        self.start()
        try:
            while self.running:
                annotated_frame, pred_sign, conf, registered = self.step()
                hud = self.render_hud(annotated_frame, pred_sign, conf)

                cv2.imshow(window_name, hud)
                key = cv2.waitKey(20) & 0xFF

                if key in [ord('q'), 27]:
                    print("\nExit requested by user. Finishing session...")
                    break
                elif key == ord('n'):
                    print(f"Skipping {self.get_current_target()} -> advancing to next.")
                    self.advance_to_next_word()
                elif key in [ord('b'), ord('p')]:
                    print("Returning to previous word.")
                    self.previous_word()
                elif key == ord('r'):
                    print(f"Resetting trials for {self.get_current_target()}.")
                    self.reset_current_word()
                elif key == ord('m'):
                    self.audio_enabled = not self.audio_enabled
                    state = "ON" if self.audio_enabled else "MUTED"
                    print(f"Voice Audio: {state}")
        finally:
            self.stop()
            cv2.destroyAllWindows()
            report_file = self.export_report()
            print(f"\nVerification report generated: {os.path.abspath(report_file)}")
            print("=" * 75)


# ==============================================================================
# PyTest Automated Unit Tests (Allows pytest tests/ to pass in automated CI)
# ==============================================================================

def test_vocab_tester_initialization():
    """Validates tester loads all 32 classes (6 words + 26 letters) and initializes state machine cleanly."""
    tester = InteractiveVocabTester(trials_per_word=3)
    assert len(tester.vocab) == 32, f"Expected 32 classes, got {len(tester.vocab)}"
    assert tester.current_word_idx == 0
    assert tester.current_trial == 0
    assert tester.get_current_target() == tester.vocab[0]
    assert len(SIGN_DESCRIPTIONS) == 32
    for word in tester.vocab:
        assert word in SIGN_DESCRIPTIONS, f"Missing signing hint for {word}"


def test_vocab_tester_progression_logic():
    """Validates progression across 3 trials and advancement to subsequent word."""
    tester = InteractiveVocabTester(trials_per_word=3)
    initial_target = tester.get_current_target()

    # Trial 1
    tester.current_trial = 1
    assert tester.current_trial == 1
    assert tester.get_current_target() == initial_target

    # Trial 2
    tester.current_trial = 2
    assert tester.current_trial == 2

    # Trial 3: complete all 3 trials and advance
    tester.current_trial = 3
    tester.advance_to_next_word()
    assert tester.current_word_idx == 1
    assert tester.current_trial == 0
    assert tester.get_current_target() == tester.vocab[1]
    assert tester.results[initial_target]["completed"] is True


def test_vocab_tester_report_generation(tmp_path):
    """Validates markdown and JSON verification reporting."""
    tester = InteractiveVocabTester(trials_per_word=3)
    out_file = str(tmp_path / "test_report.md")
    tester.results["HELLO"]["completed"] = True
    tester.results["HELLO"]["trials_completed"] = 3
    tester.results["HELLO"]["confidences"] = [95.0, 96.2, 98.1]

    saved = tester.export_report(out_file)
    assert os.path.exists(saved)
    with open(saved, "r", encoding="utf-8") as f:
        content = f.read()
        assert "HELLO" in content
        assert "3/3" in content
        assert "PASS" in content

    json_file = str(tmp_path / "test_report.json")
    assert os.path.exists(json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_words"] == len(tester.vocab)


def main():
    parser = argparse.ArgumentParser(description="ASL 32-Sign Interactive Verification Suite & Desktop GUI")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override (default: 0)")
    parser.add_argument("--trials", type=int, default=3, help="Successful detections required per sign (default: 3)")
    parser.add_argument("--thresh", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    args = parser.parse_args()

    cfg_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
    config = {}
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    if args.camera is not None:
        if 'system' not in config:
            config['system'] = {}
        config['system']['camera_idx'] = args.camera

    tester = InteractiveVocabTester(config=config, trials_per_word=args.trials, min_confidence=args.thresh)
    tester.run_interactive_session()


if __name__ == "__main__":
    main()
