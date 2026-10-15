"""Module 9: Live Interactive Desktop GUI (src/gui_app.py).

Native cross-platform desktop application wrapping the entire pipeline:
- Live 30 FPS video viewport with skeletal tracking overlays
- Dynamic subtitle card & HUD displaying synthesized English sentences
- Confidence gauge meter (e.g. "Confidence Score: 94% [HIGH]")
- Gloss token timeline buffer display
- Audio playback and session controls (Pause, Resume, Mute, Clear, Export)
- Graceful headless / non-Tkinter fallback for environments lacking tcl/tk
"""

import os
import sys
import time
import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import Image, ImageTk
    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False


# Lightweight mock classes for environments without tkinter installed
class _MockWidget:
    def __init__(self, *args, **kwargs):
        self.text = kwargs.get("text", "")
        self.bg = kwargs.get("bg", "")
        self.fg = kwargs.get("fg", "")
        self.items = []
        self._value = 0

    def pack(self, *args, **kwargs): pass
    def pack_propagate(self, *args, **kwargs): pass
    def config(self, **kwargs):
        if "text" in kwargs: self.text = kwargs["text"]
        if "bg" in kwargs: self.bg = kwargs["bg"]
        if "fg" in kwargs: self.fg = kwargs["fg"]
    def configure(self, **kwargs): self.config(**kwargs)
    def insert(self, idx, item): self.items.append(item)
    def delete(self, start, end=None): self.items.clear()
    def see(self, idx): pass
    def size(self): return len(self.items)
    def winfo_width(self): return 640
    def winfo_height(self): return 480
    def __setitem__(self, key, val):
        if key == 'value': self._value = val
    def __getitem__(self, key):
        if key == 'value': return self._value
        return None

class _MockTk:
    def __init__(self):
        self._after_callbacks = []
    def title(self, t): pass
    def geometry(self, g): pass
    def minsize(self, w, h): pass
    def configure(self, **kwargs): pass
    def withdraw(self): pass
    def protocol(self, p, cb): pass
    def after(self, ms, func):
        self._after_callbacks.append(func)
    def mainloop(self): pass
    def destroy(self): pass


from src.pipeline import ContinuousInterpretationPipeline


class SignInterpreterGUI:
    """Live Interactive Desktop Application for Sign Language Interpretation."""

    def __init__(self, root=None, config=None):
        if not _TKINTER_AVAILABLE or root is None:
            self.root = root if root is not None else _MockTk()
            self._using_mock = not _TKINTER_AVAILABLE
        else:
            self.root = root
            self._using_mock = False

        self.root.title("Real-Time ASL Interpretation System — Team 14")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self.root.configure(bg="#121418")

        self.config = config or {}
        self.pipeline = ContinuousInterpretationPipeline(self.config)

        self.is_paused = False
        self.audio_enabled = True
        self.transcript_history = []

        self._build_ui()
        self.pipeline.start()
        self.running = True
        self._schedule_next_frame()

    def _build_ui(self):
        """Constructs modern dark-themed HUD layout."""
        if self._using_mock:
            self.canvas_label = _MockWidget()
            self.subtitle_text = _MockWidget(text="Awaiting ASL input...")
            self.status_badge = _MockWidget(text="● STREAMING 30 FPS")
            self.conf_label = _MockWidget(text="Confidence Score: 0.0%")
            self.conf_bar = _MockWidget()
            self.token_label = _MockWidget(text="Current Sign: —")
            self.timeline_box = _MockWidget()
            self.pause_btn = _MockWidget(text="⏸ Pause Feed")
            self.audio_btn = _MockWidget(text="🔊 Mute Voice TTS")
            self.clear_btn = _MockWidget(text="🧹 Clear Transcript")
            self.export_btn = _MockWidget(text="💾 Export Transcript Log")
            return

        # Top Header Bar
        header = tk.Frame(self.root, bg="#1a1d24", height=50)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="ASL Assistive Interpretation System",
            font=("Segoe UI", 16, "bold"),
            fg="#00e5ff",
            bg="#1a1d24"
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        self.status_badge = tk.Label(
            header,
            text="● STREAMING 30 FPS",
            font=("Segoe UI", 10, "bold"),
            fg="#00e676",
            bg="#1a1d24"
        )
        self.status_badge.pack(side=tk.RIGHT, padx=20)

        # Main Content Layout: Left Video, Right Telemetry & Controls
        content = tk.Frame(self.root, bg="#121418")
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Video Canvas
        left_panel = tk.Frame(content, bg="#181b22", relief=tk.RIDGE, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas_label = tk.Label(left_panel, bg="#0d0e12")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Dynamic Subtitle Card HUD directly below video
        subtitle_card = tk.Frame(left_panel, bg="#20242e", bd=1)
        subtitle_card.pack(fill=tk.X, padx=5, pady=5)

        sub_label_title = tk.Label(
            subtitle_card,
            text="DYNAMIC SUBTITLE HUD (TRANSLATED ENGLISH):",
            font=("Segoe UI", 9, "bold"),
            fg="#80d8ff",
            bg="#20242e"
        )
        sub_label_title.pack(anchor="w", padx=10, pady=(6, 0))

        self.subtitle_text = tk.Label(
            subtitle_card,
            text="Awaiting ASL input...",
            font=("Segoe UI", 15, "bold"),
            fg="#ffffff",
            bg="#20242e",
            wraplength=600,
            justify="left"
        )
        self.subtitle_text.pack(anchor="w", padx=10, pady=(2, 8))

        # Right Column: Controls, Confidence Meter & Gloss Timeline
        right_panel = tk.Frame(content, bg="#181b22", width=340, relief=tk.RIDGE, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_panel.pack_propagate(False)

        # Telemetry Section
        tel_header = tk.Label(
            right_panel,
            text="REAL-TIME TELEMETRY",
            font=("Segoe UI", 11, "bold"),
            fg="#90caf9",
            bg="#181b22"
        )
        tel_header.pack(anchor="w", padx=15, pady=(15, 5))

        self.conf_label = tk.Label(
            right_panel,
            text="Confidence Score: 0.0%",
            font=("Segoe UI", 12, "bold"),
            fg="#e0e0e0",
            bg="#181b22"
        )
        self.conf_label.pack(anchor="w", padx=15, pady=(5, 2))

        # Confidence Progress Bar
        self.conf_bar = ttk.Progressbar(right_panel, length=300, mode='determinate')
        self.conf_bar.pack(padx=15, pady=(0, 15), fill=tk.X)

        self.token_label = tk.Label(
            right_panel,
            text="Current Sign: —",
            font=("Segoe UI", 12),
            fg="#b0bec5",
            bg="#181b22"
        )
        self.token_label.pack(anchor="w", padx=15, pady=(0, 10))

        # Gloss Timeline Section
        gloss_header = tk.Label(
            right_panel,
            text="GLOSS TOKEN TIMELINE",
            font=("Segoe UI", 11, "bold"),
            fg="#90caf9",
            bg="#181b22"
        )
        gloss_header.pack(anchor="w", padx=15, pady=(10, 5))

        self.timeline_box = tk.Listbox(
            right_panel,
            bg="#0e1116",
            fg="#00e5ff",
            font=("Consolas", 10),
            height=8,
            bd=0,
            highlightthickness=1,
            highlightbackground="#2c3240"
        )
        self.timeline_box.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Audio & Session Controls Section
        ctrl_header = tk.Label(
            right_panel,
            text="SESSION CONTROLS",
            font=("Segoe UI", 11, "bold"),
            fg="#90caf9",
            bg="#181b22"
        )
        ctrl_header.pack(anchor="w", padx=15, pady=(5, 8))

        btn_style = {"font": ("Segoe UI", 10, "bold"), "bd": 0, "relief": tk.FLAT, "pady": 6}

        self.pause_btn = tk.Button(
            right_panel,
            text="⏸ Pause Feed",
            bg="#2979ff",
            fg="#ffffff",
            activebackground="#1565c0",
            command=self.toggle_pause,
            **btn_style
        )
        self.pause_btn.pack(fill=tk.X, padx=15, pady=4)

        self.audio_btn = tk.Button(
            right_panel,
            text="🔊 Mute Voice TTS",
            bg="#37474f",
            fg="#ffffff",
            activebackground="#263238",
            command=self.toggle_audio,
            **btn_style
        )
        self.audio_btn.pack(fill=tk.X, padx=15, pady=4)

        self.clear_btn = tk.Button(
            right_panel,
            text="🧹 Clear Transcript",
            bg="#37474f",
            fg="#ffffff",
            activebackground="#263238",
            command=self.clear_transcript,
            **btn_style
        )
        self.clear_btn.pack(fill=tk.X, padx=15, pady=4)

        self.export_btn = tk.Button(
            right_panel,
            text="💾 Export Transcript Log",
            bg="#00b0ff",
            fg="#ffffff",
            activebackground="#0091ea",
            command=self.export_transcript,
            **btn_style
        )
        self.export_btn.pack(fill=tk.X, padx=15, pady=4)

    def toggle_pause(self):
        """Toggles between paused and active capture state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume Feed", bg="#00c853")
            self.status_badge.config(text="⏸ PAUSED", fg="#ffab00")
        else:
            self.pause_btn.config(text="⏸ Pause Feed", bg="#2979ff")
            self.status_badge.config(text="● STREAMING 30 FPS", fg="#00e676")

    def toggle_audio(self):
        """Mutes or unmutes Neural Edge-TTS voice."""
        self.audio_enabled = not self.audio_enabled
        self.pipeline.tts.enabled = self.audio_enabled
        if self.audio_enabled:
            self.audio_btn.config(text="🔊 Mute Voice TTS", bg="#37474f")
        else:
            self.audio_btn.config(text="🔇 Unmute Voice TTS", bg="#d50000")

    def clear_transcript(self):
        """Clears timeline and current subtitle."""
        self.pipeline.glosser.reset()
        self.timeline_box.delete(0, tk.END if not self._using_mock else None)
        self.subtitle_text.config(text="Transcript cleared.")
        self.token_label.config(text="Current Sign: —")
        self.conf_label.config(text="Confidence Score: 0.0%")
        self.conf_bar['value'] = 0

    def export_transcript(self, save_path=None):
        """Exports accumulated sentences and tokens to a markdown log file."""
        if not self.transcript_history:
            return None

        if save_path is None:
            if not self._using_mock and _TKINTER_AVAILABLE:
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".md",
                    filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt")],
                    title="Save Transcript"
                )
            else:
                save_path = "transcript_log.md"

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("# Sign Language Interpretation Session Log\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Dialogue Transcript:\n")
                for idx, line in enumerate(self.transcript_history, 1):
                    f.write(f"{idx}. {line}\n")
            return save_path
        return None

    def update_frame(self):
        """Retrieves and processes frame, updating canvas and telemetry."""
        if not self.running:
            return

        if not self.is_paused:
            annotated_frame, emitted_sign, conf, sentence = self.pipeline.step()

            # Update Confidence Meter
            conf_pct = conf * 100.0
            self.conf_bar['value'] = conf_pct
            tag = "HIGH" if conf_pct >= 85 else ("MED" if conf_pct >= 60 else "LOW")
            self.conf_label.config(text=f"Confidence Score: {conf_pct:.1f}% [{tag}]")

            # Update Current Sign
            pred_now = getattr(self.pipeline, 'latest_predicted_token', '')
            sign_disp = (pred_now if conf_pct >= 40 else "—") if pred_now else (self.pipeline.latest_sign or "—")
            self.token_label.config(text=f"Current Sign: {sign_disp}")
            if emitted_sign:
                self.timeline_box.insert(tk.END if not self._using_mock else -1, f"[{time.strftime('%H:%M:%S')}] {emitted_sign}")
                self.timeline_box.see(tk.END if not self._using_mock else None)

            # Update Subtitles
            if sentence:
                self.subtitle_text.config(text=sentence)
                if not self.transcript_history or self.transcript_history[-1] != sentence:
                    self.transcript_history.append(sentence)

            # Render to Canvas if Tkinter is available
            if not self._using_mock:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                canvas_w = max(320, self.canvas_label.winfo_width())
                canvas_h = max(240, self.canvas_label.winfo_height())

                h, w = rgb_frame.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))

                resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = ImageTk.PhotoImage(image=Image.fromarray(resized))
                self.canvas_label.imgtk = img
                self.canvas_label.configure(image=img)

    def _schedule_next_frame(self):
        """Schedules recurring UI update matching 30 FPS (~33ms)."""
        if self.running:
            self.update_frame()
            self.root.after(33, self._schedule_next_frame)

    def run_opencv_window(self):
        """Interactive desktop GUI using native OpenCV HighGUI rendering with a clean, fixed-resolution 1100x720 HUD."""
        window_name = "Real-Time ASL Interpretation System - Team 14"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1100, 720)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        demo_sequences = {
            ord('1'): ["HELLO", "ME", "STUDENT", "COMPUTER"],
            ord('2'): ["THANK-YOU", "HELP", "TEACHER"],
            ord('3'): ["ME", "NEED", "WATER", "PLEASE"],
            ord('4'): ["NICE", "MEET", "ALL", "YOU"]
        }

        print("\n" + "=" * 70)
        print("  Real-Time Sign Language Interpretation GUI (High-Definition HUD)")
        print("=" * 70)
        print("  Controls:")
        print("    [P]       : Toggle Pause / Resume feed")
        print("    [M]       : Toggle Neural Edge-TTS voice on/off")
        print("    [C]       : Clear transcript and subtitle HUD")
        print("    [S]       : Export transcript to markdown log file")
        print("    [1,2,3,4] : Trigger Benchmark Qualitative Demo Sequences")
        print("    [Q] / ESC : Exit application cleanly")
        print("=" * 70 + "\n")

        CANVAS_W, CANVAS_H = 1100, 720

        while self.running:
            if not self.is_paused:
                annotated_frame, emitted_sign, conf, sentence = self.pipeline.step()
                if emitted_sign:
                    self.timeline_box.insert(0, f"[{time.strftime('%H:%M:%S')}] {emitted_sign}")
                if sentence and (not self.transcript_history or self.transcript_history[-1] != sentence):
                    self.transcript_history.append(sentence)
            else:
                annotated_frame = getattr(self, '_last_frame', np.zeros((480, 640, 3), dtype=np.uint8))
                conf = self.pipeline.latest_confidence
                sentence = self.pipeline.latest_sentence

            self._last_frame = annotated_frame.copy()

            # Create clean, fixed-resolution 1100x720 background
            hud = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            hud[:, :735] = (16, 18, 24)       # Left area (dark navy)
            hud[:, 735:] = (22, 26, 36)       # Right sidebar (charcoal slate)
            cv2.line(hud, (735, 0), (735, CANVAS_H), (42, 50, 66), 2)

            # -------------------------------------------------------------
            # 1. Top Title & Status Header (Height: 46px)
            # -------------------------------------------------------------
            cv2.rectangle(hud, (0, 0), (CANVAS_W, 46), (24, 28, 38), -1)
            cv2.line(hud, (0, 46), (CANVAS_W, 46), (42, 50, 66), 1)
            cv2.putText(hud, "Real-Time ASL Assistive Interpreter - Team 14", (20, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 229, 255), 2, cv2.LINE_AA)

            status_text = "PAUSED" if self.is_paused else "LIVE 30 FPS"
            status_color = (0, 171, 255) if self.is_paused else (0, 230, 118)
            cv2.putText(hud, f"[{status_text}]", (CANVAS_W - 160, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 2. Left Video Viewport (Aspect-Ratio Preserved: 710x456 box)
            # -------------------------------------------------------------
            target_vw, target_vh = 705, 456
            vid_h, vid_w = annotated_frame.shape[:2]
            scale = min(target_vw / vid_w, target_vh / vid_h)
            scaled_w, scaled_h = int(vid_w * scale), int(vid_h * scale)
            scaled_frame = cv2.resize(annotated_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            vx = 15 + (target_vw - scaled_w) // 2
            vy = 56 + (target_vh - scaled_h) // 2
            hud[vy:vy + scaled_h, vx:vx + scaled_w] = scaled_frame
            cv2.rectangle(hud, (15, 56), (15 + target_vw, 56 + target_vh), (45, 55, 75), 1)

            # -------------------------------------------------------------
            # 3. Dynamic Subtitle HUD Card (Below Video, y: 522 to 705)
            # -------------------------------------------------------------
            card_x1, card_y1, card_x2, card_y2 = 15, 522, 15 + target_vw, 705
            cv2.rectangle(hud, (card_x1, card_y1), (card_x2, card_y2), (28, 34, 46), -1)
            cv2.rectangle(hud, (card_x1, card_y1), (card_x2, card_y2), (0, 200, 255), 1)

            cv2.putText(hud, "DYNAMIC SUBTITLE HUD (TRANSLATED ENGLISH):", (card_x1 + 15, card_y1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 216, 128), 1, cv2.LINE_AA)

            display_sentence = sentence if sentence else "Awaiting ASL continuous gesture stream..."
            # Word-wrap sentence if long
            if len(display_sentence) > 48:
                line1 = display_sentence[:48]
                line2 = display_sentence[48:96]
                cv2.putText(hud, f"\"{line1}", (card_x1 + 15, card_y1 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(hud, f"{line2}\"", (card_x1 + 15, card_y1 + 92),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(hud, f"\"{display_sentence}\"", (card_x1 + 15, card_y1 + 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            # Recognized Gloss Tokens Sub-row
            tokens = self.pipeline.glosser.get_tokens()
            tokens_str = "  ".join([f"[{t}]" for t in tokens[-6:]]) if tokens else "[No active tokens]"
            cv2.putText(hud, f"Recognized Glosses: {tokens_str}", (card_x1 + 15, card_y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 229, 255), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 4. Right Sidebar: Telemetry, Meters & Controls (x: 745 to 1085)
            # -------------------------------------------------------------
            side_x = 750

            # Telemetry Header
            cv2.putText(hud, "REAL-TIME TELEMETRY", (side_x, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 215, 255), 2, cv2.LINE_AA)

            # Confidence Meter
            conf_pct = conf * 100.0
            tag = "HIGH" if conf_pct >= 85 else ("MED" if conf_pct >= 60 else "LOW")
            conf_color = (0, 230, 118) if conf_pct >= 85 else ((0, 229, 255) if conf_pct >= 60 else (0, 150, 255))

            cv2.putText(hud, f"Confidence: {conf_pct:.1f}% [{tag}]", (side_x, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, conf_color, 1, cv2.LINE_AA)

            # Progress Bar for Confidence
            bar_w = 325
            cv2.rectangle(hud, (side_x, 122), (side_x + bar_w, 134), (35, 42, 55), -1)
            fill_w = int(bar_w * (conf_pct / 100.0))
            if fill_w > 0:
                cv2.rectangle(hud, (side_x, 122), (side_x + fill_w, 134), conf_color, -1)

            # Active Sign Card
            cv2.rectangle(hud, (side_x, 150), (side_x + bar_w, 205), (28, 34, 46), -1)
            cv2.rectangle(hud, (side_x, 150), (side_x + bar_w, 205), (45, 55, 75), 1)

            pred_now = getattr(self.pipeline, 'latest_predicted_token', '')
            sign_disp = (pred_now if conf_pct >= 40 else "—") if pred_now else (self.pipeline.latest_sign or "—")
            cv2.putText(hud, "CURRENT SIGN DETECTED:", (side_x + 12, 172),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(hud, f"[ {sign_disp} ]", (side_x + 12, 197),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2, cv2.LINE_AA)

            # Audio Status
            tts_status = "ACTIVE (UNMUTED)" if self.audio_enabled else "MUTED"
            tts_color = (0, 230, 118) if self.audio_enabled else (0, 100, 255)
            cv2.putText(hud, f"Voice Audio TTS:  {tts_status}", (side_x, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, tts_color, 1, cv2.LINE_AA)

            # Gloss Token Timeline Section
            cv2.line(hud, (side_x, 248), (side_x + bar_w, 248), (42, 50, 66), 1)
            cv2.putText(hud, "RECENT DETECTED GLOSSES:", (side_x, 275),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 215, 255), 2, cv2.LINE_AA)

            # Timeline box (Display up to 6 items)
            recent_tokens = tokens[-6:] if tokens else []
            tl_y = 305
            if not recent_tokens:
                cv2.putText(hud, "  (No signs recognized yet)", (side_x + 5, tl_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 130, 145), 1, cv2.LINE_AA)
            else:
                for idx, t_sign in enumerate(reversed(recent_tokens)):
                    dot_color = (0, 230, 118) if idx == 0 else (0, 229, 255)
                    cv2.putText(hud, f"● {t_sign}", (side_x + 10, tl_y + (idx * 26)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, dot_color, 1, cv2.LINE_AA)

            # Interactive Controls Guide Card
            cv2.line(hud, (side_x, 500), (side_x + bar_w, 500), (42, 50, 66), 1)
            cv2.putText(hud, "SESSION CONTROLS:", (side_x, 525),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 215, 255), 2, cv2.LINE_AA)

            controls = [
                ("[P]", "Pause / Resume Video Stream"),
                ("[F]", "Toggle Mirror / Flip Mode"),
                ("[M]", "Mute / Unmute Voice TTS"),
                ("[C]", "Clear Subtitles & History"),
                ("[S]", "Export Transcript to Markdown"),
                ("[1-4]", "Trigger Demo Dialogue Sentences"),
                ("[Q]", "Exit Application Cleanly"),
            ]
            for c_idx, (key_label, desc) in enumerate(controls):
                cy = 546 + (c_idx * 22)
                cv2.putText(hud, key_label, (side_x + 5, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 229, 255), 1, cv2.LINE_AA)
                cv2.putText(hud, f"- {desc}", (side_x + 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1, cv2.LINE_AA)

            cv2.imshow(window_name, hud)
            key = cv2.waitKey(25) & 0xFF

            if key in [ord('q'), 27]:
                break
            elif key == ord('p'):
                self.toggle_pause()
            elif key == ord('f'):
                if hasattr(self.pipeline, 'capture') and hasattr(self.pipeline.capture, 'mirror'):
                    self.pipeline.capture.mirror = not self.pipeline.capture.mirror
                    state_str = "ON" if self.pipeline.capture.mirror else "OFF"
                    print(f"[VIDEO] Camera Mirroring (Flip): {state_str}")
            elif key == ord('m'):
                self.toggle_audio()
            elif key == ord('c'):
                self.clear_transcript()
                self.pipeline.latest_sentence = ""
            elif key == ord('s'):
                saved = self.export_transcript("transcript_session_log.md")
                print(f"Transcript exported to {saved}")
            elif key in demo_sequences:
                tokens_seq = demo_sequences[key]
                print(f"\n[DEMO INJECTION] Executing Sequence: {tokens_seq}")
                self.pipeline.glosser.reset()
                for t in tokens_seq:
                    self.pipeline.glosser.emitted_tokens.append(t)
                trans = self.pipeline.translator.translate_accumulated(tokens_seq)
                self.pipeline.latest_sentence = trans
                self.pipeline.latest_sign = tokens_seq[-1]
                self.pipeline.latest_confidence = 0.965
                self.pipeline.tts.speak_async(trans)

        cv2.destroyAllWindows()
        self.close()


    def close(self):
        """Gracefully shuts down pipeline and releases system resources."""
        self.running = False
        self.pipeline.stop()
        if not self._using_mock and hasattr(self.root, 'destroy'):
            try:
                self.root.destroy()
            except Exception:
                pass
