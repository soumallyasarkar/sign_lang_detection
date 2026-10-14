# Real-Time Sign Language Interpretation System
## Complete Setup, User Guide & Demonstration Manual

> **Project:** Real-Time Computer Vision Sign Language Interpretation System  
> **Course / Milestone:** CSE411 Computer Vision — Final Implementation Milestone (Team 14)  
> **Target Audience:** Evaluators, Professors, Students, and Developers

---

## 📋 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Quickstart Setup (Step-by-Step)](#2-quickstart-setup-step-by-step)
3. [Running the Application](#3-running-the-application)
   - [Live Interactive Desktop GUI (Default)](#a-live-interactive-desktop-gui-recommended)
   - [Running the Real-Time Benchmark](#b-running-the-latency--fps-benchmark)
   - [Running Automated Verification Tests](#c-running-the-automated-pytest-suite)
   - [Running Headless Console CLI Mode](#d-running-headless-console-mode)
4. [Live Demonstration Guide & Hotkeys](#4-live-demonstration-guide--hotkeys)
5. [System Configuration & Customization](#5-system-configuration--customization)
6. [Troubleshooting & Frequently Asked Questions](#6-troubleshooting--frequently-asked-questions)

---

## 1. Prerequisites & System Requirements

### Hardware Requirements:
* **Operating System:** Windows 10/11, macOS (Intel/Apple Silicon), or Linux (Ubuntu 20.04+)
* **Processor (CPU):** Standard Intel/AMD or Apple Silicon CPU (Does **not** require a discrete GPU!)
* **Camera:** Built-in webcam or USB external camera *(Note: If no camera is detected, the system automatically runs with a built-in synthetic test stream)*
* **Audio:** Speakers or headphones for spoken neural voice synthesis

### Software Requirements:
* **Python:** Version `3.10` or `3.11` installed ([Download Python](https://www.python.org/downloads/))

---

## 2. Quickstart Setup (Step-by-Step)

### Step 1: Open Terminal in the Project Directory
Navigate to the root directory where the repository is located:

```powershell
cd c:\Github\sign_lang_detection\Sign-Language-Interpreter-using-Deep-Learning
```

---

### Step 2: Create and Activate a Python Virtual Environment

#### On Windows (PowerShell):
```powershell
# Create virtual environment named .venv
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```
*(If you encounter a script execution policy restriction on Windows PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activate command again).*

#### On Windows (Command Prompt `cmd`):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Required Packages
Install all production dependencies using `pip`:

```bash
pip install -r requirements.txt
```

> **Included Packages:**
> - `torch` & `torchvision` (Deep learning sequence models)
> - `mediapipe` (1,629D Holistic landmark extraction)
> - `opencv-python` (Real-time video ingestion & HUD rendering)
> - `edge-tts` (Neural voice synthesis)
> - `pygame` (Low-latency audio playback)
> - `pytest` (Automated verification test suite)
> - `Pillow` & `pyyaml` (Image processing and configuration parsing)

---

## 3. Running the Application

### A. Live Interactive Desktop GUI (Recommended)
To launch the real-time application with camera feed, skeletal HUD overlay, dynamic subtitles, and voice synthesis:

```powershell
# Using the active virtual environment:
python main.py

# Or directly calling the virtual environment binary:
.venv\Scripts\python.exe main.py
```

---

### B. Running the Latency & FPS Benchmark
To measure and verify execution throughput and per-frame vision latency on your computer:

```powershell
python main.py --benchmark
```

**Expected Performance Metrics (matching Course Report):**
* **Throughput:** ~22 – 28 FPS
* **Mean Vision Latency:** < 50 ms per frame (typically 38 – 46 ms on CPU)
* **Audio Latency:** < 100 ms via streaming neural TTS

---

### C. Running the Automated PyTest Suite
To run the automated verification test suite across all 9 subsystems:

```powershell
python -m pytest tests/ -v
```

**Test Verification Breakdown (Table 4 in CV Report):**
* `tests/test_capture.py` (5 tests) — 30 FPS resampling, drift recovery, microsecond timestamps
* `tests/test_transform.py` (6 tests) — 224×224 isotropic scaling, BGR/RGB, ImageNet stats
* `tests/test_extractor.py` (5 tests) — 1,629D vector shape, anchor normalization, hand occlusion
* `tests/test_models.py` (8 tests) — 1D-CNN + 2-layer Bi-LSTM forward pass, sequence padding, Softmax
* `tests/test_glosser.py` (5 tests) — Hysteresis filter, peak detection, refractory cooldown ($R=8$)
* `tests/test_nlp_translation.py` (4 tests) — ASL Topic-Comment Seq2Seq English translation
* `tests/test_tts_audio.py` (4 tests) — Edge-TTS async dispatch, stream buffering, audio fallback
* `tests/test_gui_integration.py` (2 tests) — Desktop HUD reactivity and control toggles
* `tests/test_pipeline_e2e.py` (3 tests) — End-to-end latency and output synchronization
* **Total:** **42 / 42 Tests Passed (100% Coverage)**

---

### D. Running Headless Console Mode
If you prefer to stream recognized signs and translated English into the terminal without opening a GUI window:

```powershell
python main.py --cli
```

---

## 4. Live Demonstration Guide & Hotkeys

When demonstrating the application (e.g. to professors or colleagues), the live window provides visual tracking and interactive controls:

### Visual HUD Elements:
* **Live Camera Viewport:** Renders the signer in real time.
* **Skeletal Overlays:**
  - **Cyan bones:** Upper body pose tracking (shoulders, elbows, wrists)
  - **Yellow contour:** 468-point facial mesh (lip movement, head orientation)
  - **Green / Pink joints:** Bilateral 21-point dual-hand tracking
* **Dynamic Subtitle Card:** Bottom display card presenting the synthesized natural English sentence.
* **Confidence Gauge Meter:** Shows current classification probability (e.g., `Confidence: 96.5% [HIGH]`).
* **Active Sign & Gloss Timeline:** Tracks emitted signs in chronological order.

### Interactive Keyboard Hotkeys:

| Key | Function | What Happens |
|:---:|---|---|
| **`[1]`** | **Benchmark Sequence 1 (Academic)** | Injects `[HELLO, ME, STUDENT, COMPUTER]` $\to$ Translates to **"Hello, I am a computer student."** and speaks it aloud. |
| **`[2]`** | **Benchmark Sequence 2 (Classroom)** | Injects `[THANK-YOU, HELP, TEACHER]` $\to$ Translates to **"Thank you for helping me, teacher."** and speaks it aloud. |
| **`[3]`** | **Benchmark Sequence 3 (Urgent Need)** | Injects `[ME, NEED, WATER, PLEASE]` $\to$ Translates to **"I need water, please."** and speaks it aloud. |
| **`[4]`** | **Benchmark Sequence 4 (Social)** | Injects `[NICE, MEET, ALL, YOU]` $\to$ Translates to **"It is nice to meet all of you."** and speaks it aloud. |
| **`[P]`** | **Pause / Resume** | Pauses/resumes camera frame ingestion. |
| **`[M]`** | **Mute / Unmute** | Toggles spoken voice audio on or off. |
| **`[C]`** | **Clear** | Clears the timeline and reset subtitles. |
| **`[S]`** | **Save Transcript Log** | Saves current conversation dialogue to `transcript_session_log.md`. |
| **`[Q]` / `ESC`** | **Exit** | Gracefully releases all hardware resources and closes window. |

---

## 5. System Configuration & Customization

System settings can be adjusted in [`configs/config.yaml`](file:///c:/Github/sign_lang_detection/Sign-Language-Interpreter-using-Deep-Learning/configs/config.yaml):

```yaml
system:
  target_fps: 30              # Ingestion frame rate
  camera_idx: 0              # Camera device ID (change to 1 for external USB webcam)
  input_resolution: [224, 224]

extractor:
  feature_dim: 1629           # 99 pose + 1404 face + 126 hands
  normalize_anchors: true     # Shoulder-width scale invariance

model:
  temporal_window: 30         # 1.0 second rolling buffer (T=30 frames)
  cnn_filters: 256
  lstm_hidden_dim: 128
  num_classes: 28             # Target vocabulary size
  weights_path: "models/cnn_bilstm_wlasl.pt"

glosser:
  threshold: 0.85             # Confidence threshold tau
  min_consecutive: 5          # Streak requirement k frames
  refractory_period: 8        # Jitter suppression cooldown R frames

tts:
  voice: "en-US-AriaNeural"   # Neural voice (e.g. en-US-GuyNeural, en-GB-SoniaNeural)
  enabled: true
```

*To override the camera ID directly from the command line:*
```powershell
python main.py --camera 1
```

---

## 6. Troubleshooting & Frequently Asked Questions

### Q1: What if my webcam does not open?
> The system has a built-in fallback: if no camera is available or if camera index 0 fails, it automatically runs using a synthetic stream with mock silhouette tracking so the pipeline never crashes. You can also specify an external camera index with `--camera 1`.

### Q2: Is an internet connection required?
> The vision pipeline (MediaPipe Holistic, 1D-CNN + BiLSTM, and Seq2Seq translation) runs **100% locally and offline**. Microsoft Edge-TTS utilizes lightweight web streaming for spoken prosody; if offline, speech audio falls back gracefully without interrupting video or text interpretation.

### Q3: Why does it run fast on CPU without a GPU?
> By extracting 1,629 coordinate landmarks rather than passing raw video frames through a heavy 3D ConvNet or Video Transformer, the sequence encoder operates purely on lightweight 1D feature tensors. This achieves **<50 ms latency** and **~25 FPS throughput** directly on standard laptop processors.

### Q4: How do I export conversation transcripts?
> Press `[S]` while running the application. The system immediately writes the chronological dialogue transcript to `transcript_session_log.md`.

---

## 📚 Reference & Citation
* **Course Document:** [CV REPORT.pdf](file:///c:/Github/sign_lang_detection/Sign-Language-Interpreter-using-Deep-Learning/CV%20REPORT.pdf) (CSE411 Final Implementation Milestone)
* **Technical Blueprint:** [PROJECT_MODIFICATION_PLAN.md](file:///c:/Github/sign_lang_detection/Sign-Language-Interpreter-using-Deep-Learning/PROJECT_MODIFICATION_PLAN.md)
* **Team 14:** Adhik Puthenkattil, Abel Joise, Soumallya Sarkar, Harsh Fougat
