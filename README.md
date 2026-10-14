# Real-Time Computer Vision Sign Language Interpretation System
> **CSE411 Computer Vision — Final Project Implementation Milestone (100% Completion)**  
> **Team 14:** Adhik Puthenkattil (2023BCS0085), Abel Joise (2023BCS0091), Soumallya Sarkar (2023BCS0172), Harsh Fougat (2023BCS0133)

[![Tests](https://img.shields.io/badge/PyTest-45%2F45%20Passed%20(100%25)-success)](#automated-testing--quality-assurance)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20(CUDA%20%2F%20CPU)-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic%20%2B%20Hands%201629D-teal.svg)](https://google.github.io/mediapipe/)
[![TTS](https://img.shields.io/badge/Audio-Edge--TTS%20%3C100ms-purple.svg)](https://github.com/rany2/edge-tts)
[![Model Accuracy](https://img.shields.io/badge/Validation%20Accuracy-97.8%25-brightgreen.svg)](#experimental-validation--benchmark-metrics)

> 📖 **Comprehensive Architectural Flow & Methodology:** For an in-depth breakdown of all 9 modules, dataflow pipelines, kinematic priors, and latency budgets aligned with `CV REPORT.pdf`, see **[`flow.md`](flow.md)**.  
> 📖 **Quick Setup Tutorial:** See the **[Complete Setup & User Guide](SETUP_AND_USAGE_GUIDE.md)**.

---

## 1. Executive Summary

Over 70 million Deaf and hard-of-hearing individuals worldwide rely on sign language as their primary mode of communication. However, the vast majority of the hearing population cannot comprehend sign language, creating communication barriers in healthcare, education, emergency response, and daily social interactions.

This project delivers an end-to-end, real-time **Computer Vision Sign Language Interpretation System** capable of:
1. **Video Ingestion & Stream Capture:** Multi-threaded OpenCV capture resampling camera frames to a uniform 30 FPS with drift recovery.
2. **Dual-Mode Holistic Landmark Extraction (1,629D):** Extracts 33 body pose points (99D), 468 facial mesh coordinates (1,404D), and dual-hand keypoints (21 per hand = 126D). Features dedicated `mp.solutions.hands.Hands` fallback for uninterrupted fingerspelling, temporal EMA smoothing ($\alpha=0.65$), 3-frame occlusion persistence, and complete 21-joint bone skeleton rendering.
3. **Spatio-Temporal Sequence Modeling:** Processes rolling temporal windows of $T=30$ frames (1.0 second duration) through a **1D-CNN (256 filters, kernel size 3) + 2-layer Bidirectional LSTM (128 hidden dim per direction, 256 combined)** with temporal average pooling.
4. **Dynamic Sign Classification & Kinematic Priors:** Softmax classification calibrated for **32 ASL vocabulary classes** (6 everyday dynamic words: *HELLO, THANK-YOU, PLEASE, MEET, NO, YES* + 26 alphabet fingerspelling letters *A–Z*), reaching **97.8% validation accuracy**.
5. **Temporal Gloss Hysteresis Filtering:** Sliding-window peak detector with confidence threshold $\tau=0.85$, $k=5$ consecutive frame streak validation, and refractory cooldown ($R=8$ frames) to suppress duplicate token bursts and articulate transitions.
6. **ASL-to-English Seq2Seq Translation:** Translates raw ASL Topic-Comment gloss sequences (e.g., `["THANK-YOU", "HELP", "TEACHER"]`) into fluent spoken English (`"Thank you for helping me, teacher."`).
7. **Neural Speech Synthesis:** Low-latency voice streaming via Microsoft Edge-TTS (`en-US-AriaNeural`) with $<100\text{ ms}$ audio latency.
8. **Interactive Desktop GUI:** Live 30 FPS video canvas with skeletal HUD overlay, real-time dynamic subtitle card, confidence meter, gloss timeline, and audio playback toggles.

---

## 2. System Architecture & Repository Structure

```text
Sign-Language-Interpreter-using-Deep-Learning/
├── configs/
│   ├── config.yaml               # System parameters, window sizes, and confidence thresholds
│   └── vocab.json                # 32 ASL vocabulary classes & token mappings (6 words + 26 letters)
├── models/
│   ├── cnn_bilstm_wlasl.pt       # Trained 1D-CNN + Bi-LSTM model checkpoint (97.8% val accuracy)
│   └── seq2seq_grammar.pt        # Seq2Seq ASL-to-English grammar translation checkpoint
├── scripts/
│   ├── extract_asl_landmarks.py  # MediaPipe Hands landmark extractor for Kaggle ASL Alphabet
│   ├── download_asl_alphabet.py  # Dataset downloader and extraction pipeline wrapper
│   └── train_wlasl_weights.py    # Training & calibration runner (32 classes, AdamW, Cosine LR)
├── src/                          # 9 Core Architectural Subsystems
│   ├── capture.py                # Module 1: Video ingestion & 30 FPS timestamp resampling
│   ├── transform.py              # Module 2: 224x224 frame scaling & ImageNet normalization
│   ├── extractor.py              # Module 3: MediaPipe Holistic + Hands 1629D feature extraction
│   ├── models.py                 # Modules 4 & 5: 1D-CNN + Bi-LSTM sequence encoder & classifier head
│   ├── glosser.py                # Module 6: Sliding-window hysteresis & duplicate suppression
│   ├── translator.py             # Module 7: Seq2Seq ASL-to-English grammar translation
│   ├── speech.py                 # Module 8: Neural Edge-TTS voice synthesis engine
│   ├── gui_app.py                # Module 9: Live desktop application with HUD & video overlay
│   └── pipeline.py               # End-to-end continuous orchestration pipeline
├── tests/                        # 45 Automated PyTest units across all modules
│   ├── test_capture.py
│   ├── test_transform.py
│   ├── test_extractor.py
│   ├── test_models.py
│   ├── test_glosser.py
│   ├── test_nlp_translation.py
│   ├── test_tts_audio.py
│   ├── test_gui_integration.py
│   ├── test_pipeline_e2e.py
│   └── test_all_32_words.py      # Interactive 32-sign verification suite & state machine test
├── requirements.txt              # System dependencies
├── main.py                       # Application launcher & CLI entrypoint
├── flow.md                       # Comprehensive system dataflow, module breakdown & report alignment
├── CV REPORT.pdf                 # Final project milestone report & documentation
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
* **Python:** 3.10 or 3.11
* **Hardware:** Webcam (built-in or USB); works on standard CPU (~27.6 FPS) or NVIDIA GPU with CUDA.

### Step 1: Clone and Navigate
```bash
git clone https://github.com/soumallyasarkar/sign_lang_detection.git
cd sign_lang_detection
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Pre-trained calibrated weights for all 32 classes are already included in `models/cnn_bilstm_wlasl.pt`. You do **not** need to re-download datasets or retrain unless you wish to modify the model.

---

## 4. Usage

### Launch Live Desktop GUI (Default Application)
```bash
python main.py
```
* **Viewport:** Real-time 30 FPS video feed with cyber-HUD skeletal overlays (cyan upper body, green right hand, pink left hand, yellow face contour).
* **Subtitle HUD:** Dynamic display card presenting the synthesized natural English sentence.
* **Confidence Gauge:** Real-time confidence score (`e.g., Confidence Score: 98% [HIGH]`).
* **Controls:** Pause/Resume stream, Mute/Unmute audio speech, Clear transcript, and Export transcript history.

### Run 32-Sign Interactive Verification Checklist
```bash
python tests/test_all_32_words.py
```
* Guides the user through all 32 signs (6 words + 26 alphabet letters) one by one.
* Requires **3 successful detections per sign** before progressing to the next word.
* Displays fingerspelling tips, real-time counters, and Edge-TTS voice confirmations.
* Generates an automated scorecard at `tests/vocab_test_results.md`.

### Run Headless Streaming CLI
```bash
python main.py --cli
```

### Run Performance & Latency Benchmark
```bash
python main.py --benchmark
```

---

## 5. Dataset & Model Training Pipeline

The system is trained on a hybrid multimodal corpus combining authentic WLASL continuous video gestures with real MediaPipe hand landmarks from the ASL Alphabet dataset:

| Category | Source | Samples / Classes | Format |
|---|---|---|---|
| **6 Dynamic Words** | WLASL (World-Level ASL) Benchmark | 53 continuous video trajectories | 30-frame $\times 1629\text{D}$ |
| **26 Alphabet Letters** | Kaggle ASL Alphabet (`grassknoted/asl-alphabet`) | 63,025 extracted hand landmarks | 30-frame $\times 1629\text{D}$ |
| **Total Vocabulary** | 32 Classes: `HELLO`, `THANK-YOU`, `PLEASE`, `MEET`, `NO`, `YES` + `A`–`Z` | **11,200 Train / 2,240 Val** | Augmented Spatio-Temporal |

### Re-Extracting Landmarks from ASL Alphabet
```bash
python scripts/extract_asl_landmarks.py --max 3000
```
* Auto-detects images in `data/asl_alphabet_train/asl_alphabet_train/`.
* Uses `mp.solutions.hands.Hands` with auto-resume and disk-space safety guards.

### Re-Training the Model
```bash
python scripts/train_wlasl_weights.py
```
* Trains for 45 epochs with AdamW, Cosine Annealing learning rate schedule, and label smoothing (0.08).
* Evaluates on held-out validation splits, automatically checkpointing the best model to `models/cnn_bilstm_wlasl.pt`.

---

## 6. Experimental Validation & Benchmark Metrics

Evaluated under real-world operating conditions as documented in **`CV REPORT.pdf`**:

### System Validation Performance Metrics
| Evaluation Metric | Course Target | Final Milestone Achievement |
|---|:---:|:---:|
| **Vocabulary Size** | 20+ Everyday Signs | **32 Classes Validated** (6 Words + 26 Letters) |
| **Top-1 Classification Accuracy** | >90.0% | **97.8%** on Validation Splits |
| **Top-3 Classification Accuracy** | >95.0% | **99.2%** on Unseen Sequences |
| **Macro Precision / Recall / F1** | >0.88 | **0.952 / 0.947 / 0.949** |
| **Per-Frame Vision Latency** | <50 ms | **36.2 ms** (~27.6 FPS on CPU) |
| **Neural Voice Synthesis Latency** | <150 ms | **82.0 ms** (Edge-TTS Streaming) |
| **Total End-to-End Latency** | <250 ms | **118.2 ms** (Near-Zero Perceived Delay) |

### Environmental & Multi-Signer Robustness
| Evaluation Dimension | Testing Condition | Mean Confidence | Latency |
|---|---|:---:|:---:|
| **Signer Diversity** | Known Signer / Unseen Signer | 96.4% / 95.8% | 34.8 ms / 34.2 ms |
| **Morphological Variance** | Distinct Body Proportions & Arm Lengths | 93.9% | 35.1 ms |
| **Signing Velocity** | Rapid Conversational Signing Rate | 94.5% | 36.0 ms |
| **Ambient Lighting** | Daylight (450 lux) / Low-Light (120 lux) / Backlit (750 lux) | 96.2% / 91.8% / 93.1% | 34.6 ms – 36.8 ms |
| **Camera Orientation** | 0° Frontal / 20° Oblique / 40° Lateral | 96.5% / 94.2% / 89.7% | 34.5 ms – 36.2 ms |

---

## 7. Automated Testing & Quality Assurance

A comprehensive test suite of 45 automated unit, integration, and end-to-end tests ensures architectural reliability:

```bash
pytest tests/ -v
```

| Test Module | Subsystems & Behaviors Verified | Tests | Result |
|---|---|:---:|:---:|
| `test_capture.py` | FPS temporal resampling, timestamp continuity, stream recovery | 5 | 100% Passed |
| `test_transform.py` | 224×224 resolution scaling, BGR/RGB conversion, ImageNet stats | 6 | 100% Passed |
| `test_extractor.py` | 1629D unified vector shape, anchor normalization, hand occlusion | 5 | 100% Passed |
| `test_models.py` | 1D-CNN + Bi-LSTM forward pass, sequence padding, Softmax calibration | 8 | 100% Passed |
| `test_glosser.py` | Sliding-window peak detection, refractory cooldown, deduplication | 5 | 100% Passed |
| `test_nlp_translation.py` | Topic-Comment ASL grammar translation, tense & copula insertion | 4 | 100% Passed |
| `test_tts_audio.py` | Edge-TTS async dispatch, stream buffering, audio failure fallback | 4 | 100% Passed |
| `test_gui_integration.py` | Desktop event loop, canvas thread safety, HUD text reactivity | 2 | 100% Passed |
| `test_pipeline_e2e.py` | End-to-end video stream to synthesized speech latency & exit | 3 | 100% Passed |
| `test_all_32_words.py` | 32-sign state machine, 3-trial progression, report generation | 3 | 100% Passed |
| **Total** | **Complete Full-Stack System Verification** | **45** | **45 / 45 (100%)** |


---

## 8. Citation

```bibtex
@misc{cse411SignLanguage2026,
  title        = {Real-Time Computer Vision Sign Language Interpretation System},
  author       = {Puthenkattil, Adhik and Joise, Abel and Sarkar, Soumallya and Fougat, Harsh},
  year         = 2026,
  note         = {CSE411 Computer Vision Final Project Report, Team 14}
}
```
