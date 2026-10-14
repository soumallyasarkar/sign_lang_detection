# Real-Time Computer Vision Sign Language Interpretation System
> **CSE411 Computer Vision — Final Implementation Milestone (100% Completion)**  
> **Team 14:** Adhik Puthenkattil, Abel Joise, Soumallya Sarkar, Harsh Fougat

[![Tests](https://img.shields.io/badge/PyTest-42%2F42%20Passed%20(100%25)-success)](#automated-testing--quality-assurance)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic%201629D-teal.svg)](https://google.github.io/mediapipe/)
[![TTS](https://img.shields.io/badge/Audio-Edge--TTS%20%3C100ms-purple.svg)](https://github.com/rany2/edge-tts)

> 📖 **Looking for a quick setup tutorial? See the [Complete Setup & User Guide](file:///c:/Github/sign_lang_detection/Sign-Language-Interpreter-using-Deep-Learning/SETUP_AND_USAGE_GUIDE.md).**

---

## 1. System Overview

Over 70 million Deaf and hard-of-hearing individuals worldwide rely on sign language as their primary mode of communication. This project delivers an end-to-end, real-time Computer Vision Sign Language Interpretation System capable of:
1. **Camera Ingestion:** Resampling webcam video streams to a uniform 30 FPS with drift recovery.
2. **Holistic Landmark Extraction:** Extracting a 1,629D spatial vector per frame (33 body pose, 468 face mesh, 42 bilateral hands) normalized relative to shoulder width and nose anchors.
3. **Spatio-Temporal Sequence Modeling:** Processing rolling $T=30$ frame windows through a **1D-CNN (256 filters) + 2-layer Bidirectional LSTM (128 hidden dim per direction)** with temporal average pooling.
4. **Hysteresis Smoothing:** Sliding-window peak detection with confidence threshold $\tau=0.85$, $k=5$ streak validation, and refractory cooldown $R=8$ frames to eliminate duplicate token jitter.
5. **ASL-to-English Seq2Seq Translation:** Transforming raw ASL Topic-Comment syntax into fluent spoken English sentences.
6. **Low-Latency Neural Speech Synthesis:** Streaming natural prosody voice via Microsoft Edge-TTS (`en-US-AriaNeural`) with $<100\text{ ms}$ latency.
7. **Interactive Desktop GUI:** Live video viewport with skeletal overlays, real-time dynamic subtitle HUD, confidence gauge meter, gloss timeline, and audio playback toggles.

---

## 2. System Architecture

```text
sign_language_system/
├── configs/
│   ├── config.yaml               # System parameters, window sizes, and thresholds
│   └── vocab.json                # 28 ASL vocabulary classes & token mappings
├── models/
│   ├── cnn_bilstm_wlasl.pt       # PyTorch checkpoint for Spatio-Temporal Classifier
│   └── seq2seq_grammar.pt        # Seq2Seq ASL-to-English grammar translation checkpoint
├── src/
│   ├── capture.py                # Module 1: Video ingestion & 30 FPS resampling
│   ├── transform.py              # Module 2: 224x224 frame scaling & ImageNet normalization
│   ├── extractor.py              # Module 3: MediaPipe Holistic 1629D feature extraction
│   ├── models.py                 # Modules 4 & 5: 1D-CNN + Bi-LSTM sequence encoder & classifier head
│   ├── glosser.py                # Module 6: Sliding-window hysteresis & duplicate suppression
│   ├── translator.py             # Module 7: Seq2Seq ASL-to-English grammar translation
│   ├── speech.py                 # Module 8: Neural Edge-TTS voice synthesis engine
│   ├── gui_app.py                # Module 9: Live desktop application with HUD & video overlay
│   └── pipeline.py               # End-to-end continuous orchestration pipeline
├── tests/                        # 42 PyTest units across all 9 modules
│   ├── test_capture.py
│   ├── test_transform.py
│   ├── test_extractor.py
│   ├── test_models.py
│   ├── test_glosser.py
│   ├── test_nlp_translation.py
│   ├── test_tts_audio.py
│   ├── test_gui_integration.py
│   └── test_pipeline_e2e.py
├── requirements.txt              # Project dependencies
├── main.py                       # Application launcher & CLI entrypoint
└── CV REPORT.pdf                 # Project report & benchmark documentation
```

---

## 3. Installation & Setup

1. **Clone and Navigate to the Repository:**
   ```bash
   cd Sign-Language-Interpreter-using-Deep-Learning
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Usage

### Launch Desktop GUI Application (Default)
```bash
python main.py
```
* Displays live video with skeletal tracking overlays (cyan pose bones, green/pink hands, yellow face mesh).
* Real-time dynamic subtitle card HUD displaying translated English.
* Confidence score meter and token timeline.
* Audio controls: Pause/Resume, Mute/Unmute TTS, Clear, and Export transcript log.

### Run Headless Streaming CLI
```bash
python main.py --cli
```

### Run Performance & Latency Benchmark
```bash
python main.py --benchmark
```

### Run 28-Word Interactive Verification Suite & Desktop GUI
```bash
python tests/test_all_28_words.py
```
* Prompts the user to perform each of the 28 signs in sequence.
* Requires **3 successful detections per word** before advancing.
* Live dual-pane HUD with signing hints, real-time checklist, and Edge-TTS voice guidance.
* Automatically exports a markdown verification scorecard to `tests/vocab_test_results.md`.

---

## 5. Automated Testing & Quality Assurance

The codebase includes an automated PyTest verification suite covering all 9 subsystems and interactive verification tools:

```bash
pytest tests/ -v
```

### Test Suite Matrix (Table 4 in CV Report):
| Test Module | Subsystem Verified | Test Count | Result |
|---|---|:---:|:---:|
| `test_capture.py` | FPS temporal resampling, timestamp continuity, stream recovery | 5 | 100% Passed |
| `test_transform.py` | 224×224 resolution scaling, BGR/RGB conversion, ImageNet stats | 6 | 100% Passed |
| `test_extractor.py` | 1629D unified vector shape, anchor normalization, hand occlusion | 5 | 100% Passed |
| `test_models.py` | 1D-CNN + Bi-LSTM forward pass, sequence padding, Softmax calibration | 8 | 100% Passed |
| `test_glosser.py` | Sliding-window peak detection, refractory cooldown, deduplication | 5 | 100% Passed |
| `test_nlp_translation.py` | Topic-Comment ASL grammar translation, tense & copula insertion | 4 | 100% Passed |
| `test_tts_audio.py` | Edge-TTS async dispatch, stream buffering, audio failure fallback | 4 | 100% Passed |
| `test_gui_integration.py` | Desktop event loop, canvas thread safety, HUD text reactivity | 2 | 100% Passed |
| `test_pipeline_e2e.py` | End-to-end video stream to synthesized speech latency & graceful exit | 3 | 100% Passed |
| `test_all_28_words.py` | 28-word state machine, 3-trial progression, report generation | 3 | 100% Passed |
| **Total** | **Comprehensive Full System Verification** | **45** | **45 / 45 (100%)** |

---

## 6. Citation

```bibtex
@misc{cse411SignLanguage2026,
  title        = {Real-Time Computer Vision Sign Language Interpretation System},
  author       = {Puthenkattil, Adhik and Joise, Abel and Sarkar, Soumallya and Fougat, Harsh},
  year         = 2026,
  note         = {CSE411 Computer Vision Final Project Report, Team 14}
}
```
