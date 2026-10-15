"""Continuous Interpretation Pipeline (src/pipeline.py).

End-to-end continuous orchestration runner integrating Modules 1 through 8
with synchronized outputs for graphical HUD and auditory speech channels.
"""

import collections
import json
import os
import cv2
import numpy as np
import torch

from src.capture import VideoStreamIngestor
from src.transform import FrameNormalizer
from src.extractor import HolisticLandmarkExtractor
from src.models import SpatioTemporalSequenceModel
from src.glosser import TemporalHysteresisGlosser
from src.translator import ASLSeq2SeqTranslator
from src.speech import NeuralSpeechSynthesizer


class ContinuousInterpretationPipeline:
    """End-to-end streaming orchestrator integrating Modules 1 through 8."""

    def __init__(self, config=None):
        if config is None:
            config = {}

        # Config extraction
        sys_cfg = config.get('system', {})
        ext_cfg = config.get('extractor', {})
        mod_cfg = config.get('model', {})
        glo_cfg = config.get('glosser', {})
        tra_cfg = config.get('translator', {})
        tts_cfg = config.get('tts', {})

        self.target_fps = sys_cfg.get('target_fps', 30)
        self.camera_idx = sys_cfg.get('camera_idx', 0)
        self.window_size = mod_cfg.get('temporal_window', 30)

        # Load vocabulary
        self.vocab = self._load_vocab()

        # Initialize 8 core backend modules
        self.mirror_camera = sys_cfg.get('mirror_camera', True)
        self.capture = VideoStreamIngestor(src=self.camera_idx, target_fps=self.target_fps, mirror=self.mirror_camera)
        self.normalizer = FrameNormalizer(target_size=tuple(sys_cfg.get('input_resolution', [224, 224])))
        self.extractor = HolisticLandmarkExtractor()

        # Device detection (NVIDIA CUDA GPU or CPU)
        target_device = sys_cfg.get('device', 'auto')
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if target_device == "auto" else target_device

        # Model loader
        weights_path = mod_cfg.get('weights_path', 'models/cnn_bilstm_wlasl.pt')
        self.model = SpatioTemporalSequenceModel.load(
            filepath=weights_path,
            in_features=ext_cfg.get('feature_dim', 1629),
            num_classes=len(self.vocab),
            device=self.device
        )

        # Glosser, Translator, TTS
        self.glosser = TemporalHysteresisGlosser(
            thresh=glo_cfg.get('threshold', 0.85),
            min_consecutive=glo_cfg.get('min_consecutive', 5),
            refractory_period=glo_cfg.get('refractory_period', 8)
        )
        self.translator = ASLSeq2SeqTranslator(weights_path=tra_cfg.get('weights_path'))
        self.tts = NeuralSpeechSynthesizer(
            voice=tts_cfg.get('voice', 'en-US-AriaNeural'),
            enabled=tts_cfg.get('enabled', True)
        )

        # Rolling temporal sequence buffer (T=30)
        self.window_buffer = collections.deque(maxlen=self.window_size)
        self.latest_sentence = ""
        self.latest_sign = ""
        self.latest_predicted_token = ""
        self.latest_confidence = 0.0

    def _load_vocab(self):
        """Loads vocabulary list from configs/vocab.json or defaults to 28 signs."""
        vocab_path = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab.json")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    id_to_token = data.get("id_to_token", {})
                    sorted_ids = sorted(map(int, id_to_token.keys()))
                    return [id_to_token[str(i)] for i in sorted_ids]
            except Exception:
                pass
        return ["SIGN_" + str(i) for i in range(28)]

    def start(self):
        """Starts video capture ingestion."""
        self.capture.start()
        return self

    def stop(self):
        """Stops video capture and TTS synthesis."""
        self.capture.stop()
        self.tts.stop()
        self.extractor.close()

    def process_frame(self, frame_bgr):
        """Processes a single raw BGR frame through the full 8-module pipeline.

        Returns:
            annotated_frame: BGR frame with skeletal overlays
            emitted_sign: Detected sign token (or "")
            confidence: Float confidence (0.0 to 1.0)
            sentence: Grammatically translated English sentence (or latest)
        """
        # Module 2: Preprocessing
        norm_img, _ = self.normalizer(frame_bgr)

        # Module 3: Holistic Feature Extraction
        # Feed high-resolution RGB to MediaPipe for sharp hand keypoint tracking without letterbox distortion
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        vec_1629d, annotated_frame = self.extractor.extract(
            full_rgb, frame_bgr, is_mirrored=getattr(self.capture, 'mirror', True)
        )

        # Temporal sequence window aggregation
        self.window_buffer.append(vec_1629d)

        emitted_sign = ""
        conf = 0.0

        if len(self.window_buffer) == self.window_size:
            window_arr = np.array(self.window_buffer, dtype=np.float32)

            # Activity Gate: Require active hand presence for signing (prevents idle ghost detections)
            rh_active = not np.all(vec_1629d[1566:1629] == 0.0)
            lh_active = not np.all(vec_1629d[1503:1566] == 0.0)
            if not rh_active and not lh_active:
                self.latest_confidence = 0.0
                self.latest_predicted_token = ""
                return annotated_frame, "", 0.0, self.latest_sentence

            # Modules 4 & 5: Spatio-Temporal Model Inference
            pred_token, conf = self.model.predict_window(window_arr, self.vocab)
            self.latest_confidence = conf
            self.latest_predicted_token = pred_token

            # Module 6: Temporal Glosser & Hysteresis Filtering
            new_emitted = self.glosser.update(pred_token, conf)
            if new_emitted:
                emitted_sign = new_emitted
                self.latest_sign = emitted_sign

                # Module 7: ASL Seq2Seq NLP Grammar Translation
                tokens = self.glosser.get_tokens()
                translated = self.translator.translate_accumulated(tokens)
                if translated:
                    self.latest_sentence = translated
                    # Module 8: Neural Edge-TTS Speech Synthesis
                    self.tts.speak_async(translated)

        return annotated_frame, emitted_sign, self.latest_confidence, self.latest_sentence

    def step(self):
        """Reads latest frame from capture and processes through pipeline."""
        ts, frame = self.capture.read()
        return self.process_frame(frame)
