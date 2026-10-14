"""Unit tests for Module 8 (src/speech.py).

Verifies 4 target behaviors:
1. Edge-TTS non-blocking asynchronous dispatch
2. Voice configuration compliance (en-US-AriaNeural)
3. Audio failure fallback handling (safe execution on headless/mock environments)
4. Audio playback stoppage and state cleanup
"""

import time
from src.speech import NeuralSpeechSynthesizer, play_audio_stream


def test_tts_async_dispatch_non_blocking():
    """Test 1: speak_async() dispatches work without blocking the caller."""
    tts = NeuralSpeechSynthesizer(enabled=True)
    t0 = time.perf_counter()
    tts.speak_async("Hello, I am testing voice synthesis.")
    elapsed = time.perf_counter() - t0
    # Dispatch must return almost immediately (< 50 ms)
    assert elapsed < 0.05
    tts.stop()


def test_tts_voice_configuration():
    """Test 2: Configures neural voice matching specification."""
    tts = NeuralSpeechSynthesizer(voice="en-US-AriaNeural")
    assert tts.voice == "en-US-AriaNeural"
    assert tts.enabled is True


def test_tts_failure_fallback_and_empty_text():
    """Test 3: Handles missing files or empty text safely without exceptions."""
    tts = NeuralSpeechSynthesizer(enabled=True)
    tts.speak_async("")
    tts.speak_async("   ")
    play_audio_stream("non_existent_audio_file.mp3")
    assert True


def test_tts_stop_and_cleanup():
    """Test 4: stop() cleanly resets worker state."""
    tts = NeuralSpeechSynthesizer(enabled=True)
    tts.speak_async("Test voice stop.")
    tts.stop()
    assert tts.is_busy is False
