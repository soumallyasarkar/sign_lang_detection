"""Unit tests for Module 6 (src/glosser.py).

Verifies 5 target behaviors:
1. Low confidence suppression (confidence < 0.85 yields None)
2. Streak validation: token is emitted only after k=5 consecutive identical frames
3. Refractory cooldown period: suppresses duplicate token bursts for R=8 frames
4. Transitions between distinct consecutive signs
5. History accumulation and token buffer reset
"""

from src.glosser import TemporalHysteresisGlosser


def test_glosser_confidence_threshold_suppression():
    """Test 1: Predictions with confidence below threshold tau=0.85 are rejected."""
    glosser = TemporalHysteresisGlosser(thresh=0.85, min_consecutive=5, refractory_period=8)
    for _ in range(10):
        emitted = glosser.update("HELLO", confidence=0.75)
        assert emitted is None
    assert len(glosser.get_tokens()) == 0


def test_glosser_consecutive_streak_requirement():
    """Test 2: Token is emitted exactly on the k=5th consecutive identical frame."""
    glosser = TemporalHysteresisGlosser(thresh=0.85, min_consecutive=5, refractory_period=8)

    # First 4 frames: candidate accumulates but does not emit
    for i in range(4):
        emitted = glosser.update("HELLO", confidence=0.92)
        assert emitted is None, f"Frame {i+1} should not emit"

    # 5th frame: streak threshold reached -> emits token
    emitted = glosser.update("HELLO", confidence=0.92)
    assert emitted == "HELLO"
    assert glosser.get_tokens() == ["HELLO"]


def test_glosser_refractory_cooldown_suppression():
    """Test 3: Duplicate emissions are blocked for R=8 cooldown frames."""
    glosser = TemporalHysteresisGlosser(thresh=0.85, min_consecutive=5, refractory_period=8)

    # Establish initial streak to emit
    for _ in range(5):
        glosser.update("HELLO", confidence=0.95)

    # Next 8 frames should be suppressed despite high confidence
    for i in range(8):
        emitted = glosser.update("HELLO", confidence=0.95)
        assert emitted is None, f"Frame in cooldown ({i+1}) should be suppressed"


def test_glosser_transition_between_distinct_signs():
    """Test 4: System smoothly transitions to a new distinct sign token."""
    glosser = TemporalHysteresisGlosser(thresh=0.85, min_consecutive=5, refractory_period=3)

    # Emit "HELLO"
    for _ in range(5):
        glosser.update("HELLO", confidence=0.95)

    # Pass cooldown period
    for _ in range(3):
        glosser.update("HELLO", confidence=0.95)

    # Start new sign "STUDENT"
    for i in range(4):
        emitted = glosser.update("STUDENT", confidence=0.90)
        assert emitted is None

    # 5th frame of "STUDENT"
    emitted = glosser.update("STUDENT", confidence=0.90)
    assert emitted == "STUDENT"
    assert glosser.get_tokens() == ["HELLO", "STUDENT"]


def test_glosser_buffer_accumulation_and_reset():
    """Test 5: Token accumulation buffer and state reset."""
    glosser = TemporalHysteresisGlosser(thresh=0.85, min_consecutive=2, refractory_period=1)

    for _ in range(2):
        glosser.update("WATER", confidence=0.90)
    glosser.update("WATER", confidence=0.90)  # cooldown frame

    for _ in range(2):
        glosser.update("PLEASE", confidence=0.90)

    assert glosser.get_tokens() == ["WATER", "PLEASE"]
    glosser.reset()
    assert len(glosser.get_tokens()) == 0
    assert glosser.last_token is None
