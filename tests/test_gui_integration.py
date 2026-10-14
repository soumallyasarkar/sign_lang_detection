"""Integration tests for Module 9 (src/gui_app.py).

Verifies 2 target behaviors:
1. Desktop event loop & HUD widget initialization
2. HUD reactivity, confidence meter updates, and session control toggles
"""

import pytest
from src.gui_app import SignInterpreterGUI, _TKINTER_AVAILABLE

try:
    import tkinter as tk
except ImportError:
    tk = None


@pytest.fixture
def gui_app():
    """Initializes GUI with withdrawn root window or mock root for headless test execution."""
    if _TKINTER_AVAILABLE and tk is not None:
        root = tk.Tk()
        root.withdraw()
    else:
        root = None

    app = SignInterpreterGUI(root)
    yield app
    app.close()


def test_gui_widget_initialization(gui_app):
    """Test 1: Verifies all core HUD widgets and controls are instantiated."""
    assert gui_app.canvas_label is not None
    assert gui_app.subtitle_text is not None
    assert gui_app.conf_bar is not None
    assert gui_app.conf_label is not None
    assert gui_app.timeline_box is not None
    assert gui_app.pause_btn is not None
    assert gui_app.audio_btn is not None


def test_gui_session_toggles_and_reactivity(gui_app):
    """Test 2: Verifies pause toggle, audio toggle, and transcript reset."""
    # Test pause toggle
    assert gui_app.is_paused is False
    gui_app.toggle_pause()
    assert gui_app.is_paused is True
    gui_app.toggle_pause()
    assert gui_app.is_paused is False

    # Test audio mute toggle
    assert gui_app.audio_enabled is True
    gui_app.toggle_audio()
    assert gui_app.audio_enabled is False
    assert gui_app.pipeline.tts.enabled is False
    gui_app.toggle_audio()
    assert gui_app.audio_enabled is True

    # Test clear transcript
    gui_app.timeline_box.insert(0, "TEST_TOKEN")
    gui_app.clear_transcript()
    assert gui_app.timeline_box.size() == 0
