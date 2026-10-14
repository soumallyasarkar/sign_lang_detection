"""Application Launcher & CLI Entrypoint (main.py).

Provides command-line arguments to launch either:
1. Live Desktop GUI Application (Default)
2. Headless Streaming CLI Evaluator
3. Benchmark Suite validating Latency & Frame Throughput
"""

import sys
import os
import argparse
import time
import yaml
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import ContinuousInterpretationPipeline


def load_config(config_path="configs/config.yaml"):
    """Loads system configuration from YAML."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_benchmark(pipeline, num_frames=60):
    """Executes a throughput and latency benchmark across the full 8-module pipeline."""
    print("=" * 65)
    print("  CSE411 Sign Language Interpretation Pipeline Benchmark")
    print("=" * 65)
    print(f"Executing {num_frames} continuous streaming frames...")

    latencies = []
    start_all = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()
        annotated_frame, emitted_sign, conf, sentence = pipeline.step()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    total_time = time.perf_counter() - start_all
    avg_latency = np.mean(latencies)
    fps = num_frames / total_time

    print(f"\nBenchmark Results:")
    print(f"  - Total Frames Processed: {num_frames}")
    print(f"  - Throughput:             {fps:.2f} FPS (Target: ~27.6 FPS)")
    print(f"  - Mean Vision Latency:    {avg_latency:.2f} ms/frame (Target: <50 ms)")
    print(f"  - Total Duration:         {total_time:.2f} seconds")
    print(f"  - Last Recognized Sign:   {pipeline.latest_sign or 'None'}")
    print(f"  - Last English Sentence:  {pipeline.latest_sentence or 'Awaiting token accumulation'}")
    print("=" * 65)


def run_cli_stream(pipeline):
    """Runs headless interpretation loop in the console."""
    print("Starting Headless ASL Interpreter Stream (Press Ctrl+C to stop)...")
    pipeline.start()
    try:
        while True:
            annotated_frame, emitted_sign, conf, sentence = pipeline.step()
            if emitted_sign:
                print(f"[SIGN DETECTED] {emitted_sign} (Conf: {conf * 100:.1f}%)")
            if sentence and sentence != getattr(run_cli_stream, '_last_sentence', ''):
                print(f"  -> English: \"{sentence}\"")
                run_cli_stream._last_sentence = sentence
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping streaming pipeline...")
    finally:
        pipeline.stop()


def run_gui(config):
    """Launches the Live Interactive Desktop GUI (Tkinter or OpenCV HUD fallback)."""
    from src.gui_app import SignInterpreterGUI, _TKINTER_AVAILABLE

    if _TKINTER_AVAILABLE:
        try:
            import tkinter as tk
            root = tk.Tk()
            app = SignInterpreterGUI(root, config)
            root.protocol("WM_DELETE_WINDOW", app.close)
            root.mainloop()
            return
        except Exception as e:
            print(f"Tkinter unavailable ({e}), launching native OpenCV HUD...")

    print("\nLaunching Live Sign Language Interpretation Desktop GUI...")
    print("Window Title: 'Real-Time ASL Interpretation System — Team 14'")
    print("Note: If the window appears behind your code editor, check your taskbar or press Alt+Tab to view it.\n")

    # Seamless native OpenCV HighGUI HUD
    app = SignInterpreterGUI(root=None, config=config)
    app.run_opencv_window()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Interpretation System")
    parser.add_argument("--gui", action="store_true", default=True, help="Launch Desktop GUI HUD (default)")
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI console mode")
    parser.add_argument("--benchmark", action="store_true", help="Run latency and throughput benchmark")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.camera is not None:
        if 'system' not in config:
            config['system'] = {}
        config['system']['camera_idx'] = args.camera

    if args.benchmark:
        pipeline = ContinuousInterpretationPipeline(config)
        pipeline.start()
        try:
            run_benchmark(pipeline)
        finally:
            pipeline.stop()
    elif args.cli:
        pipeline = ContinuousInterpretationPipeline(config)
        run_cli_stream(pipeline)
    else:
        # Launch GUI
        run_gui(config)


if __name__ == "__main__":
    main()
