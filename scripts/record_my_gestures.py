"""Live Webcam Sign Collector (scripts/record_my_gestures.py).

Allows recording your own live gestures from your webcam to fine-tune the model
specifically on your personal hand size, camera position, and signing style.
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extractor import HolisticLandmarkExtractor


def record_sign(sign_name, num_samples=30):
    """Records live 30-frame sequence windows for a specific sign."""
    cap = cv2.VideoCapture(0)
    extractor = HolisticLandmarkExtractor()

    print(f"\nPreparing to record sign: '{sign_name.upper()}' ({num_samples} samples)")
    print("Press SPACE to start recording, or 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign: {sign_name.upper()} | Press SPACE to record", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Sign Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key in [ord('q'), 27]:
            cap.release()
            cv2.destroyAllWindows()
            return None

    collected_windows = []
    print("Recording started! Perform the gesture continuously...")

    for s in range(num_samples):
        window = []
        for f in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            norm_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec_1629d, annotated = extractor.extract(norm_rgb, frame)

            cv2.putText(annotated, f"Recording Sample {s+1}/{num_samples} (Frame {f+1}/30)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Sign Collector", annotated)
            cv2.waitKey(1)
            window.append(vec_1629d)

        if len(window) == 30:
            collected_windows.append(window)

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    output_dir = "data/custom_signs"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{sign_name.upper()}.npy")
    np.save(out_file, np.array(collected_windows, dtype=np.float32))
    print(f"\nSaved {len(collected_windows)} samples to {out_file}!")


if __name__ == "__main__":
    sign = sys.argv[1] if len(sys.argv) > 1 else "HELLO"
    record_sign(sign)
