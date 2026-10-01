"""Offline smoke test for the video pipeline using deterministic mock detections.

This does not fabricate research results for the actual model. It only verifies
that the video-writing, tracking, confidence, gate, dataframe, and annotation
plumbing work end-to-end without requiring YOLO weights.
"""

from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.pipeline import process_video


class MockDetector:
    def detect(self, frame, frame_number):
        # A deterministic test object, not a research prediction.
        x = 20 + frame_number * 2
        return [
            {
                "class": "ambulance",
                "confidence": 0.90,
                "bbox": [x, 30, x + 80, 110],
                "frame": frame_number,
            }
        ]


def main():
    temp_dir = Path(tempfile.mkdtemp(prefix="safe_uve_smoke_"))
    try:
        input_path = temp_dir / "input.mp4"
        output_path = temp_dir / "output.mp4"
        writer = cv2.VideoWriter(
            str(input_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (320, 240),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the temporary test video.")
        for _ in range(8):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        results, summary = process_video(
            str(input_path),
            str(output_path),
            MockDetector(),
            confidence_threshold=0.80,
            min_confirmation_frames=5,
        )

        assert len(results) == 8
        assert summary["total_frames"] == 8
        assert summary["emergency_vehicles_detected"] == 1
        assert summary["confirmed_emergency_vehicles"] == 1
        assert output_path.exists() and output_path.stat().st_size > 0
        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
