"""Video-processing pipeline for Safe-UVE."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import pandas as pd

from .confidence import calculate_confidence
from .config import (
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_TRACK_DISTANCE_RATIO,
    EMERGENCY_CLASS_KEYWORDS,
)
from .safety_gate import SafetyState, evaluate_safety_gate, strongest_state
from .tracker import CentroidTracker


RESULT_COLUMNS = [
    "frame",
    "track_id",
    "class",
    "confidence",
    "temporal_score",
    "emergency_confidence",
    "uncertainty",
    "track_stability",
    "temporal_verification",
    "safety_state",
]


def _normalize_class_name(value: str) -> str:
    # Normalize spaces, underscores, and hyphens so names such as
    # ``emergency_vehicle`` and ``emergency vehicle`` are equivalent.
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def is_emergency_class(class_name: str) -> bool:
    normalized = _normalize_class_name(class_name)
    if normalized.startswith("non emergency"):
        return False
    return any(
        _normalize_class_name(keyword) in normalized
        for keyword in EMERGENCY_CLASS_KEYWORDS
    )


def _draw_text(frame, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    # Black outline + white foreground improves readability over traffic footage.
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_frame(frame, detections: List[dict], gate_state: SafetyState):
    output = frame.copy()
    gate_text = f"SAFETY GATE: {gate_state.value}"
    _draw_text(output, gate_text, (20, 35), 0.72)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        _draw_text(
            output,
            f'{det["class"].upper()} | ID: {det["track_id"]} | Conf: {det["confidence"]:.2f}',
            (max(5, x1), max(20, y1 - 8)),
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return output


def _open_writer(output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        "Could not create the processed video. Your OpenCV installation may not "
        "provide a compatible MP4 codec. Try installing a standard OpenCV build."
    )


def process_video(
    input_path: str,
    output_path: str,
    detector,
    confidence_threshold: float = 0.80,
    min_confirmation_frames: int = 5,
    max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video has invalid dimensions.")

    total_for_progress = declared_frames
    if max_frames is not None and max_frames > 0:
        total_for_progress = min(declared_frames, max_frames) if declared_frames else max_frames

    output_path_obj = Path(output_path)
    writer = _open_writer(output_path_obj, fps, width, height)

    # Scale matching distance with image width so the tracker behaves more consistently across videos.
    max_distance = max(40.0, width * DEFAULT_TRACK_DISTANCE_RATIO)
    tracker = CentroidTracker(
        max_distance=max_distance,
        max_missed_frames=max_missed_frames,
    )

    rows: List[dict] = []
    frame_number = 0
    total_detections = 0
    unique_track_ids = set()
    confirmed_track_ids = set()
    detection_confidences: List[float] = []
    emergency_confidences: List[float] = []
    gate_counts = {state.value: 0 for state in SafetyState}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            if max_frames is not None and max_frames > 0 and frame_number > max_frames:
                break

            raw_detections = detector.detect(frame, frame_number)
            emergency_detections = [
                det for det in raw_detections if is_emergency_class(det["class"])
            ]
            tracked = tracker.update(emergency_detections)

            current_gate_candidates: List[SafetyState] = []
            current_metrics: List[tuple[dict, object]] = []

            for det in tracked:
                track = tracker.get_track(det["track_id"])
                if track is None:
                    continue

                confidence = calculate_confidence(
                    detection_confidence=det["confidence"],
                    average_detection_confidence=track.average_confidence,
                    track_consistency=track.stability,
                    consecutive_detections=track.consecutive_detections,
                    min_confirmation_frames=min_confirmation_frames,
                )

                candidate_gate = evaluate_safety_gate(
                    emergency_detected=True,
                    emergency_confidence=confidence.emergency_confidence,
                    temporal_verified=confidence.temporal_verification,
                    track_stable=confidence.track_consistency >= 0.50,
                    confidence_threshold=confidence_threshold,
                )
                current_gate_candidates.append(candidate_gate)
                current_metrics.append((det, confidence))

            frame_gate = strongest_state(current_gate_candidates)
            gate_counts[frame_gate.value] += 1

            for det, confidence in current_metrics:
                track_id = det["track_id"]
                unique_track_ids.add(track_id)
                total_detections += 1
                detection_confidences.append(confidence.detection_confidence)
                emergency_confidences.append(confidence.emergency_confidence)
                if confidence.temporal_verification:
                    confirmed_track_ids.add(track_id)

                rows.append(
                    {
                        "frame": frame_number,
                        "track_id": track_id,
                        "class": det["class"],
                        "confidence": round(confidence.detection_confidence, 5),
                        "temporal_score": round(confidence.temporal_score, 5),
                        "emergency_confidence": round(confidence.emergency_confidence, 5),
                        "uncertainty": round(confidence.uncertainty, 5),
                        "track_stability": round(confidence.track_consistency, 5),
                        "temporal_verification": "PASS" if confidence.temporal_verification else "PENDING",
                        "safety_state": frame_gate.value,
                    }
                )

            writer.write(annotate_frame(frame, tracked, frame_gate))

            if progress_callback:
                progress_callback(frame_number, total_for_progress)
    finally:
        cap.release()
        writer.release()

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary = {
        "total_frames": frame_number,
        "total_detections": total_detections,
        "emergency_vehicles_detected": len(unique_track_ids),
        "confirmed_emergency_vehicles": len(confirmed_track_ids),
        "average_confidence": sum(detection_confidences) / len(detection_confidences) if detection_confidences else 0.0,
        "maximum_confidence": max(detection_confidences) if detection_confidences else 0.0,
        "average_emergency_confidence": sum(emergency_confidences) / len(emergency_confidences) if emergency_confidences else 0.0,
        "maximum_emergency_confidence": max(emergency_confidences) if emergency_confidences else 0.0,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "gate_counts": gate_counts,
    }
    return results_df, summary
