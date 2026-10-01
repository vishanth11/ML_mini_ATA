"""Lightweight centroid tracker used by the Safe-UVE prototype."""

from dataclasses import dataclass, field
from math import hypot, exp
from typing import Dict, List, Optional, Tuple


@dataclass
class Track:
    track_id: int
    class_name: str
    center_x: float
    center_y: float
    confidence: float
    max_distance: float
    consecutive_detections: int = 1
    total_detections: int = 1
    missed_frames: int = 0
    confidences: List[float] = field(default_factory=list)
    motion_scores: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.confidences:
            self.confidences = [self.confidence]

    @property
    def average_confidence(self) -> float:
        return sum(self.confidences) / max(1, len(self.confidences))

    @property
    def motion_consistency(self) -> float:
        if not self.motion_scores:
            return 1.0
        return sum(self.motion_scores) / len(self.motion_scores)

    @property
    def continuity_score(self) -> float:
        return min(1.0, self.consecutive_detections / 10.0)

    @property
    def stability(self) -> float:
        """Simple track consistency combining continuity and motion."""
        return 0.5 * self.continuity_score + 0.5 * self.motion_consistency


class CentroidTracker:
    """Greedy nearest-neighbor tracker; intentionally simple for research use."""

    def __init__(self, max_distance: float = 80.0, max_missed_frames: int = 5):
        self.max_distance = max(1.0, float(max_distance))
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def update(self, detections: List[dict]) -> List[dict]:
        unmatched_tracks = set(self.tracks.keys())
        assigned_detection_indices = set()
        candidates = []

        for det_idx, detection in enumerate(detections):
            cx, cy = self._center(detection["bbox"])
            for track_id, track in self.tracks.items():
                if track.class_name != detection["class"]:
                    continue
                distance = hypot(cx - track.center_x, cy - track.center_y)
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, det_idx))

        for distance, track_id, det_idx in sorted(candidates):
            if track_id not in unmatched_tracks or det_idx in assigned_detection_indices:
                continue

            detection = detections[det_idx]
            cx, cy = self._center(detection["bbox"])
            track = self.tracks[track_id]

            motion_score = exp(-distance / self.max_distance)
            track.motion_scores.append(motion_score)
            track.motion_scores = track.motion_scores[-20:]
            track.center_x = cx
            track.center_y = cy
            track.confidence = float(detection["confidence"])
            track.confidences.append(track.confidence)
            track.confidences = track.confidences[-50:]
            track.consecutive_detections += 1
            track.total_detections += 1
            track.missed_frames = 0

            detection.update(
                {
                    "track_id": track_id,
                    "center_x": cx,
                    "center_y": cy,
                    "consecutive_detections": track.consecutive_detections,
                }
            )

            unmatched_tracks.remove(track_id)
            assigned_detection_indices.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx in assigned_detection_indices:
                continue

            cx, cy = self._center(detection["bbox"])
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                class_name=detection["class"],
                center_x=cx,
                center_y=cy,
                confidence=float(detection["confidence"]),
                max_distance=self.max_distance,
            )
            detection.update(
                {
                    "track_id": track_id,
                    "center_x": cx,
                    "center_y": cy,
                    "consecutive_detections": 1,
                }
            )

        for track_id in list(unmatched_tracks):
            track = self.tracks.get(track_id)
            if track is None:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]
            else:
                # Continuity is broken while the object is missing.
                track.consecutive_detections = 0

        return detections

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)
