"""Confidence and temporal verification utilities for Safe-UVE."""

from dataclasses import dataclass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ConfidenceResult:
    detection_confidence: float
    track_consistency: float
    temporal_score: float
    emergency_confidence: float
    uncertainty: float
    temporal_verification: bool


def calculate_temporal_score(
    average_detection_confidence: float,
    track_consistency: float,
) -> float:
    """Temporal score from detection quality and track consistency."""
    return clamp01(
        0.6 * clamp01(average_detection_confidence)
        + 0.4 * clamp01(track_consistency)
    )


def calculate_emergency_confidence(
    detection_confidence: float,
    temporal_score: float,
) -> float:
    """Estimated Emergency Confidence; not statistically calibrated."""
    return clamp01(
        0.6 * clamp01(detection_confidence)
        + 0.4 * clamp01(temporal_score)
    )


def calculate_uncertainty(emergency_confidence: float) -> float:
    return clamp01(1.0 - clamp01(emergency_confidence))


def is_temporally_verified(
    consecutive_detections: int,
    min_confirmation_frames: int,
) -> bool:
    return consecutive_detections >= max(1, int(min_confirmation_frames))


def calculate_confidence(
    detection_confidence: float,
    average_detection_confidence: float,
    track_consistency: float,
    consecutive_detections: int,
    min_confirmation_frames: int,
) -> ConfidenceResult:
    detection_confidence = clamp01(detection_confidence)
    track_consistency = clamp01(track_consistency)
    temporal_score = calculate_temporal_score(
        average_detection_confidence,
        track_consistency,
    )
    emergency_confidence = calculate_emergency_confidence(
        detection_confidence,
        temporal_score,
    )
    uncertainty = calculate_uncertainty(emergency_confidence)
    verified = is_temporally_verified(
        consecutive_detections,
        min_confirmation_frames,
    )
    return ConfidenceResult(
        detection_confidence=detection_confidence,
        track_consistency=track_consistency,
        temporal_score=temporal_score,
        emergency_confidence=emergency_confidence,
        uncertainty=uncertainty,
        temporal_verification=verified,
    )
