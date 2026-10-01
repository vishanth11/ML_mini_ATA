"""Rule-based safety gate for the Safe-UVE research prototype."""

from enum import Enum


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    PREPARE = "PREPARE"
    VERIFIED_PRIORITY = "VERIFIED PRIORITY"
    FALLBACK = "FALLBACK"


STATE_PRIORITY = {
    SafetyState.NORMAL: 0,
    SafetyState.PREPARE: 1,
    SafetyState.FALLBACK: 2,
    SafetyState.VERIFIED_PRIORITY: 3,
}


def evaluate_safety_gate(
    emergency_detected: bool,
    emergency_confidence: float,
    temporal_verified: bool,
    track_stable: bool,
    confidence_threshold: float = 0.80,
) -> SafetyState:
    """Evaluate the current frame without controlling a real signal."""
    if not emergency_detected:
        return SafetyState.NORMAL

    if not temporal_verified or emergency_confidence < confidence_threshold:
        return SafetyState.PREPARE

    if temporal_verified and track_stable and emergency_confidence >= confidence_threshold:
        return SafetyState.VERIFIED_PRIORITY

    return SafetyState.FALLBACK


def strongest_state(states: list[SafetyState]) -> SafetyState:
    """Return the strongest recommendation among current emergency tracks."""
    if not states:
        return SafetyState.NORMAL
    return max(states, key=lambda state: STATE_PRIORITY[state])
