"""Central defaults for Safe-UVE."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emergency_vehicle.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_SAFETY_THRESHOLD = 0.80
DEFAULT_DETECTION_THRESHOLD = 0.25
DEFAULT_MIN_CONFIRMATION_FRAMES = 5
DEFAULT_MAX_MISSED_FRAMES = 5
DEFAULT_TRACK_DISTANCE_RATIO = 0.05

EMERGENCY_CLASS_KEYWORDS = (
    "ambulance",
    "fire truck",
    "fire_truck",
    "firetruck",
    "police",
    "police car",
    "police_car",
    "police vehicle",
    "police_vehicle",
    "emergency vehicle",
    "emergency_vehicle",
)
