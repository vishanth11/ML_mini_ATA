"""Ultralytics YOLO wrapper with clear model/class validation."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from .config import DEFAULT_MODEL_PATH, EMERGENCY_CLASS_KEYWORDS


class EmergencyVehicleDetector:
    def __init__(
        self,
        model_path: str = str(DEFAULT_MODEL_PATH),
        confidence_threshold: float = 0.25,
        class_map: Optional[Dict[int, str]] = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.class_map = class_map or {}

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Emergency vehicle model weights are missing. "
                f"Place the trained YOLO weights at {self.model_path}."
            )

        if self.model_path.suffix.lower() != ".pt":
            raise ValueError("The configured YOLO model should be a .pt weights file.")

        self.model = YOLO(str(self.model_path))

    @property
    def class_names(self) -> List[str]:
        names = self.model.names
        if isinstance(names, dict):
            return [str(value) for _, value in sorted(names.items())]
        return [str(value) for value in names]

    @staticmethod
    def _normalize_class_name(value: str) -> str:
        return " ".join(
            str(value).strip().lower().replace("_", " ").replace("-", " ").split()
        )

    @property
    def matched_emergency_classes(self) -> List[str]:
        matched = []
        for name in self.class_names:
            normalized = self._normalize_class_name(name)
            if normalized.startswith("non emergency"):
                continue
            if any(
                self._normalize_class_name(keyword) in normalized
                for keyword in EMERGENCY_CLASS_KEYWORDS
            ):
                matched.append(name)
        return matched

    def has_emergency_classes(self) -> bool:
        return bool(self.matched_emergency_classes)

    def _class_name(self, class_id: int) -> str:
        if class_id in self.class_map:
            return self.class_map[class_id]
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def detect(self, frame: np.ndarray, frame_number: int) -> List[dict]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: List[dict] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for bbox, confidence, class_id in zip(xyxy, confs, classes):
            detections.append(
                {
                    "class": self._class_name(int(class_id)).strip().lower(),
                    "confidence": float(confidence),
                    "bbox": [float(v) for v in bbox.tolist()],
                    "frame": int(frame_number),
                }
            )
        return detections
