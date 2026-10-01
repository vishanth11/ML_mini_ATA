"""Fast offline unittest coverage for Safe-UVE core logic."""

import unittest

from src.confidence import calculate_confidence
from src.safety_gate import SafetyState, evaluate_safety_gate
from src.tracker import CentroidTracker


class SafeUVECoreTests(unittest.TestCase):
    def test_confidence_bounds_and_formula(self):
        result = calculate_confidence(0.9, 0.9, 1.0, 5, 5)
        self.assertGreaterEqual(result.temporal_score, 0.0)
        self.assertLessEqual(result.temporal_score, 1.0)
        self.assertGreaterEqual(result.emergency_confidence, 0.0)
        self.assertLessEqual(result.emergency_confidence, 1.0)
        self.assertAlmostEqual(result.temporal_score, 0.94, places=9)
        self.assertAlmostEqual(result.emergency_confidence, 0.916, places=9)
        self.assertTrue(result.temporal_verification)

    def test_safety_states(self):
        self.assertEqual(
            evaluate_safety_gate(False, 0.0, False, False),
            SafetyState.NORMAL,
        )
        self.assertEqual(
            evaluate_safety_gate(True, 0.70, False, True),
            SafetyState.PREPARE,
        )
        self.assertEqual(
            evaluate_safety_gate(True, 0.90, True, False),
            SafetyState.FALLBACK,
        )
        self.assertEqual(
            evaluate_safety_gate(True, 0.90, True, True),
            SafetyState.VERIFIED_PRIORITY,
        )

    def test_tracker_keeps_id_for_nearby_object(self):
        tracker = CentroidTracker(max_distance=100, max_missed_frames=2)
        ids = []
        for x in (0, 10, 20):
            detection = {
                "class": "ambulance",
                "confidence": 0.8,
                "bbox": [x, 0, x + 20, 20],
            }
            tracked = tracker.update([detection])[0]
            ids.append(tracked["track_id"])
        self.assertEqual(ids, [1, 1, 1])
        self.assertEqual(tracker.get_track(1).consecutive_detections, 3)

    def test_tracker_rejects_different_class_assignment(self):
        tracker = CentroidTracker(max_distance=100)
        first = tracker.update(
            [{"class": "ambulance", "confidence": 0.8, "bbox": [0, 0, 20, 20]}]
        )[0]
        second = tracker.update(
            [{"class": "police vehicle", "confidence": 0.8, "bbox": [0, 0, 20, 20]}]
        )[0]
        self.assertNotEqual(first["track_id"], second["track_id"])

    def test_emergency_class_name_normalization(self):
        from src.pipeline import is_emergency_class
        self.assertTrue(is_emergency_class("emergency_vehicle"))
        self.assertTrue(is_emergency_class("emergency vehicle"))
        self.assertTrue(is_emergency_class("Emergency-Vehicle"))
        self.assertFalse(is_emergency_class("non_emergency_vehicle"))


if __name__ == "__main__":
    unittest.main()
