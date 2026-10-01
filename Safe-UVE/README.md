# Safe-UVE — Enhanced Research Prototype

**Uncertainty-Aware Vision-Based Emergency Vehicle Detection and Safety-Gated Signal Pre-emption**

This version improves the original basic prototype while keeping the architecture small and research-friendly.

## What was fixed and improved

1. **Consistent frame-level Safety Gate**
   - The original pipeline could assign different gate states while iterating through multiple tracks in the same frame.
   - The enhanced version first evaluates all current emergency tracks, selects one frame-level state, then writes that same state to every result from that frame and to the annotation.

2. **More meaningful track stability**
   - Stability is no longer only `consecutive_frames / 10`.
   - It combines continuity and recent motion consistency, while still remaining a lightweight centroid tracker.

3. **Robust model validation**
   - The app detects missing model weights with a clear message.
   - It reports the loaded model's class names and warns when no emergency-vehicle class is available.

4. **Cleaner temporary-file handling**
   - Uploaded videos are processed from temporary files and cleaned up afterward.
   - The original upload is never overwritten.

5. **More reliable video output**
   - The pipeline tries common MP4 codecs and gives a clear error if OpenCV cannot create an output writer.

6. **Debug-friendly controls**
   - Optional maximum-frame limit for fast demos and troubleshooting.

7. **Offline core tests**
   - Unit tests cover confidence calculations, Safety Gate states, and tracking IDs.
   - A separate smoke test exercises the complete video pipeline using deterministic test detections. It does not claim model accuracy or generate research results.

## Architecture

```text
                   ┌───────────────────────┐
Traffic Video ───► │ Ultralytics YOLO      │
                   │ Object Detection      │
                   └───────────┬───────────┘
                               │
                               ▼
                   Emergency-class filter
                               │
                               ▼
                   ┌───────────────────────┐
                   │ Centroid Tracker      │
                   │ Track IDs + movement  │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │ Temporal Verification│
                   │ + Track Consistency   │
                   └───────────┬───────────┘
                               │
                               ▼
                   Estimated Emergency
                        Confidence
                               │
                               ▼
                         Uncertainty
                               │
                               ▼
                   ┌───────────────────────┐
                   │ Rule-Based Safety    │
                   │ Gate                  │
                   └───────────┬───────────┘
                               │
                               ▼
            NORMAL / PREPARE / VERIFIED PRIORITY / FALLBACK
```

## Folder structure

```text
Safe-UVE/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── emergency_vehicle.pt   ← place your trained weights here
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   ├── tracker.py
│   ├── confidence.py
│   ├── safety_gate.py
│   └── pipeline.py
│
├── data/
│   └── sample/
│       └── README.md
│
├── outputs/
│   └── README.md
│
└── tests/
    ├── test_core.py
    └── smoke_pipeline.py
```

## Installation

Recommended: use a Python virtual environment.

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows CMD

```bat
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit application

From the `Safe-UVE` folder:

```bash
streamlit run app.py
```

## Model placement

The application expects:

```text
models/emergency_vehicle.pt
```

Your trained YOLO model should expose emergency-vehicle classes such as:

```text
ambulance
fire truck
police vehicle
```

A generic pretrained YOLO model can be useful for testing that model loading works, but it is **not** an emergency-vehicle detector unless the required emergency classes are actually present.

The app explicitly checks the loaded class names and warns when they are missing.

## Configuration

The Streamlit sidebar controls:

- Safety-gate confidence threshold — default `0.80`
- Minimum confirmation frames — default `5`
- YOLO detection confidence threshold — default `0.25`
- YOLO model path — default `models/emergency_vehicle.pt`
- Maximum processed frames — default `0`, meaning all frames

## Detection output

For every relevant detection the detector returns:

```python
{
    "class": "ambulance",
    "confidence": 0.87,
    "bbox": [100, 120, 250, 300],
    "frame": 50
}
```

## Tracking

A lightweight nearest-neighbor centroid tracker assigns a track ID to each emergency-vehicle detection.

Track state includes:

- `track_id`
- `class`
- `center_x`
- `center_y`
- latest detection confidence
- consecutive detections
- average confidence
- motion consistency
- overall track stability

This is intentionally not a production multi-object tracking implementation.

## Temporal verification

An emergency vehicle is not verified from one frame alone.

Default confirmation requirement:

```text
5 consecutive detections
```

Temporal score:

```text
temporal_score =
    0.6 * average_detection_confidence
    + 0.4 * track_consistency
```

The score is clipped to `[0, 1]`.

## Estimated Emergency Confidence

The prototype then calculates:

```text
emergency_confidence =
    0.6 * detection_confidence
    + 0.4 * temporal_score
```

and:

```text
uncertainty = 1 - emergency_confidence
```

The UI labels this **Estimated Emergency Confidence**.

It is **not calibrated confidence**. Statistical calibration would require suitable calibration data and explicit calibration experiments.

## Safety Gate

The detector does not directly control a traffic signal.

The current frame is classified using these rules:

```text
IF no emergency vehicle is detected:
    NORMAL

IF emergency vehicle is detected AND
   confidence < threshold OR temporal verification is incomplete:
    PREPARE

IF emergency_confidence >= threshold AND
   temporal verification passes AND
   track is stable:
    VERIFIED PRIORITY

OTHERWISE:
    FALLBACK
```

`VERIFIED PRIORITY` means that the prototype recommends emergency priority based on its configured evidence thresholds. It is not a real signal command.

## Output files

Processed videos are written to:

```text
outputs/processed_<video-name>.mp4
```

The app provides downloads for:

- annotated processed video
- results CSV

The original uploaded video is not overwritten.

## Results table

The table/CSV contains:

```text
frame
track_id
class
confidence
temporal_score
emergency_confidence
uncertainty
track_stability
temporal_verification
safety_state
```

## Testing without YOLO weights

The core logic can be tested without downloading a YOLO model.

Install the listed packages, then run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

For the end-to-end video plumbing smoke test:

```bash
python tests/smoke_pipeline.py
```

The smoke test uses deterministic mock detections solely to verify that the tracker, confidence module, Safety Gate, annotation, video writer, and results dataframe work together. It does not represent actual model predictions and must not be used as a performance result.

## Research limitations

This remains a research prototype.

- Confidence is estimated and not statistically calibrated.
- The tracker is intentionally lightweight.
- The system does not control a real traffic signal.
- `VERIFIED PRIORITY` is a recommendation state.
- Real deployment requires validated traffic-signal safety constraints and fail-safe engineering.
- No real-world traffic improvement is claimed.
- No accuracy, precision, recall, F1, mAP, or calibration numbers are fabricated.
- Any performance claims should come from controlled experiments on an appropriate test set.

## Troubleshooting

### `Emergency vehicle model weights are missing`

Place the file at:

```text
Safe-UVE/models/emergency_vehicle.pt
```

### The model loads but the app reports no emergency classes

Check the model's training label names. A standard COCO vehicle model generally does not contain classes named ambulance, fire truck, or police vehicle. Use dedicated emergency-vehicle YOLO weights or retrain/fine-tune a model with those classes.

### OpenCV cannot create the processed MP4

The pipeline tries `mp4v` and then `avc1`. If both fail, install a standard OpenCV Python package in a clean virtual environment and retry. On some systems the available codec support depends on the local OpenCV/FFmpeg build.

### Streamlit command is not found

Activate the virtual environment and reinstall:

```bash
pip install -r requirements.txt
```

Then:

```bash
python -m streamlit run app.py
```

## Final demo flow

```text
User uploads traffic video
        ↓
YOLO detects objects
        ↓
Emergency classes selected
        ↓
Centroid tracker assigns IDs
        ↓
Multiple frames evaluated
        ↓
Temporal score calculated
        ↓
Estimated Emergency Confidence calculated
        ↓
Uncertainty calculated
        ↓
Safety Gate evaluates evidence
        ↓
NORMAL / PREPARE / VERIFIED PRIORITY / FALLBACK
        ↓
Annotated video + metrics + charts + CSV
```
## 17. Troubleshooting zero emergency detections

If the standalone Ultralytics test detects `emergency_vehicle` but the Streamlit pipeline reports zero emergency detections, confirm that the application is using the same `models/emergency_vehicle.pt` file and try a YOLO detection threshold of `0.10`. The current implementation normalizes spaces, underscores, and hyphens when matching model class names, so `emergency_vehicle` is treated the same as `emergency vehicle`.


## 17. Class-name normalization

Safe-UVE normalizes spaces, underscores, and hyphens when comparing YOLO class names. This allows the supplied binary model class `emergency_vehicle` to be recognized as an emergency class. Negative classes such as `non_emergency_vehicle` are explicitly excluded.

## 18. Model diagnostic

After activating the virtual environment, you can verify the weights with:

```bash
python tools/check_model.py
```
