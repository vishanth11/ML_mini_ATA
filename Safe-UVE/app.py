"""Streamlit front end for the Safe-UVE research prototype."""

from pathlib import Path
import tempfile

import streamlit as st

from src.config import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_MIN_CONFIRMATION_FRAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAFETY_THRESHOLD,
)
from src.detector import EmergencyVehicleDetector
from src.pipeline import process_video
from src.safety_gate import SafetyState


st.set_page_config(page_title="Safe-UVE", page_icon="🚑", layout="wide")

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.title("Safe-UVE")
st.caption(
    "Uncertainty-Aware Vision-Based Emergency Vehicle Detection and "
    "Safety-Gated Signal Pre-emption"
)
st.info(
    "Research prototype only. The Safety Gate produces a recommendation; "
    "it does not directly control a real traffic signal."
)

with st.sidebar:
    st.header("Configuration")
    uploaded_video = st.file_uploader(
        "Upload traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Shorter videos are recommended for interactive demonstrations.",
    )
    safety_threshold = st.slider(
        "Safety-gate confidence threshold",
        0.50,
        0.99,
        DEFAULT_SAFETY_THRESHOLD,
        0.01,
    )
    min_confirmation_frames = st.number_input(
        "Minimum confirmation frames",
        min_value=1,
        max_value=100,
        value=DEFAULT_MIN_CONFIRMATION_FRAMES,
        step=1,
    )
    detection_threshold = st.slider(
        "YOLO detection confidence threshold",
        0.05,
        0.95,
        DEFAULT_DETECTION_THRESHOLD,
        0.05,
    )
    model_path_text = st.text_input(
        "YOLO model path",
        value="models/emergency_vehicle.pt",
    )
    max_frames = st.number_input(
        "Maximum frames (0 = all)",
        min_value=0,
        max_value=100000,
        value=0,
        step=100,
        help="Useful for quick demonstrations and debugging.",
    )

st.markdown("### Research status")
st.caption(
    "Estimated Emergency Confidence is an uncalibrated evidence score. "
    "Uncertainty is defined as 1 − Estimated Emergency Confidence. "
    "No model accuracy or traffic-improvement result is claimed."
)

if uploaded_video is None:
    st.warning("Upload a traffic video from the sidebar to begin.")
    st.stop()

# Treat a relative path as project-relative. Absolute paths are also accepted.
model_path = Path(model_path_text).expanduser()
if not model_path.is_absolute():
    model_path = Path(__file__).resolve().parent / model_path

if not model_path.exists():
    st.error(
        "Emergency vehicle model weights are missing. "
        "Place the trained YOLO weights at models/emergency_vehicle.pt."
    )
    st.code("Safe-UVE/models/emergency_vehicle.pt")
    st.warning(
        "A generic YOLO model may load successfully but is not equivalent to "
        "an emergency-vehicle detector. It must contain emergency-vehicle classes "
        "such as ambulance, fire truck, or police vehicle for this demo."
    )
    st.stop()

with tempfile.NamedTemporaryFile(
    prefix="safe_uve_input_",
    suffix=Path(uploaded_video.name).suffix or ".mp4",
    delete=False,
) as temp_file:
    temp_file.write(uploaded_video.getbuffer())
    input_path = Path(temp_file.name)

safe_stem = Path(uploaded_video.name).stem.replace(" ", "_")
output_path = OUTPUT_DIR / f"processed_{safe_stem}.mp4"

try:
    detector = EmergencyVehicleDetector(
        model_path=str(model_path),
        confidence_threshold=detection_threshold,
    )

    st.markdown("### Model information")
    model_col1, model_col2 = st.columns(2)
    model_col1.write(f"**Loaded model:** `{model_path.name}`")
    model_col2.write(f"**Classes:** {len(detector.class_names)}")
    if detector.matched_emergency_classes:
        st.success(
            "Emergency classes found: "
            + ", ".join(detector.matched_emergency_classes)
        )
    else:
        st.error(
            "This model does not expose emergency-vehicle class names. "
            "The pipeline will therefore produce no emergency detections. "
            "Use dedicated emergency-vehicle weights."
        )

    progress_bar = st.progress(0.0)
    status = st.empty()

    def update_progress(done: int, total: int) -> None:
        if total > 0:
            progress_bar.progress(min(1.0, done / total))
        status.write(f"Processing frame {done}" + (f" / {total}" if total else ""))

    with st.spinner("Running YOLO detection, tracking, temporal verification, and Safety Gate..."):
        results_df, summary = process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            detector=detector,
            confidence_threshold=safety_threshold,
            min_confirmation_frames=int(min_confirmation_frames),
            max_frames=int(max_frames) if max_frames else None,
            progress_callback=update_progress,
        )

    progress_bar.progress(1.0)
    status.success("Video processing complete.")
except Exception as exc:
    st.error("The video-processing pipeline stopped with an error.")
    st.exception(exc)
    st.stop()
finally:
    try:
        input_path.unlink(missing_ok=True)
    except Exception:
        pass

st.markdown("### Video")
video_col1, video_col2 = st.columns(2)
with video_col1:
    st.write("**Uploaded video**")
    st.video(uploaded_video.getvalue())
with video_col2:
    st.write("**Annotated output**")
    if output_path.exists():
        output_bytes = output_path.read_bytes()
        st.video(output_bytes)
        st.download_button(
            "Download processed video",
            data=output_bytes,
            file_name=output_path.name,
            mime="video/mp4",
        )

st.markdown("### Current emergency status")
if results_df.empty:
    current = {
        "safety_state": SafetyState.NORMAL.value,
        "class": "None",
        "track_id": "-",
        "confidence": 0.0,
        "temporal_score": 0.0,
        "emergency_confidence": 0.0,
        "uncertainty": 1.0,
        "track_stability": 0.0,
        "temporal_verification": "PENDING",
    }
else:
    last_frame = int(results_df["frame"].max())
    current_row = (
        results_df[results_df["frame"] == last_frame]
        .sort_values("emergency_confidence", ascending=False)
        .iloc[0]
    )
    current = current_row.to_dict()

cols = st.columns(5)
cols[0].metric("Safety Gate", current["safety_state"])
cols[1].metric("Vehicle Class", current["class"])
cols[2].metric("Track ID", current["track_id"])
cols[3].metric("Detection Confidence", f"{float(current['confidence']):.2f}")
cols[4].metric("Temporal Score", f"{float(current['temporal_score']):.2f}")

cols2 = st.columns(4)
cols2[0].metric("Estimated Emergency Confidence", f"{float(current['emergency_confidence']):.2f}")
cols2[1].metric("Uncertainty", f"{float(current['uncertainty']):.2f}")
cols2[2].metric("Track Stability", f"{float(current['track_stability']):.2f}")
cols2[3].metric("Temporal Verification", current["temporal_verification"])

st.markdown("### Measured run metrics")
metric_cols = st.columns(6)
metric_cols[0].metric("Total Frames", summary["total_frames"])
metric_cols[1].metric("Emergency Detections", summary["total_detections"])
metric_cols[2].metric("Unique Emergency Tracks", summary["emergency_vehicles_detected"])
metric_cols[3].metric("Confirmed Tracks", summary["confirmed_emergency_vehicles"])
metric_cols[4].metric("Avg Emergency Conf.", f"{summary['average_emergency_confidence']:.2f}")
metric_cols[5].metric("Max Emergency Conf.", f"{summary['maximum_emergency_confidence']:.2f}")


with st.expander("Detector diagnostics", expanded=True):
    st.write(f"**Model classes:** {', '.join(detector.class_names)}")
    st.write(f"**Emergency classes matched by Safe-UVE:** {', '.join(detector.matched_emergency_classes) or 'None'}")
    st.write(f"**YOLO detection threshold:** {detection_threshold:.2f}")
    st.write(f"**Emergency detections retained by pipeline:** {summary['total_detections']}")
    if summary["total_detections"] == 0 and detector.matched_emergency_classes:
        st.error(
            "The YOLO model exposes an emergency class, but no emergency detections "
            "survived the configured threshold on this video. Try 0.05–0.15 for diagnosis."
        )

st.markdown("### Confidence over time")
if results_df.empty:
    st.write("No emergency-vehicle detections were recorded.")
else:
    chart = (
        results_df.groupby("frame", as_index=True)["emergency_confidence"]
        .max()
        .rename("Estimated Emergency Confidence")
    )
    st.line_chart(chart)

st.markdown("### Emergency vehicles over time")
if results_df.empty:
    st.write("No emergency-vehicle detections were recorded.")
else:
    chart = (
        results_df.groupby("frame")["track_id"]
        .nunique()
        .rename("Emergency Vehicles")
    )
    st.line_chart(chart)

st.markdown("### Results")
st.dataframe(results_df, use_container_width=True)
st.download_button(
    "Download results CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="safe_uve_results.csv",
    mime="text/csv",
)

st.markdown("### Safety Gate logic")
st.write(
    "**NORMAL:** no emergency vehicle is detected.  \\n"
    "**PREPARE:** an emergency vehicle is detected, but confidence is below "
    "the configured threshold or temporal verification is incomplete.  \\n"
    "**VERIFIED PRIORITY:** confidence meets the threshold, temporal verification "
    "passes, and the track is stable.  \\n"
    "**FALLBACK:** evidence reaches the confidence/temporal requirements but "
    "the track is not stable enough for the verified-priority recommendation."
)

st.markdown("### Research limitations")
st.warning(
    "This is a research prototype. Estimated Emergency Confidence is not "
    "statistically calibrated confidence. The prototype does not control a real "
    "traffic signal, does not establish real-world traffic improvement, and does "
    "not provide fabricated accuracy, precision, recall, F1, mAP, or calibration results."
)
