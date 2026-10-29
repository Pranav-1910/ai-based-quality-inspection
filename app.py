
import av
import cv2
import time
import threading
import streamlit as st
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from database import get_inspections, save_inspection


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
IMAGE_DIR = BASE_DIR / "inspection_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

INSPECTION_INTERVAL = 5

st.set_page_config(
    page_title="PCB Quality Inspection",
    page_icon="🔍",
    layout="wide"
)

st.title("PCB Quality Inspection")
st.subheader("AI-Based PCB Defect Detection")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()
inference_lock = threading.Lock()

st.sidebar.header("Inspection Settings")

mode = st.sidebar.radio(
    "Choose Inspection Mode",
    ["Upload Image", "Live Detection"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.25,
    step=0.05
)


def save_image(image, prefix="inspection"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = IMAGE_DIR / f"{prefix}_{timestamp}.jpg"

    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError("Could not save inspection image.")

    return str(image_path)


def inspect_image(image):
    image_rgb = np.array(image.convert("RGB"))

    with inference_lock:
        results = model.predict(
            source=image_rgb,
            imgsz=640,
            conf=confidence,
            verbose=False,
            device="cpu"
        )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    defect_types = []
    confidence_scores = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            defect_types.append(model.names[class_id])
            confidence_scores.append(float(box.conf.item()))

    defect_count = len(defect_types)

    status = (
        "DEFECTS DETECTED"
        if defect_count > 0
        else "NO DEFECTS DETECTED"
    )

    return (
        annotated_rgb,
        defect_count,
        defect_types,
        confidence_scores,
        status
    )


if mode == "Upload Image":

    st.markdown("### Upload PCB Image")

    uploaded_file = st.file_uploader(
        "Choose a PCB image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            original_image,
            caption="Original PCB Image",
            use_container_width=True
        )

        if st.button("Inspect PCB", type="primary"):
            try:
                with st.spinner("Inspecting PCB..."):
                    (
                        annotated_rgb,
                        defect_count,
                        defect_types,
                        confidence_scores,
                        status
                    ) = inspect_image(original_image)

                    annotated_bgr = cv2.cvtColor(
                        annotated_rgb,
                        cv2.COLOR_RGB2BGR
                    )

                    image_path = save_image(
                        annotated_bgr,
                        prefix="upload"
                    )

                    save_inspection(
                        inspection_mode="Upload Image",
                        image_path=image_path,
                        defect_count=defect_count,
                        defect_types=defect_types,
                        confidence_scores=confidence_scores,
                        status=status
                    )

                st.success("Inspection completed and saved to history.")

                col1, col2, col3 = st.columns(3)

                col1.metric("Detected Defects", defect_count)
                col2.metric("Defect Types", len(set(defect_types)))
                col3.metric("Inspection Status", status)

                st.image(
                    annotated_rgb,
                    caption="YOLO Detection Result",
                    use_container_width=True
                )

                if defect_types:
                    st.dataframe(
                        pd.DataFrame({
                            "Defect Type": defect_types,
                            "Confidence Score": [
                                f"{score:.1%}"
                                for score in confidence_scores
                            ]
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info(
                        "No defects detected above the selected "
                        "confidence threshold."
                    )

            except Exception as error:
                st.error(f"Inspection failed: {error}")


elif mode == "Live Detection":

    st.markdown("### Automatic Live PCB Inspection")

    st.info(
        "Live inspection runs automatically every 5 seconds. "
        "Each scheduled inspection saves a record whether defects "
        "are detected or not."
    )

    st.write("Inspection interval: **5 seconds**")

    class PCBVideoProcessor:

        def __init__(self):
            self.lock = threading.Lock()
            self.last_count = 0
            self.last_names = []
            self.last_scores = []
            self.last_status = "Waiting for camera..."
            self.last_saved_time = None
            self.last_error = ""
            self.last_frame = None
            self.last_inspection_time = 0.0

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")

            height, width = image.shape[:2]
            max_width = 640

            if width > max_width:
                new_height = int(height * max_width / width)
                image = cv2.resize(
                    image,
                    (max_width, new_height)
                )

            now = time.monotonic()

            if now - self.last_inspection_time >= INSPECTION_INTERVAL:
                self.last_inspection_time = now

                try:
                    with inference_lock:
                        results = model.predict(
                            source=image,
                            imgsz=416,
                            conf=confidence,
                            verbose=False,
                            device="cpu"
                        )

                    result = results[0]
                    annotated = result.plot()

                    defect_types = []
                    confidence_scores = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            defect_types.append(model.names[class_id])
                            confidence_scores.append(float(box.conf.item()))

                    defect_count = len(defect_types)

                    status = (
                        "DEFECTS DETECTED"
                        if defect_count > 0
                        else "NO DEFECTS DETECTED"
                    )

                    image_path = save_image(
                        annotated,
                        prefix="live"
                    )

                    save_inspection(
                        inspection_mode="Live Detection",
                        image_path=image_path,
                        defect_count=defect_count,
                        defect_types=defect_types,
                        confidence_scores=confidence_scores,
                        status=status
                    )

                    with self.lock:
                        self.last_count = defect_count
                        self.last_names = list(defect_types)
                        self.last_scores = list(confidence_scores)
                        self.last_status = status
                        self.last_frame = annotated.copy()
                        self.last_saved_time = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        self.last_error = ""

                except Exception as error:
                    with self.lock:
                        self.last_error = str(error)

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

    ctx = webrtc_streamer(
        key="pcb-interval-inspection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PCBVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if ctx.state.playing and ctx.video_processor:

        @st.fragment(run_every=2)
        def show_live_status():
            processor = ctx.video_processor

            if processor is None:
                return

            with processor.lock:
                count = processor.last_count
                names = list(processor.last_names)
                scores = list(processor.last_scores)
                status = processor.last_status
                saved_time = processor.last_saved_time
                error = processor.last_error
                annotated = (
                    processor.last_frame.copy()
                    if processor.last_frame is not None
                    else None
                )

            col1, col2, col3 = st.columns(3)

            col1.metric("Detected Defects", count)
            col2.metric("Defect Types", len(set(names)))
            col3.metric("Inspection Status", status)

            if saved_time:
                st.caption(f"Last inspection saved: {saved_time}")

            if error:
                st.error(f"Automatic inspection error: {error}")

            if annotated is not None:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Last Scheduled Inspection",
                    use_container_width=True
                )

            if names:
                st.dataframe(
                    pd.DataFrame({
                        "Defect Type": names,
                        "Confidence Score": [
                            f"{score:.1%}" for score in scores
                        ]
                    }),
                    use_container_width=True,
                    hide_index=True
                )

        show_live_status()


st.markdown("---")
st.markdown("## Inspection Dashboard")


@st.fragment(run_every=2)
def show_inspection_history():
    try:
        records = get_inspections()

        columns = [
            "ID",
            "Timestamp",
            "Inspection Mode",
            "Image Path",
            "Defect Count",
            "Defect Types",
            "Confidence Scores",
            "Status"
        ]

        if records:
            history_df = pd.DataFrame(records, columns=columns)

            total_boards = len(history_df)
            defective_boards = int(
                (history_df["Defect Count"] > 0).sum()
            )
            non_defective_boards = int(
                (history_df["Defect Count"] == 0).sum()
            )
        else:
            history_df = pd.DataFrame(columns=columns)
            total_boards = 0
            defective_boards = 0
            non_defective_boards = 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Boards Inspected", total_boards)
        col2.metric("Defective Boards", defective_boards)
        col3.metric("Non-Defective Boards", non_defective_boards)

        st.markdown("### Inspection History")

        display_df = history_df.drop(
            columns=["Image Path"],
            errors="ignore"
        )

        if not display_df.empty:
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No inspection records yet.")

    except Exception as error:
        st.error(f"Could not load inspection history: {error}")


show_inspection_history()
