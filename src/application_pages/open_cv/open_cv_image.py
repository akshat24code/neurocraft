import cv2
import numpy as np
import streamlit as st

from .open_cv_shared import apply_detection
from .open_cv_core import SAMPLE_DIR


def run_image_use_case(detection_type, face_cascades, eye_cascade, smile_cascade, car_cascade=None):
    image_source = st.radio(
        "Select Image Source", ("Upload Image", "Sample Image"), horizontal=True
    )

    image: np.ndarray | None = None

    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Could not decode the uploaded image.")
    else:
        sample_map = {
            "Face Detection": str(SAMPLE_DIR / "group_pic.jpg"),
            "Eye + Smile Detection": str(SAMPLE_DIR / "henry.jpg"),
            "Real Time Face Count": str(SAMPLE_DIR / "group_pic.jpg"),
            "Stop Sign Detection": str(SAMPLE_DIR / "stop_sign.png"),
        }
        sample_path = sample_map.get(detection_type, "")

        if st.button("Load Sample Image", key="opencv_load_sample_image"):
            if not sample_path:
                 st.warning(f"No sample image defined for {detection_type}")
            else:
                image = cv2.imread(sample_path)
                if image is None:
                    st.error(f"Could not load sample image from: {sample_path}")

    if image is None:
        return

    # Use the unified dispatcher
    result = apply_detection(image, detection_type, face_cascades, eye_cascade, smile_cascade, car_cascade)
    
    st.image(
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        caption=f"Result: {detection_type}",
        channels="RGB",
        use_container_width=True
    )
