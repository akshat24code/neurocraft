import os

import cv2
import streamlit as st

from .open_cv_core import (
    CASCADE_PATHS,
    run_eye_smile_detection,
    run_face_count,
    run_face_detection,
    run_stop_sign_detection,
    run_colored_object_detection,
)


DETECTION_OPTIONS = (
    "Face Detection",
    "Eye + Smile Detection",
    "Stop Sign Detection",
    "Real Time Face Count",
    "Colored Object Detection",
    "Edge Detection",
    "Vehicle Detection",
)


def load_cascade(key: str) -> cv2.CascadeClassifier | None:
    """Load Haar cascade by key; warn and return None when missing/invalid."""
    path = CASCADE_PATHS.get(key, "")
    if not os.path.exists(path):
        st.warning(f"Cascade not found: {path}")
        return None

    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        st.warning(f"Failed to load cascade: {path}")
        return None

    return clf


def prepare_detectors(detection_type: str):
    """Build cascades needed for the selected detection mode."""
    face_cascades: list = []
    eye_cascade = None
    smile_cascade = None
    car_cascade = None

    if detection_type not in ["Stop Sign Detection", "Colored Object Detection", "Edge Detection"]:
        if detection_type == "Vehicle Detection":
             car_cascade = load_cascade("car")
        else:
            face_cascades = [
                load_cascade("default"),
                load_cascade("alt"),
                load_cascade("alt_tree"),
            ]
            face_cascades = [c for c in face_cascades if c is not None]

            if not face_cascades:
                st.error("No face cascade could be loaded. Please check the cascade files.")
                return None

            if detection_type == "Eye + Smile Detection":
                eye_cascade = load_cascade("eye")
                smile_cascade = load_cascade("smile")

    return face_cascades, eye_cascade, smile_cascade, car_cascade


def apply_detection(img, detection_type, face_cascades, eye_cascade, smile_cascade, car_cascade=None):
    """Dispatch frame/image to the selected detector."""
    from .open_cv_core import (
        run_edge_detection,
        run_vehicle_detection,
    )
    
    if detection_type == "Real Time Face Count":
        img, _ = run_face_count(img, face_cascades)
    elif detection_type == "Stop Sign Detection":
        img = run_stop_sign_detection(img)
    elif detection_type == "Colored Object Detection":
        img = run_colored_object_detection(img)
    elif detection_type == "Edge Detection":
        img = run_edge_detection(img)
    elif detection_type == "Vehicle Detection":
        img = run_vehicle_detection(img, car_cascade)
    elif detection_type == "Face Detection":
        img = run_face_detection(img, face_cascades)
    elif detection_type == "Eye + Smile Detection":
        img = run_eye_smile_detection(img, face_cascades, eye_cascade, smile_cascade)
    # Alphabet is handled separately in the UI landing
    return img

