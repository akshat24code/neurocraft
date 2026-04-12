import streamlit as st

from .open_cv_image import run_image_use_case
from .open_cv_shared import (
    DETECTION_OPTIONS,
    prepare_detectors,
)
from .open_cv_video import run_video_upload_use_case
from .open_cv_webcam import run_webcam_use_case


INPUT_OPTIONS = ("Webcam", "Upload Video", "Image")

DETECTION_META: dict[str, dict[str, str]] = {
    "Face Detection": {
        "emoji": "👤",
        "tagline": "Frontal faces · bounding boxes",
        "long": (
            "Uses an **ensemble of Haar cascades** (default, alt, alt_tree) with **non-maximum suppression** "
            "to merge overlapping detections. Good for demos and learning how cascade sliding windows behave."
        ),
        "tech": "HaarCascade · `detectMultiScale` · NMS merge",
        "tips": "Works best with front-facing faces and even lighting. Side profiles may be missed.",
    },
    "Eye + Smile Detection": {
        "emoji": "😊",
        "tagline": "ROI inside face · eyes & smile",
        "long": (
            "First finds face ROIs, then runs **eye** and **smile** cascades inside each region. "
            "Shows how hierarchical detection (coarse → fine) is composed in classical CV."
        ),
        "tech": "Face ROI · nested `detectMultiScale` · histogram equalization in ROI",
        "tips": "If smiles are missed, try better frontal pose or brighter scene.",
    },
    "Stop Sign Detection": {
        "emoji": "🛑",
        "tagline": "Color + shape heuristics",
        "long": (
            "**No face cascades** — uses **HSV red masking**, morphological cleanup, and contour aspect-ratio checks "
            "to propose stop-sign-like blobs. Illustrates geometry + color priors instead of template cascades."
        ),
        "tech": "HSV range · morphology · contour bbox · aspect filter",
        "tips": "Works on clear red signs; cluttered red backgrounds may cause false positives.",
    },
    "Real Time Face Count": {
        "emoji": "🔢",
        "tagline": "Same faces · count overlay",
        "long": (
            "Same face pipeline as face detection but draws **per-face indices** and a **running count** overlay. "
            "Useful for crowd-style visualizations (approximate, not identity tracking)."
        ),
        "tech": "Ensemble face detect · enumerated boxes · overlay text",
        "tips": "For dense crowds, counts may duplicate or miss — cascades are not deep trackers.",
    },
    "Colored Object Detection": {
        "emoji": "🎨",
        "tagline": "Dynamic Color Tracking (R/G/B)",
        "long": (
            "Converts the feed from BGR to **HSV color space** to build robust masks for **Red, Green, and Blue** objects. "
            "Filters noise via morphology, applies contours filtering, and labels the tracked areas dynamically in real time!"
        ),
        "tech": "HSV Masking · Morphological Opening · findContours",
        "tips": "Ensure good lighting. The tracker looks for specifically strong reds, blues, and greens.",
    },
    "Edge Detection": {
        "emoji": "📐",
        "tagline": "Canny Edge Discovery",
        "long": "Uses the **Canny algorithm** to find structural outlines. Great for seeing the 'skeleton' of objects in real-time.",
        "tech": "cv2.Canny · Multi-stage algorithm",
    },
    "Vehicle Detection": {
        "emoji": "🚗",
        "tagline": "Car & Traffic Detection",
        "long": "Detects moving or stationary vehicles using specialized Haar Cascades trained on traffic data.",
        "tech": "HaarCascade_cars · detectMultiScale",
    },
}

INPUT_META: dict[str, dict[str, str]] = {
    "Webcam": {
        "emoji": "📷",
        "tagline": "Live camera stream",
        "long": (
            "**Local:** `cv2.VideoCapture` for low latency. **Cloud:** WebRTC (`streamlit-webrtc`) with STUN; "
            "optional TURN secrets for strict networks. Detection runs per frame."
        ),
        "tech": "Local DirectShow / WebRTC `sendrecv` · `VideoProcessorBase`",
    },
    "Upload Video": {
        "emoji": "🎞️",
        "tagline": "MP4 / common formats",
        "long": (
            "Reads uploaded video **frame-by-frame**, runs the same detection function, "
            "and can export a processed clip. Best for repeatable demos without a camera."
        ),
        "tech": "`av` / OpenCV decode · frame loop",
    },
    "Image": {
        "emoji": "🖼️",
        "tagline": "PNG / JPG · still frames",
        "long": (
            "Single image or bundled **sample images**. Ideal for tuning cascade perception and comparing "
            "outputs across detection modes on a fixed input."
        ),
        "tech": "OpenCV imread · optional resize",
    },
}


def _inject_opencv_landing_css() -> None:
    st.markdown(
        """
        <style>
        /* OpenCV landing — 3D-style picker cards (bordered column containers; only used on this page) */
        section.main div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px !important;
            background: linear-gradient(165deg, rgba(48, 54, 82, 0.55) 0%, rgba(28, 32, 52, 0.85) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            box-shadow:
                0 10px 0 rgba(0, 0, 0, 0.18),
                0 24px 48px rgba(0, 0, 0, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        section.main div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-4px);
            box-shadow:
                0 14px 0 rgba(0, 0, 0, 0.14),
                0 32px 56px rgba(99, 102, 241, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        }
        section.main div[data-testid="stVerticalBlockBorderWrapper"] button[kind="primary"] {
            background: linear-gradient(145deg, #6366f1 0%, #8b5cf6 55%, #a855f7 100%) !important;
            box-shadow: 0 4px 0 rgba(67, 56, 202, 0.85), 0 12px 28px rgba(99, 102, 241, 0.45) !important;
        }
        section.main div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(44, 50, 78, 0.95) 0%, rgba(30, 34, 52, 0.98) 100%) !important;
            color: #e8eaf4 !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_opencv_session() -> None:
    if "opencv_det_type" not in st.session_state:
        st.session_state.opencv_det_type = DETECTION_OPTIONS[0]
    if "opencv_input_mode" not in st.session_state:
        st.session_state.opencv_input_mode = INPUT_OPTIONS[0]


def _render_detection_cards() -> None:
    _ensure_opencv_session()
    st.markdown("##### Step 1 — Detection mode")
    st.caption("Pick what the pipeline should look for. Each card uses a different classical-CV story.")

    num_cols = 3
    cols = None
    for i, opt in enumerate(DETECTION_OPTIONS):
        if i % num_cols == 0:
            cols = st.columns(num_cols)
            
        meta = DETECTION_META[opt]
        selected = st.session_state.opencv_det_type == opt
        with cols[i % num_cols]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:2rem;line-height:1;'>{meta['emoji']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center;font-weight:700;color:#e8eaf4;'>{opt}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(meta["tagline"])
                label = "✓ Active" if selected else "Tap to select"
                if st.button(
                    label,
                    key=f"ncv_det_pick_{i}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.opencv_det_type = opt
                    st.rerun()

    cur = DETECTION_META[st.session_state.opencv_det_type]
    with st.expander(f"Details · {st.session_state.opencv_det_type}", expanded=True):
        st.markdown(cur["long"])
        st.caption(f"**Pipeline:** {cur['tech']}")
        if cur.get("tips"):
            st.caption(f"**Tip:** {cur['tips']}")


def _render_input_cards() -> None:
    _ensure_opencv_session()
    st.markdown("##### Step 2 — Input source")
    st.caption("How frames enter the detector — live, file video, or still image.")

    cols = st.columns(3)
    for i, mode in enumerate(INPUT_OPTIONS):
        meta = INPUT_META[mode]
        selected = st.session_state.opencv_input_mode == mode
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"<div style='text-align:center;font-size:2rem;line-height:1;'>{meta['emoji']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;font-weight:700;color:#e8eaf4;'>{mode}</div>", unsafe_allow_html=True)
                st.caption(meta["tagline"])
                label = "✓ Active" if selected else "Tap to select"
                if st.button(
                    label,
                    key=f"ncv_in_pick_{i}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.opencv_input_mode = mode
                    st.rerun()

    cur = INPUT_META[st.session_state.opencv_input_mode]
    with st.expander(f"Details · {st.session_state.opencv_input_mode}", expanded=False):
        st.markdown(cur["long"])
        st.caption(f"**Stack:** {cur['tech']}")


def open_cv_landing_page():
    _inject_opencv_landing_css()
    _ensure_opencv_session()

    st.title("Detection using OpenCV")
    st.markdown(
        """
        <div style="
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(147,197,253,0.35);
            background: linear-gradient(135deg, rgba(30,58,95,0.65) 0%, rgba(30,41,72,0.75) 100%);
            box-shadow: 0 8px 32px rgba(0,0,0,0.25);
            margin-bottom: 1rem;
        ">
            <p style="margin:0; color:#dbeafe; font-size:0.95rem; line-height:1.55;">
                <strong style="color:#93c5fd;">Haar cascades &amp; classical CV</strong> —
                fast CPU demos, not SOTA deep learning. Expect noise in hard scenes; use this to
                <em>learn pipelines</em>, not production surveillance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_detection_cards()

    detection_type = st.session_state.opencv_det_type
    
    detectors = prepare_detectors(detection_type)
    if detectors is None:
        return
    face_cascades, eye_cascade, smile_cascade, car_cascade = detectors

    st.divider()
    _render_input_cards()

    mode = st.session_state.opencv_input_mode

    st.divider()
    st.caption("Live workspace — controls and preview appear below.")

    if mode == "Webcam":
        run_webcam_use_case(detection_type, face_cascades, eye_cascade, smile_cascade, car_cascade)
    elif mode == "Upload Video":
        run_video_upload_use_case(
            detection_type,
            face_cascades,
            eye_cascade,
            smile_cascade,
            car_cascade,
        )
    else:
        run_image_use_case(detection_type, face_cascades, eye_cascade, smile_cascade, car_cascade)
