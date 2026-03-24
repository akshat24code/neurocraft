import os
from pathlib import Path

import streamlit as st

from src.ai_playground_pages.ask_ai import explore_data_page
from src.application_pages.open_cv.open_cv_landing import open_cv_landing_page
from src.application_pages.rnn.rnn_landing import rnn_application_page
from src.assets.documents.back_propagation import back_propagation_docs_page
from src.assets.documents.forward_propagation import forward_propagation_docs_page
from src.assets.documents.mnp import mnp_docs_page
from src.assets.documents.perceptron import perceptron_docs_page
from src.learner_pages.backward_propagation import backward_propagation_page
from src.learner_pages.forward_propagation import forward_propagation_page
from src.learner_pages.mlp import mlp_page
from src.learner_pages.perceptron_ui import perceptron_page


st.set_page_config(
    page_title="NeuroCraft Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT_DIR = Path(__file__).parent


def _set_route(tab_key: str, value: str) -> None:
    # Avoid mutating widget-bound session keys outside widget callbacks.
    # Use an explicit route override for button-based navigation.
    _ = tab_key
    st.session_state["route_override"] = value


def _clear_other_tabs(active: str) -> None:
    st.session_state["route_override"] = None
    all_tabs = {
        "playground": "playground_nav",
        "learner": "learner_nav",
        "apps": "apps_nav",
        "docs": "docs_nav",
    }
    for tab, key in all_tabs.items():
        if tab != active:
            st.session_state[key] = None


def _go_home() -> None:
    st.session_state["playground_nav"] = None
    st.session_state["learner_nav"] = None
    st.session_state["apps_nav"] = None
    st.session_state["docs_nav"] = None
    st.session_state["route_override"] = "home"


def _health_rows() -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    checks.append(
        (
            "NVIDIA API key",
            "OK" if bool(nvidia_key.strip()) else "Missing",
            "Required for AI Playground dataset analysis and code generation.",
        )
    )

    iris_file = ROOT_DIR / "data" / "IRIS.csv"
    checks.append(
        (
            "IRIS sample dataset",
            "OK" if iris_file.exists() else "Missing",
            f"Expected at {iris_file}",
        )
    )

    banner_file = ROOT_DIR / "src" / "assets" / "image" / "nn_image.jpg"
    checks.append(
        (
            "Home banner image",
            "OK" if banner_file.exists() else "Missing",
            f"Expected at {banner_file}",
        )
    )

    return checks


def _render_health_dashboard() -> None:
    st.title("System Health Dashboard")
    st.caption("Quick checks to verify your toolbox setup.")

    rows = _health_rows()
    ok_count = sum(1 for _, status, _ in rows if status == "OK")
    total_count = len(rows)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Checks Passed", f"{ok_count}/{total_count}")
    with c2:
        st.metric("Readiness", f"{round((ok_count / total_count) * 100)}%")

    for name, status, details in rows:
        if status == "OK":
            st.success(f"{name}: {status}")
        else:
            st.error(f"{name}: {status}")
        st.caption(details)

    st.markdown("### Next Steps")
    st.code(
        "copy .env.example .env\n"
        "# Add your key in .env\n"
        "NVIDIA_API_KEY=your_real_key_here\n"
        "streamlit run neurocraft_app.py",
        language="powershell",
    )


def _render_interactive_home() -> None:
    st.markdown(
        """
        <style>
        .nc-card {
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 10px;
            background: rgba(30, 41, 59, 0.06);
        }
        .nc-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 style="text-align:center;">NeuroCraft Lab</h1>
        <p style="text-align:center; font-size:16px;">
        Interactive neural network toolbox for learning and practical AI demos.
        </p>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Experience Mode",
        options=["Guided", "Builder", "Explorer"],
        horizontal=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules", "10+")
    c2.metric("Modes", "Webcam / Video / Image")
    c3.metric("Built-in Data", "IRIS")
    c4.metric("LLM Assist", "NVIDIA")

    st.markdown("### Command Center")
    command = st.selectbox(
        "Jump directly to a module",
        options=[
            "Perceptron",
            "Forward Propagation",
            "Backward Propagation",
            "Multi-Layer Perceptron (MLP)",
            "Explore Data",
            "RNN Applications",
            "OpenCV Lab",
            "System Health",
        ],
        index=0,
    )
    if st.button("Launch Selected Module", type="primary", use_container_width=True):
        key_map = {
            "Explore Data": "playground_nav",
            "Perceptron": "learner_nav",
            "Forward Propagation": "learner_nav",
            "Backward Propagation": "learner_nav",
            "Multi-Layer Perceptron (MLP)": "learner_nav",
            "RNN Applications": "apps_nav",
            "OpenCV Lab": "apps_nav",
            "System Health": "apps_nav",
        }
        _set_route(key_map.get(command, "learner_nav"), command)
        st.rerun()

    st.markdown("### Feature Tour")
    tab1, tab2, tab3 = st.tabs(["Learn", "Build", "Deploy"])
    with tab1:
        st.markdown(
            """
            <div class="nc-card">
              <div class="nc-title">Learner Modules</div>
              Step-by-step modules for Perceptron, Forward/Backward Propagation, and MLP.
              Use this mode when you want concept clarity with structured flow.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tab2:
        st.markdown(
            """
            <div class="nc-card">
              <div class="nc-title">Applications + AI Playground</div>
              Build practical projects using OpenCV and RNN pages, then auto-profile CSV data
              and generate training scripts from AI Playground.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tab3:
        st.markdown(
            """
            <div class="nc-card">
              <div class="nc-title">Streamlit Deployment</div>
              App is deployment-ready. Keep API keys in Streamlit secrets and deploy from GitHub.
              Follow the new `DEPLOY_STREAMLIT.md` guide.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if mode == "Guided":
        st.markdown("### Guided Path")
        st.progress(10, text="Step 1: Perceptron basics")
        st.progress(35, text="Step 2: Forward propagation intuition")
        st.progress(55, text="Step 3: Backpropagation updates")
        st.progress(75, text="Step 4: MLP training on IRIS/custom CSV")
        st.progress(100, text="Step 5: RNN and OpenCV applications")
    elif mode == "Builder":
        st.info("Builder mode: Start with AI Playground or OpenCV Lab for hands-on outputs.")
    else:
        st.info("Explorer mode: Use Command Center to jump across modules quickly.")

    st.markdown("### Setup Checklist")
    for name, status, _ in _health_rows():
        mark = "OK" if status == "OK" else "Missing"
        st.write(f"- {name}: {mark}")
    st.info("Tip: Click `System Health` in sidebar for detailed diagnostics.")


st.sidebar.title("NeuroCraft Lab")
st.sidebar.caption("Interactive NN Toolbox")

active_route = st.session_state.get("route_override") or st.session_state.get("active_route", "home")


def _sidebar_nav_button(label: str, route_value: str) -> None:
    button_type = "primary" if active_route == route_value else "secondary"
    if st.sidebar.button(label, use_container_width=True, type=button_type):
        st.session_state["route_override"] = route_value
        st.rerun()


_sidebar_nav_button("🏠 Home", "home")
st.sidebar.markdown("---")

st.sidebar.caption("Learner Modules")
_sidebar_nav_button("🧠 Perceptron", "Perceptron")
_sidebar_nav_button("➡️ Forward Propagation", "Forward Propagation")
_sidebar_nav_button("⬅️ Backward Propagation", "Backward Propagation")
_sidebar_nav_button("🧩 Multi-Layer Perceptron (MLP)", "Multi-Layer Perceptron (MLP)")

st.sidebar.markdown("---")
st.sidebar.caption("RNN Applications")
_sidebar_nav_button("🔁 RNN Hub", "RNN Applications")
_sidebar_nav_button("✍️ Next Word Predictor", "Next Word Predictor")
_sidebar_nav_button("😊 Sentiment Analyzer", "Sentiment Analyzer")

st.sidebar.markdown("---")
st.sidebar.caption("Computer Vision")
_sidebar_nav_button("👁️ OpenCV Hub", "OpenCV Lab")
_sidebar_nav_button("📷 Webcam Detection", "Webcam Detection")
_sidebar_nav_button("🎞️ Video Detection", "Video Detection")
_sidebar_nav_button("🖼️ Image Detection", "Image Detection")

st.sidebar.markdown("---")
st.sidebar.caption("AI Playground")
_sidebar_nav_button("🤖 Auto-Profile & Train", "Explore Data")
_sidebar_nav_button("🩺 System Health", "System Health")

st.sidebar.markdown("---")
st.sidebar.caption("Documentation")
_sidebar_nav_button("📘 Perceptron Guide", "Perceptron Guide")
_sidebar_nav_button("📘 Forward Propagation Guide", "Forward Propagation Guide")
_sidebar_nav_button("📘 Backward Propagation Guide", "Backward Propagation Guide")
_sidebar_nav_button("📘 MLP Guide", "MLP Guide")

route = st.session_state.get("route_override") or st.session_state.get("active_route", "home")
st.session_state["active_route"] = route


if route == "home":
    _render_interactive_home()

elif route == "Explore Data":
    explore_data_page()
elif route == "Perceptron":
    perceptron_page()
elif route == "Forward Propagation":
    forward_propagation_page()
elif route == "Backward Propagation":
    backward_propagation_page()
elif route == "Multi-Layer Perceptron (MLP)":
    mlp_page()
elif route == "RNN Applications":
    rnn_application_page()
elif route in {"Next Word Predictor", "Sentiment Analyzer"}:
    # Keep direct shortcuts in sidebar while reusing central RNN hub page.
    rnn_application_page()
elif route == "OpenCV Lab":
    open_cv_landing_page()
elif route in {"Webcam Detection", "Video Detection", "Image Detection"}:
    # Reuse existing OpenCV landing page for mode-specific workflows.
    open_cv_landing_page()
elif route == "System Health":
    _render_health_dashboard()
elif route == "Perceptron Guide":
    perceptron_docs_page()
elif route == "Forward Propagation Guide":
    forward_propagation_docs_page()
elif route == "Backward Propagation Guide":
    back_propagation_docs_page()
elif route == "MLP Guide":
    mnp_docs_page()


st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px;">
    NeuroCraft Lab | Educational Neural Network Toolbox
    </p>
    """,
    unsafe_allow_html=True,
)
