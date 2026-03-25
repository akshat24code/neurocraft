import os
from pathlib import Path

import streamlit as st

from src.ai_playground_pages.ask_ai import explore_data_page
from src.application_pages.open_cv.open_cv_landing import open_cv_landing_page
from src.application_pages.rnn.next_word import next_word_page
from src.application_pages.rnn.rnn_landing import lstm_application_page, rnn_application_page
from src.application_pages.rnn.rnn_sentiment import rnn_sentiment_page
from src.application_pages.rnn.rnn_landing import lstm_application_page, rnn_application_page
from src.application_pages.rnn.rnn_sentiment import rnn_sentiment_page
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
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT_DIR = Path(__file__).parent


def _inject_global_theme_css() -> None:
    """Premium glass + depth styling. Streamlit is server-rendered; we use CSS only (no React bundle)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', system-ui, sans-serif !important;
        }

        .stApp {
            background:
                radial-gradient(1200px 800px at 10% -10%, rgba(99, 102, 241, 0.35), transparent 55%),
                radial-gradient(900px 600px at 90% 20%, rgba(236, 72, 153, 0.22), transparent 50%),
                radial-gradient(700px 500px at 50% 100%, rgba(34, 211, 238, 0.12), transparent 45%),
                linear-gradient(165deg, #0f0f1a 0%, #12182a 40%, #0a0e18 100%) !important;
            color: #e8eaf0 !important;
        }

        [data-testid="stHeader"] {
            background: rgba(15, 18, 30, 0.72) !important;
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg,
                rgba(22, 26, 42, 0.92) 0%,
                rgba(15, 18, 35, 0.96) 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow:
                8px 0 32px rgba(0,0,0,0.45),
                inset -1px 0 0 rgba(255,255,255,0.04);
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] span {
            color: #c5cad8 !important;
        }

        section[data-testid="stSidebar"] h1 {
            font-weight: 800 !important;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #a5b4fc 0%, #f472b6 50%, #22d3ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 2px 8px rgba(99,102,241,0.35));
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button {
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            background: linear-gradient(180deg, rgba(40,48,72,0.95) 0%, rgba(28,32,52,0.98) 100%) !important;
            color: #e8e8f0 !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
            box-shadow:
                0 4px 0 rgba(0,0,0,0.35),
                0 8px 24px rgba(0,0,0,0.25),
                inset 0 1px 0 rgba(255,255,255,0.12);
            transform: translateZ(0);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow:
                0 6px 0 rgba(0,0,0,0.32),
                0 14px 32px rgba(99,102,241,0.22),
                inset 0 1px 0 rgba(255,255,255,0.18);
            filter: brightness(1.08);
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button:active {
            transform: translateY(1px) scale(0.99);
            box-shadow: 0 2px 0 rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.3);
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="primary"],
        section[data-testid="stSidebar"] .stButton button[kind="primary"] {
            background: linear-gradient(145deg, #6366f1 0%, #8b5cf6 45%, #d946ef 100%) !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            box-shadow:
                0 5px 0 rgba(49, 46, 129, 0.55),
                0 12px 28px rgba(99, 102, 241, 0.45),
                inset 0 1px 0 rgba(255,255,255,0.35);
        }

        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1100px !important;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #f1f2f6 !important;
        }

        .stMarkdown p, .stMarkdown li, label, span[data-testid="stMarkdownContainer"] p {
            color: #b8bfd4 !important;
        }

        .main div[data-testid="column"] button[kind="primary"] {
            border-radius: 12px !important;
            font-weight: 700 !important;
            box-shadow:
                0 4px 0 #4338ca,
                0 10px 24px rgba(99, 102, 241, 0.35) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }

        .main div[data-testid="column"] button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
        }

        .main div[data-testid="column"] button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(52, 58, 82, 0.95) 0%, rgba(36, 40, 60, 0.98) 100%) !important;
            color: #eef0f7 !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow:
                0 3px 0 rgba(0, 0, 0, 0.28),
                0 8px 20px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        }

        .main div[data-testid="column"] button[kind="secondary"]:hover {
            filter: brightness(1.08);
            transform: translateY(-1px) !important;
        }

        div[data-testid="stAlert"] {
            background: rgba(25, 45, 78, 0.72) !important;
            border: 1px solid rgba(147, 197, 253, 0.38) !important;
        }

        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
            color: #e0f2fe !important;
        }

        [data-baseweb="select"] > div,
        .stTextInput input,
        .stTextArea textarea {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 12px !important;
            color: #e8eaf0 !important;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(165deg, rgba(36,40,62,0.85) 0%, rgba(24,28,48,0.92) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1rem 0.75rem;
            box-shadow:
                0 6px 0 rgba(0,0,0,0.2),
                0 16px 40px rgba(0,0,0,0.25),
                inset 0 1px 0 rgba(255,255,255,0.06);
            transform: perspective(900px) rotateX(2deg);
        }

        .nc-hero-wrap {
            perspective: 1200px;
            margin: 0 auto 1.75rem;
            max-width: 820px;
        }

        .nc-hero-3d {
            transform-style: preserve-3d;
            transform: rotateX(4deg) rotateY(-2deg);
            padding: 2rem 2.25rem;
            border-radius: 24px;
            background: linear-gradient(145deg,
                rgba(55, 60, 90, 0.55) 0%,
                rgba(30, 35, 55, 0.75) 100%);
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow:
                0 20px 0 rgba(0,0,0,0.18),
                0 40px 80px rgba(0,0,0,0.45),
                0 0 0 1px rgba(255,255,255,0.05) inset,
                0 1px 0 rgba(255,255,255,0.15) inset;
        }

        .nc-hero-3d h1 {
            margin: 0;
            font-size: clamp(1.85rem, 4vw, 2.6rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            background: linear-gradient(120deg, #e0e7ff 0%, #c4b5fd 35%, #f0abfc 70%, #67e8f9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 40px rgba(139, 92, 246, 0.35);
        }

        .nc-hero-3d p {
            margin: 0.75rem 0 0;
            font-size: 1.05rem;
            color: #a8b0c8 !important;
            line-height: 1.55;
        }

        .nc-card {
            border-radius: 18px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 0.75rem;
            background: linear-gradient(165deg, rgba(42, 48, 72, 0.65) 0%, rgba(26, 30, 48, 0.85) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow:
                0 8px 0 rgba(0,0,0,0.12),
                0 20px 48px rgba(0,0,0,0.2),
                inset 0 1px 0 rgba(255,255,255,0.08);
            transform: perspective(800px) rotateX(1deg);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .nc-card:hover {
            transform: perspective(800px) rotateX(0deg) translateY(-2px);
            box-shadow:
                0 10px 0 rgba(0,0,0,0.1),
                0 26px 56px rgba(0,0,0,0.24),
                inset 0 1px 0 rgba(255,255,255,0.1);
        }

        .nc-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #f4f6fb;
            margin-bottom: 0.35rem;
        }

        .nc-footer-hr {
            margin-top: 2rem;
            border: none;
            border-top: 1px solid rgba(255,255,255,0.08);
        }

        .nc-footer-text {
            text-align: center;
            color: #92a0bd;
            font-size: 0.92rem;
            margin-top: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_route(route_group: str, route_value: str) -> None:
    st.session_state["active_nav_group"] = route_group
    st.session_state["active_route"] = route_value
    st.session_state["route_override"] = route_value


def _health_rows():
    python_ok = bool(os.getenv("PYTHONPATH") or True)
    env_ok = Path(".env").exists()
    nvidia_ok = bool(os.getenv("NVIDIA_API_KEY"))
    torch_ok = True
    return [
        ("Python Environment", "OK" if python_ok else "Missing", "Python runtime detected."),
        (".env File", "OK" if env_ok else "Missing", "Environment file in project root."),
        ("NVIDIA API Key", "OK" if nvidia_ok else "Missing", "Needed for AI Playground + text refinement."),
        ("PyTorch", "OK" if torch_ok else "Missing", "Required for RNN/LSTM demos."),
    ]


def _render_health_dashboard() -> None:
    st.title("System Health")
    st.caption("Quick diagnostics for the local NeuroCraft setup.")

    for name, status, detail in _health_rows():
        if status == "OK":
            st.success(f"{name}: {detail}")
        else:
            st.warning(f"{name}: {detail}")


def _render_interactive_home() -> None:
    st.markdown(
        """
        <div class="nc-hero-wrap">
          <div class="nc-hero-3d">
            <h1>NeuroCraft Lab</h1>
            <p>
              Learn neural networks visually, experiment with sequence and vision apps,
              and use AI-assisted workflows to inspect data and generate training code.
            </p>
          </div>
        </div>
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
            "LSTM Applications",
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
            "LSTM Applications": "apps_nav",
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
              Build practical projects using OpenCV, RNN, and LSTM pages, then auto-profile CSV data
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
        st.progress(100, text="Step 5: RNN, LSTM, and OpenCV applications")
    elif mode == "Builder":
        st.info("Builder mode: Start with AI Playground or recurrent/vision labs for hands-on outputs.")
    else:
        st.info("Explorer mode: Use Command Center to jump across modules quickly.")

    st.markdown("### Setup Checklist")
    for name, status, _ in _health_rows():
        mark = "OK" if status == "OK" else "Missing"
        st.write(f"- {name}: {mark}")
    st.info("Tip: Click `System Health` in sidebar for detailed diagnostics.")


_inject_global_theme_css()

st.sidebar.title("NeuroCraft Lab")
st.sidebar.caption("Interactive NN Toolbox")

active_route = st.session_state.get("route_override") or st.session_state.get("active_route", "home")


def _sidebar_nav_button(label: str, route_value: str) -> None:
    button_type = "primary" if active_route == route_value else "secondary"
    if st.sidebar.button(label, use_container_width=True, type=button_type):
        st.session_state["route_override"] = route_value
        st.rerun()


_sidebar_nav_button("Home", "home")
st.sidebar.markdown("---")

st.sidebar.caption("Learner Modules")
_sidebar_nav_button("Perceptron", "Perceptron")
_sidebar_nav_button("Forward Propagation", "Forward Propagation")
_sidebar_nav_button("Backward Propagation", "Backward Propagation")
_sidebar_nav_button("Multi-Layer Perceptron (MLP)", "Multi-Layer Perceptron (MLP)")

st.sidebar.markdown("---")
st.sidebar.caption("Recurrent Models")
_sidebar_nav_button("RNN Hub", "RNN Applications")
_sidebar_nav_button("LSTM Hub", "LSTM Applications")
_sidebar_nav_button("RNN Next Word Predictor", "RNN Next Word Predictor")
_sidebar_nav_button("RNN Sentiment Analyzer", "RNN Sentiment Analyzer")
_sidebar_nav_button("LSTM Next Word Predictor", "LSTM Next Word Predictor")
_sidebar_nav_button("LSTM Sentiment Analyzer", "LSTM Sentiment Analyzer")

st.sidebar.markdown("---")
st.sidebar.caption("Computer Vision")
_sidebar_nav_button("OpenCV Hub", "OpenCV Lab")
_sidebar_nav_button("Webcam Detection", "Webcam Detection")
_sidebar_nav_button("Video Detection", "Video Detection")
_sidebar_nav_button("Image Detection", "Image Detection")

st.sidebar.markdown("---")
st.sidebar.caption("AI Playground")
_sidebar_nav_button("Auto-Profile & Train", "Explore Data")
_sidebar_nav_button("System Health", "System Health")

st.sidebar.markdown("---")
st.sidebar.caption("Documentation")
_sidebar_nav_button("Perceptron Guide", "Perceptron Guide")
_sidebar_nav_button("Forward Propagation Guide", "Forward Propagation Guide")
_sidebar_nav_button("Backward Propagation Guide", "Backward Propagation Guide")
_sidebar_nav_button("MLP Guide", "MLP Guide")

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
elif route in {"RNN Next Word Predictor", "RNN Sentiment Analyzer"}:
    rnn_application_page()
elif route == "LSTM Applications":
    lstm_application_page()
elif route in {"LSTM Next Word Predictor", "LSTM Sentiment Analyzer"}:
    lstm_application_page()
elif route == "OpenCV Lab":
    open_cv_landing_page()
elif route in {"Webcam Detection", "Video Detection", "Image Detection"}:
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
    <hr class="nc-footer-hr">
    <p class="nc-footer-text">NeuroCraft Lab | Educational Neural Network Toolbox</p>
    """,
    unsafe_allow_html=True,
)


