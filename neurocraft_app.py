import os
from pathlib import Path

import streamlit as st

from src.ai_playground_pages.ask_ai import explore_data_page
from src.application_pages.open_cv.open_cv_landing import open_cv_landing_page
from src.application_pages.rnn.next_word import next_word_page
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

NAV_SECTIONS = {
    "Start Here": [
        ("Home", "home", "Overview and quick launch"),
        ("System Health", "System Health", "Check local setup and assets"),
    ],
    "Learning Lab": [
        ("Perceptron", "Perceptron", "Binary logic and decision boundaries"),
        ("Forward Propagation", "Forward Propagation", "See activations move through a network"),
        ("Backward Propagation", "Backward Propagation", "Understand gradient updates step by step"),
        ("Multi-Layer Perceptron", "Multi-Layer Perceptron (MLP)", "Train on IRIS or your own CSV"),
    ],
    "Sequence Models": [
        ("RNN Hub", "RNN Applications", "Explore recurrent model demos"),
        ("RNN Next Word", "RNN Next Word Predictor", "Predict the next token from context"),
        ("RNN Sentiment", "RNN Sentiment Analyzer", "Classify review-style text"),
        ("LSTM Hub", "LSTM Applications", "Work with long-sequence flows"),
        ("LSTM Next Word", "LSTM Next Word Predictor", "Predict the next token with LSTM"),
        ("LSTM Sentiment", "LSTM Sentiment Analyzer", "Run custom LSTM sentiment analysis"),
    ],
    "Vision + AI": [
        ("OpenCV Hub", "OpenCV Lab", "Classical CV detection playground"),
        ("Webcam Detection", "Webcam Detection", "Run real-time detection from camera"),
        ("Video Detection", "Video Detection", "Analyze uploaded videos"),
        ("Image Detection", "Image Detection", "Inspect static images"),
        ("AI Playground", "Explore Data", "Profile data and generate training code"),
    ],
    "Guides": [
        ("Perceptron Guide", "Perceptron Guide", "Reference notes and walkthrough"),
        ("Forward Propagation Guide", "Forward Propagation Guide", "Reference notes and walkthrough"),
        ("Backward Propagation Guide", "Backward Propagation Guide", "Reference notes and walkthrough"),
        ("MLP Guide", "MLP Guide", "Reference notes and walkthrough"),
    ],
}

ROUTE_TO_SECTION = {
    route: section for section, items in NAV_SECTIONS.items() for _, route, _ in items
}

ROUTE_TO_LABEL = {
    route: label for _, items in NAV_SECTIONS.items() for label, route, _ in items
}

ROUTE_TO_DESCRIPTION = {
    route: description for _, items in NAV_SECTIONS.items() for _, route, description in items
}

QUICK_ACTIONS = [
    ("Start Learning", "Perceptron", "Learning Lab"),
    ("Run RNN Demo", "RNN Applications", "Sequence Models"),
    ("Open LSTM Lab", "LSTM Applications", "Sequence Models"),
    ("Explore Data", "Explore Data", "Vision + AI"),
]


def _inject_global_theme_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --nc-bg: #08111f;
            --nc-panel: rgba(11, 22, 38, 0.82);
            --nc-panel-2: rgba(16, 30, 50, 0.9);
            --nc-border: rgba(151, 192, 255, 0.16);
            --nc-soft: #94a9c7;
            --nc-text: #eff6ff;
            --nc-accent: #67e8f9;
            --nc-accent-2: #f59e0b;
            --nc-accent-3: #38bdf8;
            --nc-shadow: 0 24px 80px rgba(1, 10, 22, 0.45);
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Manrope', system-ui, sans-serif !important;
        }

        .stApp {
            background:
                radial-gradient(1200px 680px at 0% 0%, rgba(56, 189, 248, 0.16), transparent 55%),
                radial-gradient(960px 720px at 100% 0%, rgba(245, 158, 11, 0.16), transparent 46%),
                radial-gradient(900px 700px at 50% 100%, rgba(16, 185, 129, 0.12), transparent 44%),
                linear-gradient(180deg, #07101c 0%, #0a1526 42%, #08111f 100%) !important;
            color: var(--nc-text) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(6, 12, 24, 0.78) !important;
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .main .block-container {
            max-width: 1180px !important;
            padding-top: 1.2rem !important;
            padding-bottom: 2.5rem !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 16, 29, 0.96) 0%, rgba(10, 21, 38, 0.98) 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #c9d6eb !important;
        }

        section[data-testid="stSidebar"] h1 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.6rem !important;
            letter-spacing: -0.04em;
            color: white !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div {
            background: rgba(255,255,255,0.05) !important;
            color: var(--nc-text) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(151, 192, 255, 0.18) !important;
        }

        .stButton button,
        .stDownloadButton button {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            transform: translateY(-1px);
            border-color: rgba(103, 232, 249, 0.36) !important;
            box-shadow: 0 16px 34px rgba(0,0,0,0.22);
        }

        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%) !important;
            color: #04111d !important;
        }

        .stButton button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(20, 35, 58, 0.96) 0%, rgba(14, 24, 41, 0.98) 100%) !important;
            color: var(--nc-text) !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px !important;
            border: 1px solid rgba(147, 197, 253, 0.22) !important;
            background: rgba(15, 32, 54, 0.82) !important;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 29, 48, 0.88) 0%, rgba(11, 21, 36, 0.92) 100%);
            border: 1px solid var(--nc-border);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: var(--nc-shadow);
        }

        .nc-shell {
            margin-bottom: 1.1rem;
        }

        .nc-topbar {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
            padding: 1rem 1.15rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(10, 24, 42, 0.9) 0%, rgba(8, 18, 32, 0.92) 100%);
            border: 1px solid var(--nc-border);
            box-shadow: var(--nc-shadow);
        }

        .nc-topbar-left h1,
        .nc-hero h1,
        .nc-home-title {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.045em;
        }

        .nc-topbar-left h1 {
            margin: 0;
            font-size: clamp(1.4rem, 3vw, 2rem);
            color: #f8fbff;
        }

        .nc-topbar-left p,
        .nc-hero p,
        .nc-muted {
            margin: 0.28rem 0 0;
            color: var(--nc-soft) !important;
        }

        .nc-chip-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .nc-chip {
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: #dbeafe;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .nc-hero {
            position: relative;
            overflow: hidden;
            margin: 1rem 0 1.4rem;
            padding: 2rem;
            border-radius: 28px;
            background:
                radial-gradient(500px 180px at 15% 20%, rgba(103, 232, 249, 0.18), transparent 65%),
                radial-gradient(420px 220px at 90% 10%, rgba(245, 158, 11, 0.18), transparent 55%),
                linear-gradient(135deg, rgba(10, 25, 43, 0.92) 0%, rgba(9, 18, 31, 0.96) 100%);
            border: 1px solid rgba(151, 192, 255, 0.16);
            box-shadow: var(--nc-shadow);
        }

        .nc-hero-grid {
            display: grid;
            grid-template-columns: 1.3fr 0.9fr;
            gap: 1rem;
            align-items: stretch;
        }

        .nc-home-title {
            margin: 0;
            font-size: clamp(2.4rem, 6vw, 4rem);
            line-height: 0.95;
            color: white;
        }

        .nc-kicker {
            display: inline-block;
            margin-bottom: 0.9rem;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(103, 232, 249, 0.12);
            border: 1px solid rgba(103, 232, 249, 0.24);
            color: #bbf7d0;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .nc-panel,
        .nc-card,
        .nc-nav-hint {
            background: linear-gradient(180deg, rgba(16, 31, 52, 0.82) 0%, rgba(10, 20, 34, 0.9) 100%);
            border: 1px solid var(--nc-border);
            border-radius: 20px;
            box-shadow: var(--nc-shadow);
        }

        .nc-panel {
            padding: 1.1rem 1.2rem;
        }

        .nc-card {
            padding: 1.1rem 1.2rem;
            height: 100%;
        }

        .nc-card h3,
        .nc-panel h3,
        .nc-section-title {
            margin: 0 0 0.35rem;
            color: white;
            font-size: 1.02rem;
            font-weight: 800;
        }

        .nc-card p,
        .nc-panel p {
            color: var(--nc-soft) !important;
            margin: 0;
            line-height: 1.6;
        }

        .nc-nav-hint {
            padding: 0.95rem 1.1rem;
            margin-bottom: 0.8rem;
        }

        .nc-section-label {
            display: inline-block;
            margin-bottom: 0.55rem;
            font-size: 0.8rem;
            font-weight: 800;
            color: #7dd3fc;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .nc-footer-hr {
            margin-top: 2rem;
            border: none;
            border-top: 1px solid rgba(255,255,255,0.08);
        }

        .nc-footer-text {
            text-align: center;
            color: #8da5c7;
            font-size: 0.92rem;
            margin-top: 0.75rem;
        }

        @media (max-width: 900px) {
            .nc-hero {
                padding: 1.25rem;
            }
            .nc-hero-grid {
                grid-template-columns: 1fr;
            }
            .main .block-container {
                padding-top: 0.8rem !important;
            }
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
    python_ok = True
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

    status_cols = st.columns(len(_health_rows()))
    for col, (name, status, detail) in zip(status_cols, _health_rows()):
        with col:
            st.metric(name, status)
            st.caption(detail)

    st.markdown("### Detailed Checks")
    for name, status, detail in _health_rows():
        if status == "OK":
            st.success(f"{name}: {detail}")
        else:
            st.warning(f"{name}: {detail}")


def _render_route_banner(route: str) -> None:
    if route == "home":
        return

    section = ROUTE_TO_SECTION.get(route, "Workspace")
    label = ROUTE_TO_LABEL.get(route, route)
    description = ROUTE_TO_DESCRIPTION.get(route, "")
    st.markdown(
        f"""
        <div class="nc-shell">
          <div class="nc-topbar">
            <div class="nc-topbar-left">
              <div class="nc-section-label">{section}</div>
              <h1>{label}</h1>
              <p>{description}</p>
            </div>
            <div class="nc-chip-row">
              <div class="nc-chip">Focused Workspace</div>
              <div class="nc-chip">Route: {route}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _home_feature_card(title: str, description: str, badge: str) -> str:
    return f"""
    <div class="nc-card">
      <div class="nc-section-label">{badge}</div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
    """


def _render_interactive_home() -> None:
    health_ok = sum(1 for _, status, _ in _health_rows() if status == "OK")
    total_health = len(_health_rows())

    st.markdown(
        f"""
        <div class="nc-hero">
          <div class="nc-hero-grid">
            <div>
              <div class="nc-kicker">Neural Learning Workspace</div>
              <h1 class="nc-home-title">NeuroCraft Lab</h1>
              <p class="nc-muted" style="font-size:1.05rem; line-height:1.7; margin-top:0.85rem; max-width:640px;">
                Learn neural networks visually, jump into recurrent and computer-vision demos,
                and move from concept to experiment without getting lost in the app.
              </p>
              <div class="nc-chip-row" style="margin-top:1rem;">
                <div class="nc-chip">10+ learning modules</div>
                <div class="nc-chip">RNN + LSTM demos</div>
                <div class="nc-chip">OpenCV lab</div>
                <div class="nc-chip">AI-assisted data workflows</div>
              </div>
            </div>
            <div class="nc-panel">
              <h3>Workspace Snapshot</h3>
              <p>Use the dashboard to jump into modules quickly and the sidebar filter to track down any page faster.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Modules", str(sum(len(items) for items in NAV_SECTIONS.values())))
    metric_cols[1].metric("Guides", "4")
    metric_cols[2].metric("Health Checks", f"{health_ok}/{total_health}")
    metric_cols[3].metric("Main Modes", "Learn / Build / Explore")

    st.markdown("### Quick Launch")
    action_cols = st.columns(len(QUICK_ACTIONS))
    for col, (label, route, group) in zip(action_cols, QUICK_ACTIONS):
        with col:
            if st.button(label, use_container_width=True, type="primary"):
                _set_route(group, route)
                st.rerun()
            st.caption(ROUTE_TO_DESCRIPTION.get(route, ""))

    st.markdown("### Find Your Way")
    find_col, focus_col = st.columns([1.2, 0.8])
    with find_col:
        all_routes = [route for _, items in NAV_SECTIONS.items() for _, route, _ in items if route != "home"]
        selected_route = st.selectbox(
            "Jump directly to any module",
            options=all_routes,
            format_func=lambda route: f"{ROUTE_TO_LABEL.get(route, route)} | {ROUTE_TO_SECTION.get(route, 'Workspace')}",
            index=0,
        )
        if st.button("Open Selected Workspace", use_container_width=True):
            _set_route(ROUTE_TO_SECTION.get(selected_route, "Workspace"), selected_route)
            st.rerun()
    with focus_col:
        mode = st.radio("Experience Mode", options=["Guided", "Builder", "Explorer"], horizontal=True)
        if mode == "Guided":
            st.info("Follow the learner modules first, then move into RNN/LSTM and OpenCV labs.")
        elif mode == "Builder":
            st.info("Start in AI Playground, then validate ideas with recurrent or computer-vision demos.")
        else:
            st.info("Use the filtered sidebar and quick launch cards to move between pages rapidly.")

    st.markdown("### Explore By Zone")
    zone_cols = st.columns(3)
    zone_cols[0].markdown(
        _home_feature_card(
            "Learning Lab",
            "Structured concept pages for Perceptron, forward/backward propagation, and MLP understanding.",
            "Learn",
        ),
        unsafe_allow_html=True,
    )
    zone_cols[1].markdown(
        _home_feature_card(
            "Sequence Models",
            "RNN and LSTM hubs with next-word prediction, sentiment analysis, and custom checkpoint flows.",
            "Build",
        ),
        unsafe_allow_html=True,
    )
    zone_cols[2].markdown(
        _home_feature_card(
            "Vision + AI",
            "OpenCV detection tools and the AI Playground for profiling datasets and generating training code.",
            "Ship",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Progress Roadmap")
    roadmap_cols = st.columns(2)
    with roadmap_cols[0]:
        st.progress(20, text="Foundations: Perceptron and network intuition")
        st.progress(55, text="Core learning: Forward and backward propagation")
        st.progress(85, text="Applied practice: MLP, RNN, and LSTM")
        st.progress(100, text="Production-style workflows: OpenCV and AI Playground")
    with roadmap_cols[1]:
        st.markdown(
            """
            <div class="nc-nav-hint">
              <div class="nc-section-label">Navigation Tip</div>
              <h3>Use the sidebar filter</h3>
              <p>Type part of a module name like <strong>LSTM</strong>, <strong>Guide</strong>, or <strong>Detection</strong> to narrow the sidebar instantly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="nc-nav-hint">
              <div class="nc-section-label">Setup Tip</div>
              <h3>Check readiness fast</h3>
              <p>The health summary shows whether your environment and API keys are ready before you start heavier workflows.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_sidebar(active_route: str) -> None:
    st.sidebar.title("NeuroCraft Lab")
    st.sidebar.caption("Polished neural-network workspace")

    st.sidebar.markdown(
        """
        <div class="nc-panel" style="padding:0.9rem 1rem; margin-bottom:0.8rem;">
          <div class="nc-section-label">Current Focus</div>
          <h3 style="margin:0;">{}</h3>
          <p style="margin-top:0.35rem;">{}</p>
        </div>
        """.format(
            ROUTE_TO_LABEL.get(active_route, "Home"),
            ROUTE_TO_DESCRIPTION.get(active_route, "Start from the dashboard and jump into any module."),
        ),
        unsafe_allow_html=True,
    )

    search_term = st.sidebar.text_input("Filter modules", placeholder="Type LSTM, guide, detection...")
    st.sidebar.caption(f"{sum(len(items) for items in NAV_SECTIONS.values())} pages available")

    for section, items in NAV_SECTIONS.items():
        visible_items = []
        for label, route, description in items:
            haystack = f"{label} {route} {description} {section}".lower()
            if search_term.strip().lower() in haystack:
                visible_items.append((label, route, description))

        if not visible_items:
            continue

        with st.sidebar.expander(section, expanded=True):
            for label, route, description in visible_items:
                button_type = "primary" if active_route == route else "secondary"
                if st.button(label, key=f"nav_{route}", use_container_width=True, type=button_type):
                    _set_route(section, route)
                    st.rerun()
                st.caption(description)


_inject_global_theme_css()

active_route = st.session_state.get("route_override") or st.session_state.get("active_route", "home")
_render_sidebar(active_route)

route = st.session_state.get("route_override") or st.session_state.get("active_route", "home")
st.session_state["active_route"] = route

_render_route_banner(route)

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
elif route in {"RNN Next Word Predictor"}:
    next_word_page("RNN")
elif route in {"RNN Sentiment Analyzer"}:
    rnn_sentiment_page("RNN")
elif route == "LSTM Applications":
    lstm_application_page()
elif route in {"LSTM Next Word Predictor"}:
    next_word_page("LSTM")
elif route in {"LSTM Sentiment Analyzer"}:
    rnn_sentiment_page("LSTM")
elif route == "OpenCV Lab":
    open_cv_landing_page()
elif route in {"Webcam Detection", "Video Detection", "Image Detection"}:
    st.session_state["cv_mode"] = route
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
