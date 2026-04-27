import streamlit as st
import streamlit.components.v1 as components
import os
from pathlib import Path

# ======================================================
# CONFIGURATION
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
DRIFT_REPORT = BASE_DIR / "reports" / "data_drift_report.html"

# ======================================================
# UI SETUP
# ======================================================
st.set_page_config(
    page_title="Monitoring — Rakuten MLOps",
    layout="wide"
)

# ======================================================
# DESIGN
# ======================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1380px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero-light {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #CBD5E1;
        margin-bottom: 1rem;
    }

    .hero-light h3 {
        margin: 0;
        color: #0F172A;
        font-size: 1.4rem;
        font-weight: 800;
    }

    .hero-light p {
        margin: 0.35rem 0 0 0;
        color: #475569;
        font-size: 0.95rem;
    }

    .soft-card {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 14px;
        height: 100%;
    }

    .soft-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        color: #0F172A;
    }

    .soft-card p {
        margin-bottom: 0;
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .info-box {
        background: #F8FAFC;
        border-left: 5px solid #14B8A6;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        color: #334155;
    }

    .warn-box {
        background: #FFF7ED;
        border-left: 5px solid #F97316;
        border-top: 1px solid #FED7AA;
        border-right: 1px solid #FED7AA;
        border-bottom: 1px solid #FED7AA;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        color: #9A3412;
    }

    .metric-header {
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-header h3 {
        margin: 0 0 0.2rem 0;
        color: #0F172A;
        font-size: 1.15rem;
        font-weight: 700;
        font-family: monospace;
    }

    .metric-header .caption {
        color: #64748B;
        font-size: 0.87rem;
        margin-bottom: 0.5rem;
    }

    .metric-header p {
        color: #475569;
        font-size: 0.95rem;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# HELPERS
# ======================================================
def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-light">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def card(title: str, text: str):
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def info(text: str):
    st.markdown(f"""<div class="info-box">{text}</div>""", unsafe_allow_html=True)

def warn(text: str):
    st.markdown(f"""<div class="warn-box">{text}</div>""", unsafe_allow_html=True)

def metric_header(name: str, caption: str, description: str):
    st.markdown(
        f"""
        <div class="metric-header">
            <h3>{name}</h3>
            <div class="caption">{caption}</div>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_image(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.info(f"Missing image: {path.name}")

# ======================================================
# HEADER
# ======================================================
hero(
    "Monitoring",
    "Real-time API observability and training pipeline metrics via Prometheus and Grafana."
)

tab_overview, tab_api, tab_training, tab_drift = st.tabs(
    ["Overview", "API Metrics", "Training Metrics", "Data Drift"]
)

# ======================================================
# TAB — OVERVIEW
# ======================================================
with tab_overview:
    st.markdown("### Two Metric Flows")
    c1, c2 = st.columns(2)
    with c1:
        card(
            "API Metrics",
            "Emitted continuously by the running API. Every prediction and HTTP request is tracked in real time via Prometheus."
        )
    with c2:
        card(
            "Training Metrics",
            "Pushed once at the end of each Airflow pipeline run via Pushgateway. Prometheus scrapes the gateway."
        )
    st.markdown("")
    _, img_col, _ = st.columns([1, 2, 1])
    with img_col:
        show_image(FIGURES_DIR / "Monitoring-overview.png")
    st.markdown(
        "<div style='text-align:center; color:#475569; font-size:0.95rem;'>"
        "<b>Both flows end up in Grafana</b>, provisioned automatically — no manual dashboard configuration needed."
        "</div>",
        unsafe_allow_html=True
    )

# ======================================================
# TAB — API METRICS
# ======================================================
with tab_api:
    api_page = st.radio(
        "api_page",
        ["Flow Diagram", "Color Predictions", "Request Count", "Request Duration"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if api_page == "Flow Diagram":
        _, img_col, _ = st.columns([1, 1, 1])
        with img_col:
            show_image(FIGURES_DIR / "API-metrics-flow.png")

    elif api_page == "Color Predictions":
        metric_header("rakuten_color_predictions_total", "Counter — labeled by color", "Incremented every time a color is predicted.")
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_color_predictions_total.png")

    elif api_page == "Request Count":
        metric_header("rakuten_requests_total", "Counter — labeled by method, endpoint, status_code", "Tracks every HTTP request.")
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_requests_total.png")

    elif api_page == "Request Duration":
        metric_header("rakuten_request_duration_seconds", "Histogram — labeled by method, endpoint", "Records request duration in seconds.")
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_request_duration_seconds.png")

# ======================================================
# TAB — TRAINING METRICS
# ======================================================
with tab_training:
    train_page = st.radio(
        "train_page",
        ["Flow Diagram", "Training Run F1", "Training Duration", "Champion F1", "Champion Version"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if train_page == "Flow Diagram":
        _, img_col, _ = st.columns([1.5, 1, 1.5])
        with img_col:
            show_image(FIGURES_DIR / "Training-metrics-flow.png")

    elif train_page == "Training Run F1":
        metric_header("rakuten_training_run_f1", "Labeled by model_version and run_id", "F1 score per training run.")
        _, img_col, _ = st.columns([1, 14, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_training_run_f1.png")

    elif train_page == "Training Duration":
        metric_header("rakuten_training_duration_seconds", "Labeled by model_version", "Training time per run.")
        _, img_col, _ = st.columns([1, 4, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_training_duration_seconds.png")

    elif train_page == "Champion F1":
        metric_header("rakuten_champion_f1", "Single value", "F1 score of the currently promoted champion model.")
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_champion_f1.png")

    elif train_page == "Champion Version":
        metric_header("rakuten_champion_version", "Single value", "Version number of the champion.")
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            show_image(FIGURES_DIR / "rakuten_champion_version.png")

# ======================================================
# TAB — DATA DRIFT
# ======================================================
with tab_drift:
    st.markdown("### Data Drift with Evidently")
    info(
        "Data drift occurs when the statistical distribution of input data changes over time"
        " - either in the input features, target labels, or both."
    )
    st.markdown("")
    st.markdown("#### Potential Causes in This Project")
    c1, c2 = st.columns(2)
    with c1:
        card(
            "Color Trends Change — Label Drift",
            "The distribution of target labels shifts over time. "
        )
    with c2:
        card(
            "Photo Lighting Change — Training-Serving Skew",
            "Systematic change in photos."
        )
    st.markdown("")
    st.markdown("#### How Evidently Works")
    st.markdown(
        """
        - Compares a **reference** dataset vs a **current** dataset
        - Runs statistical tests per feature column (Jensen-Shannon distance for numerical, chi-square for categorical)
        - Outputs a drift score and a drifted/not-drifted verdict per feature
        - Generates an **interactive HTML report** with distribution visualizations

        In a real production setup, this comparison would run periodically. If drift is detected, it triggers retraining via the Airflow pipeline.
        """
    )
    st.markdown("#### Live Drift Report")
    if DRIFT_REPORT.exists():
        html_content = DRIFT_REPORT.read_text(encoding="utf-8")
        components.html(html_content, height=800, scrolling=True)
    else:
        warn(f"Drift report not found at <code>{DRIFT_REPORT.name}</code>. Run <code>python -m src.monitoring.drift</code> to generate it.")
