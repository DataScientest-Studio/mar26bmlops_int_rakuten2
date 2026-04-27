import streamlit as st
import requests
import os
from PIL import Image
from pathlib import Path

# ======================================================
# CONFIGURATION
# ======================================================
API_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict/upload"
MODEL_INFO_ENDPOINT = f"{API_URL}/model/info"
HEALTH_ENDPOINT = f"{API_URL}/health"

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"

IMG_INTRO = ASSET_DIR / "Intro_tab_MLFLOW.png"
IMG_DEPLOY = ASSET_DIR / "Training_Deployment.png"
IMG_EXPERIMENTS = ASSET_DIR / "ExperimentsO.png"
IMG_METRICS = ASSET_DIR / "Metrics.png"
IMG_RUN = ASSET_DIR / "metricsprorun.png"
IMG_PARAMS = ASSET_DIR / "Parameterprorun.png"
IMG_REG = ASSET_DIR / "RegM.png"

# ======================================================
# UI SETUP
# ======================================================
st.set_page_config(
    page_title="Rakuten MLOps Presentation",
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

    .note-box {
        background: #FEFCE8;
        border: 1px solid #FACC15;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-top: 1rem;
        color: #334155;
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

    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }

    .metric-title {
        color: #64748B;
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: #0F172A;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .small-muted {
        color: #64748B;
        font-size: 0.88rem;
    }

        .promo-card {
        border-radius: 16px;
        padding: 1rem;
        min-height: 150px;
        border: 1px solid #CBD5E1;
        background: #F8FAFC;
    }

    .promo-card.champion {
        background: #ECFDF5;
        border-color: #10B981;
    }

    .promo-card.candidate {
        background: #FFF7ED;
        border-color: #F59E0B;
    }

    .promo-title {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .promo-score {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.35rem;
        line-height: 1.1;
    }

    .promo-badge {
        display: inline-block;
        padding: 0.3rem 0.55rem;
        border-radius: 999px;
        background: white;
        border: 1px solid #CBD5E1;
        color: #475569;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .promo-card.champion .promo-badge {
        border-color: #10B981;
        color: #065F46;
    }

    .promo-card.candidate .promo-badge {
        border-color: #F59E0B;
        color: #9A3412;
    }

    .promo-status {
        margin-top: 0.7rem;
        color: #475569;
        font-size: 0.9rem;
    }

    .premium-panel {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .premium-panel h4 {
        margin-top: 0;
        margin-bottom: 0.45rem;
        color: #0F172A;
        font-size: 1.02rem;
        font-weight: 800;
    }

    .premium-panel p {
        margin: 0;
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    .premium-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    .premium-list li {
        margin-bottom: 0.35rem;
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

def note(text: str):
    st.markdown(f"""<div class="note-box">{text}</div>""", unsafe_allow_html=True)

def info(text: str):
    st.markdown(f"""<div class="info-box">{text}</div>""", unsafe_allow_html=True)

def warn(text: str):
    st.markdown(f"""<div class="warn-box">{text}</div>""", unsafe_allow_html=True)

def show_image(path: Path, caption: str = ""):
    if path.exists():
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            st.image(path, width=650)

            if caption:
                st.caption(caption)
    else:
        st.warning(f"Image not found: {path}")

def promotion_flag_card(title: str, score: str, status: str, variant: str = "neutral"):
    label = "Tested"
    extra_class = ""

    if variant == "champion":
        label = "Champion"
        extra_class = "champion"
    elif variant == "candidate":
        label = "Candidate"
        extra_class = "candidate"

    st.markdown(
        f"""
        <div class="promo-card {extra_class}">
            <div class="promo-title">{title}</div>
            <div class="promo-score">{score}</div>
            <div class="promo-badge">{label}</div>
            <div class="promo-status">Status: {status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# HEADER
# ======================================================
st.title("Rakuten MLOps System")
st.caption("MLflow Tracking, Registry, Model Governance and Production Validation")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Why MLflow", "MLflow Workflow", "MLflow Lifecycle", "Live Validation"]
)


# ======================================================
# TAB 1 — WHY MLFLOW
# ======================================================
with tab1:
    hero(
        "Why We Introduced MLflow",
        "From manual experimentation to professional model lifecycle management."
    )

    left, right = st.columns(2)

    with left:
        card(
            "Before MLflow",
            """
            • models stored locally<br>
            • difficult run comparison<br>
            • no central experiment history<br>
            • higher risk of using the wrong version
            """
        )

    with right:
        card(
            "After MLflow",
            """
            • centralized run tracking<br>
            • visible parameters and metrics<br>
            • model registry with versions<br>
            • candidate / champion logic
            """
        )

    st.divider()
    st.subheader("What We Logged During Training")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Parameters", "epochs, batch size, learning rate, dataset fraction")

    with c2:
        card("Metrics", "micro-F1, macro-F1, validation loss and evaluation values")

    with c3:
        card("Artifacts", "model files, plots, reports and saved outputs")

    with c4:
        card("Metadata", "runtime information, timestamps and experiment context")

    note(
        "<b>MLflow improved reproducibility, comparison, version control and deployment readiness.</b> "
    )


# ======================================================
# TAB 2 — MLFLOW WORKFLOW
# ======================================================
with tab2:
    hero(
        "MLflow in Our End-to-End Workflow",
        "How experiment tracking and model governance fit into our MLOps architecture."
    )

    show_image(
        IMG_INTRO,
        caption="Source: Bytepawn.com — adapted for academic presentation"
    )
    

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card(
            "Track Runs",
            "Store parameters, metrics, artifacts and model outputs for every training run."
        )

    with c2:
        card(
            "Compare Results",
            "Evaluate multiple experiments in a reproducible and structured way."
        )

    with c3:
        card(
            "Register Versions",
            "Promote successful runs into governed model versions instead of unmanaged local files."
        )

    with c4:
        card(
            "Deploy Safely",
            "Connect experiment management with model-serving decisions and deployment readiness."
        )

    note(
        "<b>MLflow connects experimentation with deployment."
        " It transforms isolated training runs into a structured machine learning workflow.</b> "
    )


# ======================================================
# TAB 3 — MLFLOW LIFECYCLE
# ======================================================
with tab3:
    hero(
        "MLflow Lifecycle Inside Our Pipeline",
        "From training to tracking, registry and deployment."
    )

    show_image( IMG_DEPLOY, caption="Source: BingInfo.in — adapted for educational use" )

    st.divider()
    st.subheader("Real Project Evidence")

    # 1 Tracking
    st.markdown("### 1. Experiment Tracking")
    left, right = st.columns([1, 1.8])

    with left:
        info(
            """
            <b>What we track:</b><br><br>
            • parameters<br>
            • dataset fraction<br>
            • evaluation metrics<br>
            • artifacts<br>
            • runtime
            """
        )


    with right:
        show_image(
            IMG_METRICS,
            caption="Tracked metrics dashboard"
        )
    st.markdown(
            """
            <div class="small-muted">
            This proves that model quality is measured systematically and not guessed manually.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # 2 Single run transparency
    st.markdown("### 2. Single Run Transparency")
    left, right = st.columns([1.4, 1])

    with left:
        show_image(
            IMG_PARAMS,
            caption="Single run parameters"
        )

    with right:
        card(
            "What a single run reveals",
            """
            • exact parameter settings<br>
            • detailed run-level transparency<br>
            • reproducibility<br>
            • traceability of decisions
            """
        )

    st.divider()

    # 3 Registry
    st.markdown("### 3. Model Registry and Promotion")

    st.image(
        IMG_REG,
        caption="Registry with model versions and aliases",
        width=820
    )

    info(
        """
        <b>Operational meaning:</b><br><br>
        Versioned models are managed centrally.<br>
        Champion and Candidate aliases simplify deployment decisions.<br>
        The best validated model can be promoted safely into production.
        """
    )
    # 4 Compare & Promote Automation
    st.divider()

    st.markdown("## 4. Compare & Promote Automation")
    st.caption("Automated retraining, fair comparison, and model promotion with MLflow aliases")

    c1, c2, c3 = st.columns(3)

    with c1:
        promotion_flag_card(
            title="Run A",
            score="0.24",
            status="30% data fraction tested",
            variant="neutral"
        )

    with c2:
        promotion_flag_card(
            title="Run B",
            score="0.27",
            status="highest micro-F1 → Champion",
            variant="champion"
        )

    with c3:
        promotion_flag_card(
            title="Run C",
            score="0.25",
            status="second-best → Candidate",
            variant="candidate"
        )

    st.markdown("")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            """
            <div class="premium-panel">
                <h4>Automated orchestration flow</h4>
                <ul class="premium-list">
                    <li>Launch multiple sequential training jobs</li>
                    <li>Use increasing data fractions (30%, 60%, 100%)</li>
                    <li>Register every trained version in MLflow</li>
                    <li>Evaluate all versions on the same validation split</li>
                    <li>Use micro-F1 for objective comparison</li>
                    <li>Promote the best version to <b>Champion</b></li>
                    <li>Keep the second-best as <b>Candidate</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
    <div class="premium-panel">
<h4>Why this matters</h4>

<ul style="font-size:18px; line-height:1.9; padding-left:22px; margin-top:10px;">
<li>Faster model promotion</li>
<li>Fair comparison using same validation data</li>
<li>Reproducible deployment decisions</li>
<li>Continuous retraining ready</li>
</ul>

</div>
           
            """,
            unsafe_allow_html=True
        )
    
    note(
        "<b>Built on top of MLflow Registry:</b> "
        "Champion always points to the best validated model, "
        "while Candidate remains the next deployment option."
    )
   
# ======================================================
# TAB 4 — LIVE VALIDATION
# ======================================================
with tab4:
    hero(
        "Live Deployment Validation",
        "Proof that backend service and model source are operational."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Check API Health", use_container_width=True):
            try:
                r = requests.get(HEALTH_ENDPOINT, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    st.success("API is healthy.")

                    summary = {
                        "status": data.get("status"),
                        "model_loaded": data.get("model_loaded"),
                        "db_connected": data.get("db_connected"),
                        "version": data.get("version"),
                    }
                    st.json(summary)
                else:
                    st.error(f"API returned status code {r.status_code}.")
            except Exception:
                warn("<b>API unreachable.</b> The backend is not reachable from the current session.")

    with col2:
        if st.button("Check Champion Model", use_container_width=True):
            try:
                r = requests.get(MODEL_INFO_ENDPOINT, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    st.success("Champion endpoint reachable.")

                    summary = {
                        "model_type": data.get("model_type"),
                        "model_source": data.get("model_source"),
                        "is_mock": data.get("is_mock"),
                        "device": data.get("device"),
                    }
                    if "num_labels" in data:
                        summary["num_labels"] = data.get("num_labels")

                    st.json(summary)
                else:
                    st.error(f"Model endpoint returned status code {r.status_code}.")
            except Exception:
                warn("<b>Model endpoint unreachable.</b> The model-serving API is not reachable from the current session.")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Tracking</div><div class="metric-value">ON</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Registry</div><div class="metric-value">ACTIVE</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">API</div><div class="metric-value">LIVE</div></div>',
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Serving</div><div class="metric-value">READY</div></div>',
            unsafe_allow_html=True
        )

    note(
        "<b>This proves that our project is not only theoretical.<b>"
        "<b>The ML model can be managed and served in a real runtime environment.</b>"
    )