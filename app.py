import streamlit as st
import requests
import os
import pandas as pd
from PIL import Image
from pathlib import Path

# ======================================================
# CONFIGURATION
# ======================================================
API_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict/upload"
MODEL_INFO_ENDPOINT = f"{API_URL}/model/info"
HEALTH_ENDPOINT = f"{API_URL}/health"
RELOAD_ENDPOINT = f"{API_URL}/admin/reload"

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "mlflow" / "assets"

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
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.info(f"Missing image: {path.name}")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["MLflow Workflow", "Why MLflow", "MLflow Lifecycle", "Live Validation", "Live Demo"]
)

# ======================================================
# TAB 1 — MLFLOW WORKFLOW
# ======================================================
with tab1:
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
        "<b>Presentation message:</b> MLflow connects experimentation with deployment. "
        "It transforms isolated training runs into a structured machine learning workflow."
    )

# ======================================================
# TAB 2 — WHY MLFLOW
# ======================================================
with tab2:
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
        "<b>Presentation message:</b> MLflow improved reproducibility, comparison, version control and deployment readiness."
    )

# ======================================================
# TAB 3 — MLFLOW LIFECYCLE
# ======================================================
with tab3:
    hero(
        "MLflow Lifecycle Inside Our Pipeline",
        "From training to tracking, registry and deployment."
    )

    show_image(
        IMG_DEPLOY,
        caption="Source: BingInfo.in — adapted for educational use"
    )

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
        st.markdown(
            """
            <div class="small-muted">
            This proves that model quality is measured systematically and not guessed manually.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        show_image(
            IMG_METRICS,
            caption="Tracked metrics dashboard"
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
    left, right = st.columns([1.5, 1])

    with left:
        show_image(
            IMG_REG,
            caption="Registry with model versions and aliases"
        )

    with right:
        info(
            """
            <b>Operational meaning:</b><br><br>
            Versioned models are managed centrally.<br><br>
            The best validated model can be promoted and used as the selected serving candidate.
            """
        )

    st.divider()

    # 4 Compare & Promote Automation
    st.markdown("## 4. Compare & Promote Automation")
    st.caption("Automated champion selection using fair model comparison")

    c1, c2, c3 = st.columns(3)

    with c1:
        promotion_flag_card(
            title="Run A",
            score="0.24",
            status="evaluated on same validation split",
            variant="neutral"
        )

    with c2:
        promotion_flag_card(
            title="Run B",
            score="0.27",
            status="best micro-F1",
            variant="champion"
        )

    with c3:
        promotion_flag_card(
            title="Run C",
            score="0.25",
            status="second-best version",
            variant="candidate"
        )

    st.markdown("")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
            <div class="premium-panel">
                <h4>Automated orchestration flow</h4>
                <ul class="premium-list">
                    <li>Multiple training jobs run sequentially</li>
                    <li>Each run can use a larger training data fraction</li>
                    <li>Every registered version is evaluated on the same validation split</li>
                    <li>micro-F1 is used for fair comparison</li>
                    <li>The best validated model is promoted to <b>champion</b></li>
                    <li>The second-best model can remain <b>candidate</b></li>
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
                <p>
                Instead of choosing models manually, we automate a controlled comparison process.
                This makes model promotion faster, fairer and more reproducible.
                It also strengthens governance because deployment decisions are based on measured performance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    note(
        "<b>Presentation message:</b> We automated model selection on top of MLflow. "
        "The best validated version becomes champion, while the next-best version can remain candidate."
    )

# ======================================================
# TAB 4 — LIVE VALIDATION
# ======================================================
with tab4:
    hero(
        "Live Deployment Validation",
        "Proof that backend service and model source are operational."
    )

    col1, col2, col3 = st.columns(3)

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

    with col3:
        if st.button("Reload Champion", use_container_width=True):
            # Triggers POST /admin/reload on the API — forces the ModelService
            # to re-load the champion from the MLflow Registry. Call this after
            # Airflow promoted a new model so the API picks it up without a restart.
            try:
                r = requests.post(RELOAD_ENDPOINT, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    st.success(
                        f"Reloaded — source: **{data.get('model_source')}**, "
                        f"mock: **{data.get('is_mock')}**"
                    )
                    st.json(data)
                else:
                    st.error(f"Reload returned status code {r.status_code}.")
                    st.text(r.text)
            except Exception as e:
                warn(f"<b>Reload failed.</b> {e}")

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
        "<b>Presentation message:</b> This proves that our project is not only theoretical. "
        "The ML model can be managed and served in a real runtime environment."
    )

# ======================================================
# TAB 5 — LIVE DEMO
# ======================================================
with tab5:
    st.title("🛍️ Rakuten Color Predictor")

    demo_mode = st.selectbox(
        "Select Prediction Mode",
        ["Single Product Upload", "Bulk Batch Processing (Excel + Images)"]
    )

    st.divider()

    # --- MODE 1: SINGLE UPLOAD ---
    if demo_mode == "Single Product Upload":
        st.subheader("Single Item Prediction")
        st.markdown("Upload a product image and provide a title and description.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Data")
            item_name = st.text_area("Product Name", placeholder="e.g., Blue Denim Jacket")
            item_caption = st.text_area(
                "Product Description",
                placeholder="e.g., Classic light-wash blue jean jacket with silver buttons."
            )
            uploaded_file = st.file_uploader("Product Image", type=["jpg", "jpeg", "png"], key="single_upload")

        with col2:
            st.subheader("Preview")
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                st.info("Upload an image to see the preview.")

        if st.button("Predict Color", type="primary", use_container_width=True):
            if not uploaded_file or not item_caption:
                st.error("Please provide both a description and an image.")
            else:
                img_bytes = uploaded_file.getvalue()
                files = {"image": (uploaded_file.name, img_bytes, uploaded_file.type)}
                params = {"item_name": item_name, "item_caption": item_caption}

                with st.spinner("Analyzing image and text features..."):
                    try:
                        response = requests.post(PREDICT_ENDPOINT, params=params, files=files, timeout=30)
                        if response.status_code == 200:
                            prediction = response.json()

                            st.divider()
                            st.subheader("Prediction Result")

                            res_col_img, res_col_details = st.columns([1, 2])

                            with res_col_img:
                                st.image(img_bytes, use_container_width=True)

                            with res_col_details:
                                st.markdown(f"**Item:** {item_name}")

                                predicted = prediction.get("predicted_colors", [])
                                all_scores = prediction.get("all_scores", [])

                                if predicted:
                                    st.markdown("**Detected Colors & Confidence:**")
                                    for score_data in all_scores:
                                        if score_data['color'] in predicted:
                                            conf = score_data['score']
                                            st.write(f"🏷️ **{score_data['color']}**")
                                            st.progress(conf, text=f"{conf:.2%} confidence")
                                else:
                                    st.warning("No colors met the confidence threshold.")
                        else:
                            st.error(f"API Error ({response.status_code}): {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to the API at {API_URL}. Ensure the Docker container is running.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    # --- MODE 2: BATCH UPLOAD ---
    else:
        st.subheader("Bulk Batch Processing")
        st.markdown("Upload an Excel/CSV and corresponding images to process multiple items.")

        metadata_file = st.file_uploader("1. Upload Metadata (Excel/CSV)", type=["xlsx", "csv"])

        uploaded_images = st.file_uploader(
            "2. Upload All Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            if not metadata_file or not uploaded_images:
                st.error("Please upload both the metadata file and the image set.")
            else:
                try:
                    if metadata_file.name.endswith('csv'):
                        df = pd.read_csv(metadata_file)
                    else:
                        df = pd.read_excel(metadata_file)

                    image_map = {img.name: img for img in uploaded_images}

                    payload_files = []
                    item_names = []
                    item_captions = []

                    for _, row in df.iterrows():
                        fname = str(row.get('image_file_name', '')).strip()
                        if fname in image_map:
                            item_names.append(row.get('item_name', ''))
                            item_captions.append(row.get('item_caption', ''))
                            img_obj = image_map[fname]
                            payload_files.append(
                                ("images", (img_obj.name, img_obj.getvalue(), img_obj.type))
                            )

                    if not payload_files:
                        st.error("No matching images found. Check that filenames in Excel match your uploaded files.")
                    else:
                        with st.spinner(f"Processing batch of {len(payload_files)} items..."):
                            BATCH_UPLOAD_ENDPOINT = f"{API_URL}/predict/batch/upload"

                            response = requests.post(
                                BATCH_UPLOAD_ENDPOINT,
                                data={
                                    "item_names": item_names,
                                    "item_captions": item_captions
                                },
                                files=payload_files,
                                timeout=120
                            )

                            if response.status_code == 200:
                                results = response.json().get("predictions", [])
                                st.success("Batch Processing Complete!")

                                for i, res in enumerate(results):
                                    with st.container():
                                        col_img, col_details = st.columns([1, 2])
                                        with col_img:
                                            st.image(payload_files[i][1][1], use_container_width=True)
                                        with col_details:
                                            st.markdown(f"**Item:** {item_names[i]}")
                                            predicted = res.get("predicted_colors", [])
                                            all_scores = res.get("all_scores", [])
                                            if predicted:
                                                st.markdown("**Detected Colors & Confidence:**")
                                                for score_data in all_scores:
                                                    if score_data['color'] in predicted:
                                                        conf = score_data['score']
                                                        st.write(f"**{score_data['color']}**")
                                                        st.progress(conf, text=f"{conf:.2%} confidence")
                                            else:
                                                st.warning("No colors met the threshold.")
                                        st.divider()
                            else:
                                st.error(f"Batch API Error: {response.text}")
                except Exception as e:
                    st.error(f"Process failed: {e}")