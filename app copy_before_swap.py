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
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # width=650 statt full-width — sieht in Wide-Layout sauberer aus
            # caption wird mitgegeben damit die Quellenangabe nicht verloren geht
            st.image(str(path), width=650, caption=caption)
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17 = st.tabs(
    [
        # --- INTRO (8 tabs from colleague's app_intro.py) ---
        "1. Project Overview",
        "2. Business Problem",
        "3. Dataset & Inputs",
        "4. Sample Images",
        "5. Docker Databank on AWS",
        "6. Model Architecture",
        "7. ML Pipeline & API",
        "8. Handover",
        # --- MONITORING (4 tabs from colleague's app_monitoring.py) ---
        "9. Monitoring Overview",
        "10. API Metrics",
        "11. Training Metrics",
        "12. Data Drift",
        # --- OWN (your 5 tabs, unchanged content) ---
        "13. MLflow Workflow",
        "14. Why MLflow",
        "15. MLflow Lifecycle",
        "16. Live Validation",
        "17. Live Demo",
    ]
)


# ============================================================================
# HELPERS for INTRO + MONITORING tabs (shared, prefixed _hub_)
# ============================================================================
import streamlit.components.v1 as _hub_components

@st.cache_data
def _hub_load_csv_safe(path_str):
    p = Path(path_str)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


@st.cache_data
def _hub_resized_image(path_str, target_height=220):
    img = Image.open(path_str).convert("RGB")
    w, h = img.size
    new_w = int((target_height / h) * w)
    return img.resize((new_w, target_height))


def _hub_metric_card(label, value):
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:14px;border:1px solid #e6e6e6;background:#fafafa;min-height:90px;">
            <div style="font-size:14px;color:#666;">{label}</div>
            <div style="font-size:28px;font-weight:700;color:#111;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hub_find(candidates):
    for path in candidates:
        p = Path(path)
        if p.exists():
            return p
    return None


def _hub_show_image(path, caption=""):
    if path and Path(path).exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        name = Path(path).name if path else "(none)"
        st.info(f"Image not found: {name}")


def _hub_card(title, text):
    st.markdown(
        f"""
        <div style="background:white;border:1px solid #E2E8F0;padding:1rem;border-radius:14px;height:100%;">
            <h4 style="margin-top:0;">{title}</h4>
            <p style="margin:0.4rem 0 0 0;color:#475569;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hub_hero(title, subtitle):
    st.markdown(
        f"""
        <div style="padding:1rem 1.2rem;border-radius:16px;background:linear-gradient(135deg,#F8FAFC 0%,#EEF2FF 100%);border:1px solid #CBD5E1;margin-bottom:1rem;">
            <h3 style="margin:0;color:#0F172A;font-size:1.4rem;font-weight:800;">{title}</h3>
            <p style="margin:0.35rem 0 0 0;color:#475569;font-size:0.95rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hub_metric_header(name, caption, description):
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:1rem;">
            <h3 style="margin:0 0 0.2rem 0;color:#0F172A;font-size:1.15rem;font-weight:700;font-family:monospace;">{name}</h3>
            <div style="color:#64748B;font-size:0.87rem;margin-bottom:0.5rem;">{caption}</div>
            <p style="color:#475569;font-size:0.95rem;margin:0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Asset paths — looking in several common locations.
# ----------------------------------------------------------------------------
_HUB_X_TRAIN = _hub_find(["X_train.csv", "data/raw/X_train.csv", "src/streamlit/X_train.csv"])
_HUB_X_TEST = _hub_find(["X_test.csv", "data/raw/X_test.csv", "src/streamlit/X_test.csv"])
_HUB_Y_TRAIN = _hub_find(["y_train.csv", "data/raw/y_train.csv", "src/streamlit/y_train.csv"])

_HUB_IMAGE_DIR = _hub_find([
    "src/streamlit/images", "images", "data/images",
])

# Intro figures
_HUB_AWS_SCREENSHOT = _hub_find([
    "src/streamlit/assets/AWS_EC2_Screenshot.png", "assets/AWS_EC2_Screenshot.png", "AWS_EC2_Screenshot.png",
])
_HUB_MINIO_DATA = _hub_find([
    "src/streamlit/assets/MINIO_Data.png", "assets/MINIO_Data.png", "MINIO_Data.png",
])
_HUB_MINIO_IMAGES = _hub_find([
    "src/streamlit/assets/MINIO_images.png", "assets/MINIO_images.png", "MINIO_images.png",
])
_HUB_MODEL_DIAGRAM = _hub_find([
    "src/streamlit/assets/model-diagram.png", "assets/model-diagram.png", "model-diagram.png",
])
_HUB_END_TO_END = _hub_find([
    "src/streamlit/assets/end_to_end_architecture.png", "assets/end_to_end_architecture.png", "end_to_end_architecture.png",
])
_HUB_PIPELINE_EVO = _hub_find([
    "src/streamlit/assets/pipeline_evolution.png", "assets/pipeline_evolution.png", "pipeline_evolution.png",
])

# Monitoring figures (in reports/figures by colleague's convention)
_HUB_FIG_DIR = _hub_find(["src/streamlit/ice", "src/streamlit/reports/figures"])

def _hub_fig(name):
    if _HUB_FIG_DIR is None:
        return None
    p = Path(_HUB_FIG_DIR) / name
    return p if p.exists() else None

_HUB_DRIFT_REPORT = _hub_find([
    "reports/data_drift_report.html", "data_drift_report.html",
])

@st.cache_data
def _hub_basic_stats(df_x, df_y):
    s = {}
    if df_x is not None:
        s["train_rows"] = len(df_x)
        if "item_caption" in df_x.columns:
            s["missing_caption"] = int(df_x["item_caption"].isna().sum())
    if df_y is not None:
        s["label_rows"] = len(df_y)
        if "color_tags" in df_y.columns:
            s["top_labels"] = df_y["color_tags"].astype(str).value_counts().head(10)
    return s

_hub_X_train = _hub_load_csv_safe(str(_HUB_X_TRAIN)) if _HUB_X_TRAIN else None
_hub_X_test = _hub_load_csv_safe(str(_HUB_X_TEST)) if _HUB_X_TEST else None
_hub_y_train = _hub_load_csv_safe(str(_HUB_Y_TRAIN)) if _HUB_Y_TRAIN else None
_hub_stats = _hub_basic_stats(_hub_X_train, _hub_y_train)

_hub_train_rows = _hub_stats.get("train_rows", "-")
_hub_test_rows = len(_hub_X_test) if _hub_X_test is not None else "-"
_hub_label_rows = _hub_stats.get("label_rows", "-")
_hub_missing_caption = _hub_stats.get("missing_caption", "-")

_hub_sample_images = [
    "100054_10006497_1.jpg",
    "100054_10006798_1.jpg",
    "100054_10006900_1.jpg",
    "100054_10006905_1.jpg",
    "100054_10008846_1.jpg",
]

# ============================================================================
# TAB 1 — PROJECT OVERVIEW
# ============================================================================
with tab1:
    st.header("1. Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: _hub_metric_card("Training rows", str(_hub_train_rows))
    with c2: _hub_metric_card("Test rows", str(_hub_test_rows))
    with c3: _hub_metric_card("Label rows", str(_hub_label_rows))
    with c4: _hub_metric_card("Missing captions", str(_hub_missing_caption))
    st.markdown("### Objective")
    st.write(
        "Build a multimodal AI system that predicts product colour tags from three inputs: "
        "**product image**, **product title**, and **product description**."
    )
    st.markdown("### Challenge")
    st.success(
        "This is a multi-label classification task. One product can have one or more colours, "
        "for example: ['Black'] or ['Black', 'White']."
    )
    st.markdown("### Presentation")
    st.write(
        "This project is based on the Rakuten Multimodal Colour Extraction challenge. "
        "The aim is to extract structured colour information from unstructured e-commerce data. "
        "We organize the system using Dockerized services: a MinIO data bank on AWS EC2, "
        "an ML pipeline container, and an API container."
    )


# ============================================================================
# TAB 2 — BUSINESS PROBLEM
# ============================================================================
with tab2:
    st.header("2. Business Problem")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Why this matters")
        st.write(
            "In e-commerce, product metadata is often incomplete or inconsistent. "
            "Sellers may upload an image and title, but colour information may be missing, "
            "ambiguous, or not standardized."
        )
        st.markdown("### Impact")
        st.write(
            "Accurate colour tagging improves:\n"
            "- search and filtering\n"
            "- product grouping\n"
            "- recommendation systems\n"
            "- catalog consistency"
        )
    with right:
        st.info(
            "**Input:** image + title + description\n\n"
            "**Task:** predict one or more colour tags\n\n"
            "**Outcome:** structured metadata for downstream systems"
        )
    st.markdown("### Presentation")
    st.write(
        "From a business point of view, this project converts raw product information into useful metadata. "
        "This metadata can later be used by platforms for filtering, discovery, recommendation, and catalog cleanup."
    )


# ============================================================================
# TAB 3 — DATASET & INPUTS
# ============================================================================
with tab3:
    st.header("3. Dataset & Inputs")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🖼️ Image")
        st.write("Visual information about the product appearance, colour, texture, and shape.")
    with b:
        st.markdown("#### 📝 Title")
        st.write("Short product name. Often contains strong colour-related clues.")
    with c:
        st.markdown("#### 📄 Description")
        st.write("Additional text context. Useful when image information is ambiguous.")

    st.markdown("### Training data preview")
    if _hub_X_train is not None:
        st.dataframe(_hub_X_train.head(5), use_container_width=True)
    else:
        st.warning("X_train.csv not found. Put it in project root, data/, or src/streamlit/.")

    st.markdown("### Label preview")
    if _hub_y_train is not None:
        st.dataframe(_hub_y_train.head(5), use_container_width=True)
    else:
        st.warning("y_train.csv not found. Put it in project root, data/, or src/streamlit/.")

    _hub_top_labels = _hub_stats.get("top_labels")
    if _hub_top_labels is not None:
        st.markdown("### Most frequent label strings")
        st.bar_chart(_hub_top_labels)

    st.markdown("### Presentation")
    st.write(
        "The dataset combines visual and textual information. Images capture appearance, "
        "while titles and descriptions provide semantic clues. The target is multi-label, meaning the model "
        "must decide independently which colour tags apply to each product."
    )


# ============================================================================
# TAB 4 — SAMPLE IMAGES
# ============================================================================
with tab4:
    st.header("4. Sample Product Images")
    st.write(
        "These sample product images show why colour extraction can be challenging. "
        "Some products are metallic, reflective, dark, or visually ambiguous."
    )

    if _HUB_IMAGE_DIR:
        cols = st.columns(5)
        shown = 0
        for idx, img_name in enumerate(_hub_sample_images):
            img_path = Path(_HUB_IMAGE_DIR) / img_name
            if img_path.exists():
                with cols[idx % 5]:
                    img = _hub_resized_image(str(img_path), target_height=190)
                    st.image(img, caption=img_name.replace(".jpg", ""))
                shown += 1
        if shown == 0:
            st.warning("Sample image files were not found inside the images folder.")
    else:
        st.warning("Images folder not found. Expected one of: src/streamlit/images/, images/, data/images/")

    st.markdown("### Presentation")
    st.write(
        "These examples illustrate why image-only classification can be difficult. "
        "Lighting, shadows, product material, and background can all influence the visual colour. "
        "That is why we combine image information with text information."
    )


# ============================================================================
# TAB 5 — DOCKER DATABANK ON AWS
# ============================================================================
with tab5:
    st.header("5. Docker Databank on AWS EC2")
    st.markdown("### What we built")
    st.write(
        "We created a Dockerized data bank using **MinIO Object Store** and deployed it on an **AWS EC2 instance**. "
        "The data bank stores both the CSV files and the product images centrally, "
        "so that team members and downstream services can access the same data source."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### AWS EC2 instance")
        _hub_show_image(_HUB_AWS_SCREENSHOT)
    with c2:
        st.markdown("#### MinIO data bucket")
        _hub_show_image(_HUB_MINIO_DATA)
    st.markdown("#### MinIO image bucket")
    _hub_show_image(_HUB_MINIO_IMAGES)
    st.markdown("### Presentation")
    st.write(
        "Instead of keeping the dataset only on a local machine, we moved the data into a Dockerized MinIO object store. "
        "This MinIO container is hosted on AWS EC2 and exposed through an Elastic IP. "
        "This gives the team one shared data bank for CSV files and images, "
        "which can later be consumed by the ML pipeline and API containers."
    )


# ============================================================================
# TAB 6 — MODEL ARCHITECTURE
# ============================================================================
with tab6:
    st.header("6. Model Architecture")
    st.markdown("### Multimodal dual-encoder classifier")
    st.write(
        "The model is a multimodal dual-encoder classifier built for multilabel colour prediction of product listings. "
        "It combines text information and visual information into one joint representation before predicting colour tags."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Architecture diagram")
        _hub_show_image(_HUB_MODEL_DIAGRAM)
    with c2:
        st.markdown("### Model details")
        st.write("**Text encoder:** Japanese BERT `cl-tohoku/bert-base-japanese-v3` processes the product name and description.")
        st.write("**Vision encoder:** OpenAI CLIP Vision Transformer `ViT-B/16` processes the product image and extracts visual features.")
        st.write("**Fusion:** The text and image embeddings are concatenated into a single joint representation.")
        st.write("**Classifier:** A neural network classifier predicts which colour tags apply to the product.")
        st.success("Because both encoders are pretrained, training mainly focuses on fine-tuning the final layers and classifier head.")
    st.markdown("### Presentation")
    st.write(
        "The model uses two pretrained encoders. On the text side, Japanese BERT understands product names and descriptions. "
        "On the image side, CLIP ViT-B/16 extracts visual features from the product image. "
        "The two embeddings are concatenated and passed through a multilabel classifier to predict the final colour tags."
    )


# ============================================================================
# TAB 7 — ML PIPELINE & API
# ============================================================================
with tab7:
    st.header("7. ML Pipeline & API Integration")
    st.markdown("### End-to-end architecture")
    if _HUB_END_TO_END:
        _hub_show_image(_HUB_END_TO_END)
    else:
        st.code(
            "MinIO Databank on AWS EC2\n        |\n        v\nSQL/PostgreSQL\n        |\n        v\nML Pipeline Docker\n        |\n        v\nAPI Docker\n        |\n        v\nFrontend / Streamlit",
            language="text",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ML Pipeline Docker")
        st.write(
            "Reads images and CSVs from MinIO, structures data into SQL/PostgreSQL, "
            "preprocesses inputs, trains the model, and logs experiments."
        )
    with col2:
        st.markdown("### API Docker")
        st.write(
            "Loads the trained model and exposes prediction endpoints "
            "so other applications can request colour-tag predictions."
        )

    st.markdown("### Presentation")
    st.write(
        "The architecture is modular. The MinIO data bank stores raw objects such as images and CSV files. "
        "The SQL layer structures the data for reliable querying. "
        "The ML pipeline container uses this data for preprocessing, training, and inference. "
        "Finally, the API container serves the trained model to external applications."
    )


# ============================================================================
# TAB 8 — HANDOVER
# ============================================================================
with tab8:
    st.header("8. Handover to Next Speaker")
    st.markdown("### Next step in the project")
    st.write(
        "So far, we have focused on the data foundation: storing CSV files and images "
        "in a Dockerized MinIO data bank on AWS EC2."
    )
    st.markdown("### Pipeline evolution")
    if _HUB_PIPELINE_EVO:
        _hub_show_image(_HUB_PIPELINE_EVO)
    else:
        st.code(
            "MinIO Databank → SQL Databank → ML Pipeline Docker → API Docker (Model Serving)",
            language="text",
        )
    st.markdown("### Transition statement")
    st.info(
        "At this point, we hand over to the MLOps part: how the data is transformed into a SQL databank "
        "and how it is integrated into the ML pipeline, MLflow tracking, model registry, and API layers."
    )


# ============================================================================
# TAB 9 — MONITORING OVERVIEW
# ============================================================================
with tab9:
    _hub_hero("Monitoring", "Real-time API observability and training pipeline metrics via Prometheus and Grafana.")
    st.markdown("### Two Metric Flows")
    c1, c2 = st.columns(2)
    with c1:
        _hub_card(
            "API Metrics",
            "Emitted continuously by the running API. Every prediction and HTTP request is tracked "
            "in real time via Prometheus."
        )
    with c2:
        _hub_card(
            "Training Metrics",
            "Pushed once at the end of each Airflow pipeline run via Pushgateway. Prometheus scrapes the gateway."
        )
    st.markdown("")
    _, img_col, _ = st.columns([1, 2, 1])
    with img_col:
        _hub_show_image(_hub_fig("Monitoring-overview.png"))
    st.markdown(
        "<div style='text-align:center; color:#475569; font-size:0.95rem;'>"
        "<b>Both flows end up in Grafana</b>, provisioned automatically — no manual dashboard configuration needed.</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB 10 — API METRICS
# ============================================================================
with tab10:
    _hub_hero("API Metrics", "Continuous observability of the live API.")
    api_page = st.radio(
        "api_page",
        ["Flow Diagram", "Color Predictions", "Request Count", "Request Duration"],
        horizontal=True,
        label_visibility="collapsed",
        key="hub_api_page",
    )
    st.divider()

    if api_page == "Flow Diagram":
        _, img_col, _ = st.columns([1, 1, 1])
        with img_col:
            _hub_show_image(_hub_fig("API-metrics-flow.png"))
    elif api_page == "Color Predictions":
        _hub_metric_header("rakuten_color_predictions_total", "Counter — labeled by color", "Incremented every time a color is predicted.")
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_color_predictions_total.png"))
    elif api_page == "Request Count":
        _hub_metric_header("rakuten_requests_total", "Counter — labeled by method, endpoint, status_code", "Tracks every HTTP request.")
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_requests_total.png"))
    elif api_page == "Request Duration":
        _hub_metric_header("rakuten_request_duration_seconds", "Histogram — labeled by method, endpoint", "Records request duration in seconds.")
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_request_duration_seconds.png"))


# ============================================================================
# TAB 11 — TRAINING METRICS
# ============================================================================
with tab11:
    _hub_hero("Training Metrics", "Pushed at the end of each pipeline run.")
    train_page = st.radio(
        "train_page",
        ["Flow Diagram", "Training Run F1", "Training Duration", "Champion F1", "Champion Version"],
        horizontal=True,
        label_visibility="collapsed",
        key="hub_train_page",
    )
    st.divider()

    if train_page == "Flow Diagram":
        _, img_col, _ = st.columns([1.5, 1, 1.5])
        with img_col:
            _hub_show_image(_hub_fig("Training-metrics-flow.png"))
    elif train_page == "Training Run F1":
        _hub_metric_header("rakuten_training_run_f1", "Labeled by model_version and run_id", "F1 score per training run.")
        _, img_col, _ = st.columns([1, 14, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_training_run_f1.png"))
    elif train_page == "Training Duration":
        _hub_metric_header("rakuten_training_duration_seconds", "Labeled by model_version", "Training time per run.")
        _, img_col, _ = st.columns([1, 4, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_training_duration_seconds.png"))
    elif train_page == "Champion F1":
        _hub_metric_header("rakuten_champion_f1", "Single value", "F1 score of the currently promoted champion model.")
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_champion_f1.png"))
    elif train_page == "Champion Version":
        _hub_metric_header("rakuten_champion_version", "Single value", "Version number of the champion.")
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            _hub_show_image(_hub_fig("rakuten_champion_version.png"))


# ============================================================================
# TAB 12 — DATA DRIFT
# ============================================================================
with tab12:
    _hub_hero("Data Drift", "Detected with Evidently — comparing reference vs current distributions.")
    st.markdown("### Data Drift with Evidently")
    st.info(
        "Data drift occurs when the statistical distribution of input data changes over time "
        "- either in the input features, target labels, or both."
    )
    st.markdown("")
    st.markdown("#### Potential Causes in This Project")
    c1, c2 = st.columns(2)
    with c1:
        _hub_card(
            "Color Trends Change — Label Drift",
            "The distribution of target labels shifts over time."
        )
    with c2:
        _hub_card(
            "Photo Lighting Change — Training-Serving Skew",
            "Systematic change in product photographs."
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
    if _HUB_DRIFT_REPORT and Path(_HUB_DRIFT_REPORT).exists():
        html_content = Path(_HUB_DRIFT_REPORT).read_text(encoding="utf-8")
        _hub_components.html(html_content, height=800, scrolling=True)
    else:
        st.warning("Drift report not found at reports/data_drift_report.html. Run `python -m src.monitoring.drift` to generate it.")



# ======================================================
# TAB 1 — MLFLOW WORKFLOW
# ======================================================
with tab13:
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
with tab14:
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
with tab15:
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
with tab16:
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
with tab17:
    st.title("Rakuten Color Predictor")

    demo_mode = st.selectbox(
        "Select Prediction Mode",
        ["Single Product Upload", 
         "Predict by Product ID (from DB)",
         "Bulk Batch Processing (Excel + Images)"]
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
            # Only the text description is required. The image is optional — if
            # omitted, the API falls back to a neutral gray 224x224 placeholder.
            if not item_caption or item_caption.strip() == "":
                st.error("Please enter a product description.")
            else:
                # Save image to a shared temp path so the API container can read it,
                # or use /predict/upload if an image was uploaded, or /predict
                # (JSON) if no image is present
                with st.spinner("Analyzing..."):
                    try:
                        if uploaded_file is not None:
                            # With image: use upload endpoint
                            img_bytes = uploaded_file.getvalue()
                            files = {"image": (uploaded_file.name, img_bytes, uploaded_file.type)}
                            params = {"item_name": item_name, "item_caption": item_caption}
                            response = requests.post(
                                PREDICT_ENDPOINT,    # already /predict/upload
                                params=params,
                                files=files,
                                timeout=30,
                            )
                        else:
                            # No image: use JSON endpoint with image_path left empty
                            # API handles this gracefully via internal gray placeholder
                            response = requests.post(
                                f"{API_URL}/predict",
                                json={
                                    "item_name": item_name,
                                    "item_caption": item_caption,
                                    "image_path": None,
                                },
                                timeout=30,
                            )

                        if response.status_code == 200:
                            prediction = response.json()
                            st.divider()
                            st.subheader("Prediction Result")

                            res_col_img, res_col_details = st.columns([1, 2])

                            with res_col_img:
                                if uploaded_file is not None:
                                    st.image(uploaded_file.getvalue(), use_container_width=True)
                                else:
                                    st.info("No image — text-only prediction")

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
                        st.error(f"Could not connect to the API at {API_URL}.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")


    # --- MODE 2b: PREDICT BY PRODUCT ID ---
    elif demo_mode == "Predict by Product ID (from DB)":
        st.subheader("Predict from Database")
        st.markdown(
            "Enter a product ID — the app fetches its stored name, caption and "
            "image from the DB, displays them, then requests a prediction."
        )

        col_id, col_btn = st.columns([2, 1])

        with col_id:
            product_id = st.number_input(
                "Product ID",
                min_value=1,
                value=232,
                step=1,
                help="ID from the products table in Postgres",
            )

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # small vertical padding
            run_lookup = st.button(
                "Load & Predict",
                type="primary",
                use_container_width=True,
            )

        if run_lookup:
            # Step 1: fetch product metadata from DB via the API
            try:
                prod_resp = requests.get(
                    f"{API_URL}/products/{int(product_id)}",
                    timeout=10,
                )
            except Exception as e:
                st.error(f"DB lookup failed: {e}")
                st.stop()

            if prod_resp.status_code == 404:
                st.warning(f"Product {product_id} not found in DB.")
                st.stop()
            if prod_resp.status_code != 200:
                st.error(f"DB lookup returned {prod_resp.status_code}: {prod_resp.text}")
                st.stop()

            product = prod_resp.json()

            # Step 2: display the product info so the user can see what they're
            # running inference on
            st.divider()
            st.subheader(f"Product #{product.get('id')} (split = {product.get('split')})")

            info_col, img_col = st.columns([1, 1])

            with info_col:
                st.markdown(f"**Item name:** {product.get('item_name') or '_(empty)_'}")
                st.markdown(
                    f"**Caption:** {product.get('item_caption') or '_(empty)_'}"
                )
                if product.get("color_labels"):
                    st.markdown(
                        "**Ground-truth color labels:** "
                        + ", ".join(product["color_labels"])
                    )
                else:
                    st.markdown("**Ground-truth color labels:** _(none — test split?)_")

            with img_col:
                img_file = product.get("image_file")
                if img_file:
                    image_path_in_container = Path("/app/data/images") / img_file
                    if image_path_in_container.exists():
                        st.image(
                            str(image_path_in_container),
                            caption=img_file,
                            use_container_width=True,
                        )
                    else:
                        st.info(f"Image file not accessible: `{img_file}`")
                else:
                    st.info("No image associated with this product.")

            # Step 3: run prediction via the dedicated endpoint
            with st.spinner("Running prediction..."):
                try:
                    pred_resp = requests.get(
                        f"{API_URL}/predict/product/{int(product_id)}",
                        timeout=30,
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            if pred_resp.status_code != 200:
                st.error(
                    f"Prediction endpoint returned {pred_resp.status_code}: "
                    f"{pred_resp.text}"
                )
                st.stop()

            prediction = pred_resp.json()

            st.divider()
            st.subheader("Prediction Result")

            predicted = prediction.get("predicted_colors", [])
            all_scores = prediction.get("all_scores", [])

            if predicted:
                st.markdown("**Detected Colors & Confidence:**")
                for score_data in all_scores:
                    if score_data["color"] in predicted:
                        conf = score_data["score"]
                        st.write(f"🏷️ **{score_data['color']}**")
                        st.progress(conf, text=f"{conf:.2%} confidence")

                # Ground-truth vs. prediction comparison
                gt = set(product.get("color_labels") or [])
                pr = set(predicted)
                if gt:
                    tp = gt & pr
                    fp = pr - gt
                    fn = gt - pr
                    c1, c2, c3 = st.columns(3)
                    c1.metric("True Positives", len(tp), help=", ".join(tp) or "—")
                    c2.metric("False Positives", len(fp), help=", ".join(fp) or "—")
                    c3.metric("False Negatives", len(fn), help=", ".join(fn) or "—")
            else:
                st.warning("No colors met the confidence threshold.")

            st.markdown(
                f"<div class='small-muted'>Model: <b>{prediction.get('model_type')}</b> · "
                f"Inference: <b>{prediction.get('inference_ms'):.1f} ms</b></div>",
                unsafe_allow_html=True,
            )

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