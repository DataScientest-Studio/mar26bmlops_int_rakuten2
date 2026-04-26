import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Rakuten Multimodal Colour Extraction",
    page_icon="🎨",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .metric-box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #e6e6e6;
        background: #fafafa;
        min-height: 90px;
    }
    .metric-label {font-size: 14px; color: #666;}
    .metric-value {font-size: 28px; font-weight: 700; color: #111;}
    .section-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e8e8e8;
        background: #fbfbfb;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv_safe(path: str):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_resized_image(path: str, target_height: int = 220):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = int((target_height / h) * w)
    return img.resize((new_w, target_height))


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class='metric-box'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_existing_path(candidates):
    for path in candidates:
        p = Path(path)
        if p.exists():
            return p
    return None


@st.cache_data
def compute_basic_stats(df_x, df_y):
    stats = {}
    if df_x is not None:
        stats["train_rows"] = len(df_x)
        stats["missing_caption"] = int(df_x["item_caption"].isna().sum()) if "item_caption" in df_x.columns else None
    if df_y is not None:
        stats["label_rows"] = len(df_y)
        if "color_tags" in df_y.columns:
            stats["top_labels"] = df_y["color_tags"].astype(str).value_counts().head(10)
    return stats


# -----------------------------
# Paths
# -----------------------------
X_TRAIN_PATH = find_existing_path(["X_train.csv", "data/X_train.csv", "src/streamlit/X_train.csv"])
X_TEST_PATH = find_existing_path(["X_test.csv", "data/X_test.csv", "src/streamlit/X_test.csv"])
Y_TRAIN_PATH = find_existing_path(["y_train.csv", "data/y_train.csv", "src/streamlit/y_train.csv"])
Y_RANDOM_PATH = find_existing_path(["y_random.csv", "data/y_random.csv", "src/streamlit/y_random.csv"])

IMAGE_DIR = find_existing_path(["images", "data/images", "src/streamlit/images"])

AWS_SCREENSHOT = find_existing_path([
    "src/streamlit/assets/AWS_EC2_Screenshot.png",
    "assets/AWS_EC2_Screenshot.png",
    "AWS_EC2_Screenshot.png",
])
MINIO_DATA_SCREENSHOT = find_existing_path([
    "src/streamlit/assets/MINIO_Data.png",
    "assets/MINIO_Data.png",
    "MINIO_Data.png",
])
MINIO_IMAGES_SCREENSHOT = find_existing_path([
    "src/streamlit/assets/MINIO_images.png",
    "assets/MINIO_images.png",
    "MINIO_images.png",
])
MODEL_DIAGRAM = find_existing_path([
    "src/streamlit/assets/model-diagram.png",
    "assets/model-diagram.png",
    "model-diagram.png",
])
END_TO_END_DIAGRAM = find_existing_path([
    "src/streamlit/assets/end_to_end_architecture.png",
    "assets/end_to_end_architecture.png",
    "end_to_end_architecture.png",
])
PIPELINE_EVOLUTION_DIAGRAM = find_existing_path([
    "src/streamlit/assets/pipeline_evolution.png",
    "assets/pipeline_evolution.png",
    "pipeline_evolution.png",
])

# -----------------------------
# Data loading
# -----------------------------
X_train = load_csv_safe(str(X_TRAIN_PATH)) if X_TRAIN_PATH else None
X_test = load_csv_safe(str(X_TEST_PATH)) if X_TEST_PATH else None
y_train = load_csv_safe(str(Y_TRAIN_PATH)) if Y_TRAIN_PATH else None
y_random = load_csv_safe(str(Y_RANDOM_PATH)) if Y_RANDOM_PATH else None

stats = compute_basic_stats(X_train, y_train)
train_rows = stats.get("train_rows", "-")
test_rows = len(X_test) if X_test is not None else "-"
label_rows = stats.get("label_rows", "-")
missing_caption = stats.get("missing_caption", "-")

sample_image_names = [
    "100054_10006497_1.jpg",
    "100054_10006798_1.jpg",
    "100054_10006900_1.jpg",
    "100054_10006905_1.jpg",
    "100054_10008846_1.jpg",
]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Presentation Flow")
section = st.sidebar.radio(
    "Go to",
    [
        "1. Project Overview",
        "2. Business Problem",
        "3. Dataset & Inputs",
        "4. Sample Images",
        "5. Docker Databank on AWS",
        "6. Model Architecture",
        "7. ML Pipeline & API Integration",
        "8. Handover to Next Speaker",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("Overview → Problem → Data → Samples → AWS/MinIO → Model → Pipeline/API → Handover")

# -----------------------------
# Header
# -----------------------------
st.title("🎨 Rakuten Multimodal Colour Extraction")
st.caption("Challenge overview + sample data + Docker MinIO databank + multimodal ML model + API architecture")

# -----------------------------
# Sections
# -----------------------------
if section == "1. Project Overview":
    st.header("1. Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Training rows", str(train_rows))
    with c2:
        metric_card("Test rows", str(test_rows))
    with c3:
        metric_card("Label rows", str(label_rows))
    with c4:
        metric_card("Missing captions", str(missing_caption))

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
        "To make the solution scalable and shareable, we organized the system using Dockerized services: "
        "a MinIO data bank on AWS EC2, an ML pipeline container, and an API container."
    )

elif section == "2. Business Problem":
    st.header("2. Business Problem")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Why this matters")
        st.write(
            "In e-commerce, product metadata is often incomplete or inconsistent. Sellers may upload an image and title, "
            "but colour information may be missing, ambiguous, or not standardized."
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

elif section == "3. Dataset & Inputs":
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
    if X_train is not None:
        st.dataframe(X_train.head(5), use_container_width=True)
    else:
        st.warning("X_train.csv not found. Put it in project root, data/, or src/streamlit/.")

    st.markdown("### Label preview")
    if y_train is not None:
        st.dataframe(y_train.head(5), use_container_width=True)
    else:
        st.warning("y_train.csv not found. Put it in project root, data/, or src/streamlit/.")

    top_labels = stats.get("top_labels")
    if top_labels is not None:
        st.markdown("### Most frequent label strings")
        st.bar_chart(top_labels)

    st.markdown("### Presentation")
    st.write(
        "The dataset combines visual and textual information. Images capture appearance, while titles and descriptions provide semantic clues. "
        "The target is multi-label, meaning the model must decide independently which colour tags apply to each product."
    )

elif section == "4. Sample Images":
    st.header("4. Sample Product Images")
    st.write(
        "These sample product images show why colour extraction can be challenging. "
        "Some products are metallic, reflective, dark, or visually ambiguous."
    )

    if IMAGE_DIR:
        cols = st.columns(5)
        shown = 0
        for idx, img_name in enumerate(sample_image_names):
            img_path = IMAGE_DIR / img_name
            if img_path.exists():
                with cols[idx % 5]:
                    img = load_resized_image(str(img_path), target_height=190)
                    st.image(img, caption=img_name.replace(".jpg", ""))
                shown += 1
        if shown == 0:
            st.warning("Sample image files were not found inside the images folder.")
    else:
        st.warning("Images folder not found. Expected one of: images/, data/images/, src/streamlit/images/")

    st.markdown("### Presentation")
    st.write(
        "These examples illustrate why image-only classification can be difficult. "
        "For example, lighting, shadows, product material, and background can influence the visual colour. "
        "That is why we combine image information with text information."
    )

elif section == "5. Docker Databank on AWS":
    st.header("5. Docker Databank on AWS EC2")

    st.markdown("### What we built")
    st.write(
        "We created a Dockerized data bank using **MinIO Object Store** and deployed it on an **AWS EC2 instance**. "
        "The data bank stores both the CSV files and the product images centrally, so that team members and downstream services can access the same data source."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### AWS EC2 instance")
        if AWS_SCREENSHOT:
            st.image(str(AWS_SCREENSHOT), use_container_width=True)
        else:
            st.warning("AWS_EC2_Screenshot.png not found in assets folder.")
    with c2:
        st.markdown("#### MinIO data bucket")
        if MINIO_DATA_SCREENSHOT:
            st.image(str(MINIO_DATA_SCREENSHOT), use_container_width=True)
        else:
            st.warning("MINIO_Data.png not found in assets folder.")

    st.markdown("#### MinIO image bucket")
    if MINIO_IMAGES_SCREENSHOT:
        st.image(str(MINIO_IMAGES_SCREENSHOT), use_container_width=True)
    else:
        st.warning("MINIO_images.png not found in assets folder.")

    st.markdown("### Presentation")
    st.write(
        "Instead of keeping the dataset only on a local machine, we moved the data into a Dockerized MinIO object store. "
        "This MinIO container is hosted on AWS EC2 and exposed through an Elastic IP. "
        "This gives the team one shared data bank for CSV files and images, which can later be consumed by the ML pipeline and API containers."
    )

elif section == "6. Model Architecture":
    st.header("6. Model Architecture")

    st.markdown("### Multimodal dual-encoder classifier")
    st.write(
        "The model is a multimodal dual-encoder classifier built for multilabel colour prediction of product listings. "
        "It combines text information and visual information into one joint representation before predicting colour tags."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Architecture diagram")
        if MODEL_DIAGRAM:
            st.image(str(MODEL_DIAGRAM), use_container_width=True)
        else:
            st.warning("model-diagram.png not found. Place it in src/streamlit/assets/ or assets/.")
    with c2:
        st.markdown("### Model details")
        st.write(
            "**Text encoder:** Japanese BERT `cl-tohoku/bert-base-japanese-v3` processes the product name and description."
        )
        st.write(
            "**Vision encoder:** OpenAI CLIP Vision Transformer `ViT-B/16` processes the product image and extracts visual features such as colour, shape, and texture."
        )
        st.write(
            "**Fusion:** The text and image embeddings are concatenated into a single joint representation."
        )
        st.write(
            "**Classifier:** A neural network classifier predicts which colour tags apply to the product."
        )
        st.success(
            "Because both encoders are pretrained, training mainly focuses on fine-tuning the final layers and classifier head."
        )

    st.markdown("### Presentation")
    st.write(
        "The model uses two pretrained encoders. On the text side, Japanese BERT understands product names and descriptions. "
        "On the image side, CLIP ViT-B/16 extracts visual features from the product image. "
        "The two embeddings are concatenated and passed through a multilabel classifier to predict the final colour tags."
    )

elif section == "7. ML Pipeline & API Integration":
    st.header("7. ML Pipeline & API Integration")

    st.markdown("### End-to-end architecture")
    if END_TO_END_DIAGRAM:
        st.image(str(END_TO_END_DIAGRAM), use_container_width=True)
    else:
        st.code(
            """
MinIO Databank on AWS EC2
        |
        v
SQL Databank / PostgreSQL
        |
        v
ML Pipeline Docker
  - data loading
  - preprocessing
  - training / prediction
  - MLflow logging
        |
        v
API Docker
  - serves predictions
  - exposes model to applications
        |
        v
Frontend / Streamlit / Consumer App
            """,
            language="text",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ML Pipeline Docker")
        st.write(
            "Reads images and CSVs from MinIO, structures data into SQL/PostgreSQL, preprocesses inputs, trains the model, and logs experiments."
        )
    with col2:
        st.markdown("### API Docker")
        st.write(
            "Loads the trained model and exposes prediction endpoints so other applications can request colour-tag predictions."
        )

    st.markdown("### Presentation")
    st.write(
        "The architecture is modular. The MinIO data bank stores raw objects such as images and CSV files. "
        "The SQL layer structures the data for reliable querying. "
        "The ML pipeline container uses this data for preprocessing, training, and inference. "
        "Finally, the API container serves the trained model to external applications."
    )

elif section == "8. Handover to Next Speaker":
    st.header("8. Handover to Next Speaker")

    st.markdown("### Next step in the project")
    st.write(
        "So far, I have focused on the data foundation: storing CSV files and images in a Dockerized MinIO data bank on AWS EC2."
    )

    st.markdown("### Pipeline evolution")
    if PIPELINE_EVOLUTION_DIAGRAM:
        st.image(str(PIPELINE_EVOLUTION_DIAGRAM), use_container_width=True)
    else:
        st.code(
            """
MinIO Databank (Images + CSV)
        |
        v
SQL Databank (Structured storage)
        |
        v
ML Pipeline Docker
        |
        v
API Docker (Model Serving)
            """,
            language="text",
        )

    st.markdown("### Transition statement")
    st.info(
        "At this point, I will hand over to my colleague, who will explain in more detail how the data is transformed into a SQL databank and how it is integrated into the ML pipeline and API layers."
    )
