import streamlit as st
import requests
import os
from PIL import Image
import io

# CONFIGURATION
API_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict/upload"
MODEL_INFO_ENDPOINT = f"{API_URL}/model/info"
HEALTH_ENDPOINT = f"{API_URL}/health"

# UI SETUP
st.set_page_config(
    page_title="Rakuten MLOps Presentation",
    layout="centered"
)

st.title("Rakuten MLOps System")
st.caption("MLflow Tracking, Model Registry, Champion Model Serving, Airflow Orchestration, and Evidently Monitoring")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", 
    "MLflow Pipeline", 
    "Airflow Orchestration", 
    "Evidently Monitoring", 
    "Live Validation",
    "Live Demo"
])

# ======================================================
# TAB 1 — OVERVIEW MLFLOW
# ======================================================
with tab1:
    st.header("Why We Introduced MLflow")

    st.write(
        """
        As the project became more complex, manual model handling was no longer enough.
        We needed a structured way to track experiments, compare model versions,
        and connect the best model directly to deployment.
        """
    )

    st.divider()

    st.subheader("What We Implemented")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Experiment Tracking")
        st.caption(
            "Training runs log parameters, metrics and artifacts in MLflow."
        )

    with col2:
        st.markdown("### Model Registry")
        st.caption(
            "Trained models are stored and versioned centrally in the registry."
        )

    with col3:
        st.markdown("### Champion Serving")
        st.caption(
            "The API loads the current champion model directly from MLflow."
        )

    st.divider()

    st.subheader("Why This Matters")
    st.markdown(
        """
        - **Reproducibility:** every run is tracked  
        - **Version Control:** models are versioned and managed centrally  
        - **Deployment Consistency:** the API serves the current champion model  
        - **Better Collaboration:** the team works from the same model lifecycle  
        """
    )

# ======================================================
# TAB 2 — PIPELINE MLFLOW
# ======================================================
with tab2:
    st.header("MLflow in Our Project Pipeline")

    st.write(
        "In our project, MLflow is part of the full MLOps workflow rather than a separate add-on."
    )

    st.divider()

    st.subheader("System Flow")

    st.info(
        "Training Script  →  MLflow Tracking  →  Model Registry  →  Compare & Promote  →  Champion Alias  →  FastAPI Serving  →  Streamlit Demo"
    )

    st.divider()

    st.subheader("What Happens Step by Step")

    st.markdown(
        """
        **1. Training**  
        The training script runs and logs metrics, parameters and artifacts.

        **2. Tracking**  
        MLflow stores each run so experiments can be compared later.

        **3. Registry**  
        The trained model is registered and versioned.

        **4. Compare & Promote**  
        New versions can be evaluated against previous ones.

        **5. Serving**  
        The API loads the current champion model from the registry.

        **6. Presentation Layer**  
        Streamlit is used to demonstrate the system and its live integration.
        """
    )

# ======================================================
# TAB 3 — AIRFLOW ORCHESTRATION
# ======================================================
with tab3:
    st.header("Why We Introduced Apache Airflow")
    st.write(
        """
        As the pipeline grows to include data ingestion, training, and deployment, 
        running scripts manually became a bottleneck. We needed an orchestrator 
        to manage dependencies and ensure the pipeline runs reliably.
        """
    )
    st.divider()
    st.subheader("What We Implemented")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### DAG Orchestration")
        st.caption("Workflow stages (Ingest -> Train -> Register) are scheduled as Directed Acyclic Graphs.")
    with col2:
        st.markdown("### Dependency Management")
        st.caption("Ensures training only starts if data ingestion is successful and the DB is ready.")
    with col3:
        st.markdown("### Automated Retries")
        st.caption("Handles failures automatically, ensuring high availability of the pipeline.")

    st.divider()
    st.subheader("Why This Matters")
    st.markdown(
        """
        - **Automation:** Moves from manual execution to scheduled, automated pipelines.  
        - **Scalability:** Handles complex workflows with multiple parallel tasks.  
        - **Visibility:** Provides a clear UI to track the status and logs of every task.  
        - **Error Handling:** Alerting and management when a step fails.  
        """
    )

# ======================================================
# TAB 4 — EVIDENTLY MONITORING
# ======================================================
with tab4:
    st.header("Why We Introduced Evidently AI")
    st.write(
        """
        Once a model is deployed, its performance can degrade over time due to data drift. 
        We use Evidently to monitor model performance and detect when 
        new production data looks different from our training data.
        """
    )
    st.divider()
    st.subheader("What We Implemented")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Data Drift Detection")
        st.caption("Compares current inference data distributions against training baselines.")
    with col2:
        st.markdown("### Model Performance")
        st.caption("Tracks metrics like Accuracy and F1-score on live production data.")
    with col3:
        st.markdown("### Data Quality Reports")
        st.caption("Generates visual dashboards and reports.")

    st.divider()
    st.subheader("Why This Matters")
    st.markdown(
        """
        - **Early Warning:** Detects data drift before it impacts the business.  
        - **Data Integrity:** Identifies missing values or unexpected schema changes in inputs.  
        - **Model Trust:** Provides transparency into how the model behaves on real-world data.  
        - **Feedback Loop:** Triggers Airflow to re-train the model if drift is detected.  
        """
    )

# ======================================================
# TAB 5 — LIVE VALIDATION MLFLOW
# ======================================================
with tab5:
    st.header("Live Validation")

    st.write(
        "This section validates that the deployed API is healthy and connected to the champion model from MLflow Registry."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Check API Health", use_container_width=True):
            try:
                response = requests.get(HEALTH_ENDPOINT, timeout=20)
                if response.status_code == 200:
                    st.success("API is healthy.")
                    st.json(response.json())
                else:
                    st.error(f"Health endpoint returned {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(str(e))

    with col2:
        if st.button("Check Model Info", use_container_width=True):
            try:
                response = requests.get(MODEL_INFO_ENDPOINT, timeout=20)
                if response.status_code == 200:
                    data = response.json()

                    st.success("Live champion model successfully validated.")
                    st.json(
                        {
                            "model_type": data.get("model_type"),
                            "model_source": data.get("model_source"),
                            "is_mock": data.get("is_mock"),
                            "device": data.get("device"),
                            "num_labels": data.get("num_labels"),
                        }
                    )
                else:
                    st.error(f"Model info endpoint returned {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(str(e))

    st.divider()

    st.subheader("Expected Validation Result")
    st.code(
        """model_type: ice_dual_encoder
model_source: mlflow_registry
is_mock: false
device: cpu"""
    )

# ======================================================
# TAB 6 — LIVE DEMO
# ======================================================
with tab1:
    st.title("🛍️ Rakuten Color Predictor")
    st.markdown("""Upload a product image and provide a title and description. 
The model will predict the primary color using the Dual-Encoder weights.""")

    st.divider()

    # LAYOUT
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Data")
        product_name = st.text_area(
            "Product Name",
            placeholder="e.g., Blue Denim Jacket"
        )
        product_description = st.text_area(
            "Product Description",
            placeholder="e.g., Classic light-wash blue jean jacket with silver buttons and chest pockets."
        )
        uploaded_file = st.file_uploader("Product Image", type=["jpg", "jpeg", "png"])

    with col2:
        st.subheader("Preview & Prediction")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.info("Upload an image to see the preview.")

    st.divider()

    # PREDICTION LOGIC
    if st.button("Predict Color", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Please upload an image first!")
        elif not product_description or product_description.strip() == "":
            st.error("Please enter a product description!")
        else:
            # Prepare binary data for the image
            img_bytes = uploaded_file.getvalue()

            # Match the API argument name: 'image'
            files = {"image": (uploaded_file.name, img_bytes, uploaded_file.type)}

            # Match the API query parameters: 'item_name' and 'item_caption'
            params = {
                "item_name": product_name,  # Short version for name
                "item_caption": product_description  # Full version for caption
            }

            with st.spinner("Analyzing image and text features..."):
                try:
                    response = requests.post(
                        PREDICT_ENDPOINT,
                        params=params,
                        files=files,
                        timeout=30
                    )

                    if response.status_code == 200:
                        prediction = response.json()

                        # Your API returns 'predicted_colors' (a list)
                        colors = prediction.get("predicted_colors", [])
                        all_scores = prediction.get("all_scores", [])

                        if colors:
                            st.success(f"### Predicted Color: **{', '.join(colors)}**")

                            # Show the top confidence score if available in all_scores
                            if all_scores:
                                top_score = all_scores[0].get("score", 0)
                                st.metric("Confidence Score", f"{top_score:.2%}")

                        else:
                            st.warning("Model processed the request but returned no specific colors.")

                    elif response.status_code == 422:
                        st.error("Validation Error (422): The API expected different fields. Check /docs.")
                        st.json(response.json())
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the API at {API_URL}. Ensure the Docker container is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")