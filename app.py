import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
MODEL_INFO_ENDPOINT = f"{API_URL}/model/info"
HEALTH_ENDPOINT = f"{API_URL}/health"

st.set_page_config(
    page_title="Rakuten MLOps Presentation",
    layout="centered"
)

st.title("Rakuten MLOps System")
st.caption("MLflow Tracking, Model Registry and Champion Model Serving")

tab1, tab2, tab3 = st.tabs(["Overview", "MLflow Pipeline", "Live Validation"])

# ======================================================
# TAB 1 — OVERVIEW
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
# TAB 2 — PIPELINE
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
# TAB 3 — LIVE VALIDATION
# ======================================================
with tab3:
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

    st.caption("MLflow UI is available at: http://localhost:5000")