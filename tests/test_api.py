from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"



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
        # --- ML Flow --- 
        "9. MLflow Workflow",
        "10. Why MLflow",
        "11. MLflow Lifecycle",
        "12. Live Validation",
        # --- MONITORING (4 tabs from colleague's app_monitoring.py) ---
        "13. Monitoring Overview",
        "14. API Metrics",
        "15. Training Metrics",
        "16. Data Drift",
        # --- OWN (your 5 tabs, unchanged content) ---
        "17. Live Demo",
    ]
)