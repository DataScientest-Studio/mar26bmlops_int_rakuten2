import streamlit as st
from pathlib import Path
import runpy

# ======================================================
# RAKUTEN TEAM PRESENTATION HUB
# Location: src/streamlit/app.py
# ======================================================

st.set_page_config(
    page_title="Rakuten MLOps Presentation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

PAGES = {
    "🏁 Intro / Project Overview": BASE_DIR / "app_intro.py",
    "📊 MLflow / Training Pipeline": BASE_DIR / "app_mlflow.py",
    "📈 Monitoring / Grafana": BASE_DIR / "app_monitoring.py",
}

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🚀 Rakuten Team Demo")
st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Choose Section",
    list(PAGES.keys())
)

st.sidebar.markdown("---")
st.sidebar.caption("Unified Team Presentation")

# ======================================================
# LOAD PAGE
# ======================================================

page_file = PAGES[selected_page]

if page_file.exists():
    runpy.run_path(str(page_file), run_name="__main__")
else:
    st.warning(f"{page_file.name} not available yet.")
    st.info("This section will be added soon.")

# ======================================================
# FOOTER
# ======================================================

st.sidebar.markdown("---")
st.sidebar.caption("MLOps • MLflow • Monitoring • Deployment")