import streamlit as st
import requests
import os
from PIL import Image
import io

# CONFIGURATION
API_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict/upload"

# UI SETUP
st.set_page_config(page_title="Rakuten Color Predictor")

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
            "item_name": product_name, # Short version for name
            "item_caption": product_description    # Full version for caption
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
                st.error(f"Could not connect to the API at `{API_URL}`. Ensure the Docker container is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
