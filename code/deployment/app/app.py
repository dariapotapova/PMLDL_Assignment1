import os

import requests
import streamlit as st

# Inside the compose network containers resolve each other by SERVICE name
# ("api"); "localhost" here would point to this container itself.
# The env var allows running the app outside Docker against a local API.
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Penguin Classifier")
st.title("Penguin Species Classifier")
st.image("img/lter_penguins.png", width=450)
st.write("Enter the penguin's characteristics — the model will predict its species.")

col1, col2 = st.columns(2)
with col1:
    bill_length = st.slider("Bill length (mm)", 30.0, 60.0, 45.0)
    bill_depth = st.slider("Bill depth (mm)", 13.0, 22.0, 17.0)
with col2:
    flipper = st.slider("Flipper length (mm)", 170.0, 235.0, 200.0)
    mass = st.slider("Body mass (g)", 2500, 6500, 4200)

island = st.selectbox("Island", ["Torgersen", "Biscoe", "Dream"])
sex = st.selectbox("Sex", ["male", "female"])

if st.button("Predict", type="primary"):
    payload = {
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper,
        "body_mass_g": float(mass),
        "island": island,
        "sex": sex,
    }
    try:
        # timeout: a hanging API must not freeze the UI forever.
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        st.success(f"Predicted species: **{result['prediction']}**")
        st.subheader("Classes' probabilities")
        for cls, prob in result["probabilities"].items():
            st.progress(prob, text=f"{cls}: {prob:.1%}")

    # Containers are recreated every 5 minutes by the pipeline, so brief
    # unavailability windows are normal — show a friendly error instead
    # of a traceback.
    except requests.RequestException as e:
        st.error(f"API is currently not available: {e}")