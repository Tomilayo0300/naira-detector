import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Naira Fake vs Genuine Detector",
    layout="centered"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "naira_fake_detector_google_drive_only.h5"
    )

model = load_model()

# ----------------------------
# UI HEADER
# ----------------------------
st.title("💵 Naira Fake vs Genuine Detector")
st.markdown(
    "Upload an image of a Nigerian Naira note to determine whether it is **FAKE** or **GENUINE**."
)

uploaded_file = st.file_uploader(
    "📷 Upload Naira note image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# PREDICTION
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    if prob >= 0.75:
        label = "✅ GENUINE"
        confidence = prob
        color = "green"
    elif prob <= 0.25:
        label = "❌ FAKE"
        confidence = 1 - prob
        color = "red"
    else:
        label = "⚠️ UNCERTAIN"
        confidence = abs(prob - 0.5)
        color = "orange"

    st.markdown(
        "### 🔍 Result" +
        f"<br>**Prediction:** <span style='color:{color}; font-size:22px'>{label}</span>" +
        f"<br>**Confidence:** `{confidence:.2f}`",
        unsafe_allow_html=True,
    )

    st.caption(
        "⚠️ This tool assists detection. Manual verification is recommended for uncertain cases."
    )
