# ==========================================
# ♻️ RecycleVision — Garbage Image Classifier
# ==========================================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "models/best_model.h5"        # update if filename differs
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "batteries", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]
 # update according to your dataset

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="♻️ RecycleVision",
    page_icon="♻️",
    layout="centered",
)
st.title("♻️ RecycleVision — Garbage Image Classifier")
st.write("Upload a waste image to automatically identify its category.")

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Try upgrading TensorFlow to >=2.15 if problem persists.")
        return None

model = load_model()
if model:
    st.success("✅ Model loaded successfully!")

# -----------------------------
# IMAGE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image of garbage (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Classify Image"):
        with st.spinner("Analyzing... please wait..."):
            start = time.time()
            preds = model.predict(img_array)
            end = time.time()

            top3_idx = preds[0].argsort()[-3:][::-1]
            st.subheader("Prediction Results 🧩")

            for rank, idx in enumerate(top3_idx, start=1):
                st.write(f"{rank}. **{CLASS_NAMES[idx]}** — {preds[0][idx]*100:.2f}%")

            st.caption(f"⏱ Inference time: {end - start:.3f} s")

            # Save predictions log
            os.makedirs("output", exist_ok=True)
            log_df = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "predicted_class": CLASS_NAMES[top3_idx[0]],
                "confidence": float(preds[0][top3_idx[0]])
            }])
            log_df.to_csv("output/predictions.csv",
                          mode="a", index=False,
                          header=not os.path.exists("output/predictions.csv"))
else:
    st.info("⬆️ Upload a garbage image to get started!")
