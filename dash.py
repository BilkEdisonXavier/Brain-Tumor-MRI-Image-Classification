# 🧠 Brain Tumor Classification Streamlit App

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

# 🔧 CONFIGURATION

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

MODEL_PATH = "D:/Projects\Project-6-Brain Tumor MRI Image Classification\models_outputs/InceptionV3_best.pkl"  # Update with your model path
IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Adjust if needed

# 🧩 LOAD TRAINED MODEL

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Model load failed: {e}")
    st.stop()

# 🖼️ APP TITLE

st.title("🧠 Brain Tumor Classification")
st.write("Upload a **brain MRI image** below to classify the tumor type.")

# 📤 IMAGE UPLOAD SECTION

uploaded_file = st.file_uploader("Upload an MRI Image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🩺 Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_resized = tf.image.resize(img_array, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)  # add batch dimension

    # Predict
    preds = model.predict(img_expanded)
    pred_class = np.argmax(preds, axis=1)[0]
    pred_label = CLASS_NAMES[pred_class]
    confidence = preds[0][pred_class]

    # 🎯 DISPLAY PREDICTION RESULTS

    st.subheader("🧾 Prediction Results")
    st.success(f"**Predicted Tumor Type:** {pred_label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Show confidence for all classes
    st.markdown("### 🔍 Model Confidence per Class:")
    conf_data = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(conf_data)

else:
    st.info("👆 Please upload a brain MRI image to begin prediction.")

# 📚 ABOUT SECTION

with st.expander("ℹ️ About this App"):
    st.write("""
        This application uses a **deep learning model (EfficientNetB0)** trained on brain MRI images 
        to classify the tumor type into one of four categories:
        - Glioma
        - Meningioma
        - No Tumor
        - Pituitary Tumor
        
        **Developed by:** Edison Xavier  
        **Framework:** TensorFlow + Streamlit  
        **Model File:** EfficientNetB0_best.pkl  
    """)
