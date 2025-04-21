# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your model
model = load_model('models/lung_cancer_model.h5')  # Make sure path is correct

# Page config
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

# Title
st.title("🫁 Lung Cancer Detection from X-ray")
st.write("Upload an X-ray image and check if lung cancer is detected.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((150, 150))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "✅ No Cancer Detected" if prediction < 0.5 else "⚠️ Cancer Detected"
    st.markdown(f"### Result: {result}")
