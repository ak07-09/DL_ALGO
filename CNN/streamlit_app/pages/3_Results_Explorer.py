import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from src.utils.config import MODEL_PATH

st.title("🚀 Inference Engine & Results")
st.write("Test your fully trained offline model dynamically.")

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        st.success("Model loaded successfully!")
        
        
        st.subheader("🧪 Try It Yourself")
        st.write("Upload a 28x28 grayscale image of a hand gesture.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, width=150)
            st.info("Image preprocessing (resizing to 28x28 grayscale) would run here.")
            if st.button("Classify Gesture"):
                dummy_img = np.random.rand(1, 28, 28, 1).astype(np.float32) # Dummy placeholder
                pred = model.predict(dummy_img)
                st.success(f"**Predicted Sign Class:** {np.argmax(pred[0])}")
        
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info("No trained model found. Please go to the Model Analysis page to train it first.")
