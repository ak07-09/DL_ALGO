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
        st.write("Input historical climate values (temp, humidity) to predict the next day.")
        temp = st.number_input("Recent Temperature", value=25.0)
        hum = st.number_input("Recent Humidity", value=60.0)
        if st.button("Forecast Next Day"):
            input_data = np.array([[[temp, hum]]]).astype(np.float32)
            pred = model.predict(input_data)
            st.success(f"**Predicted Temp:** {pred[0][0]:.2f}°C, **Predicted Humidity:** {pred[0][1]:.2f}%")
        
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info("No trained model found. Please go to the Model Analysis page to train it first.")
