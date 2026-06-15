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
        st.write("Input a recent historical sequence to predict the next value.")
        seq_input = st.text_input("Enter comma-separated numerical sequence (e.g. 50, 45, 60, 55)", "50, 45, 60")
        if st.button("Forecast Next Value"):
            try:
                vals = [float(x.strip()) for x in seq_input.split(",")]
                input_data = np.array([vals]).reshape(1, 1, len(vals)).astype(np.float32)
                pred = model.predict(input_data)
                st.success(f"**Forecasted Value:** {pred[0][0]:.2f}")
            except Exception as e:
                st.error("Invalid input format.")
        
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info("No trained model found. Please go to the Model Analysis page to train it first.")
