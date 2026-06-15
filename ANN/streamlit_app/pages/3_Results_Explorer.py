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
        st.write("Input custom chemical features to predict wine quality.")
        cols = st.columns(3)
        fixed_acidity = cols[0].number_input("Fixed Acidity", value=8.3)
        volatile_acidity = cols[1].number_input("Volatile Acidity", value=0.5)
        citric_acid = cols[2].number_input("Citric Acid", value=0.3)
        residual_sugar = cols[0].number_input("Residual Sugar", value=2.5)
        chlorides = cols[1].number_input("Chlorides", value=0.09)
        free_sulfur = cols[2].number_input("Free Sulfur Dioxide", value=16.0)
        total_sulfur = cols[0].number_input("Total Sulfur Dioxide", value=47.0)
        density = cols[1].number_input("Density", value=0.997)
        ph = cols[2].number_input("pH", value=3.3)
        sulphates = cols[0].number_input("Sulphates", value=0.66)
        alcohol = cols[1].number_input("Alcohol", value=10.4)
        
        if st.button("Predict Quality"):
            input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]], dtype=np.float32)
            pred = model.predict(input_data)
            st.success(f"**Predicted Wine Quality:** {pred[0][0]:.2f}")
        
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info("No trained model found. Please go to the Model Analysis page to train it first.")
