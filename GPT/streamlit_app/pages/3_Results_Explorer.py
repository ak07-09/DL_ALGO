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
        st.write("Provide a code prompt to auto-complete.")
        user_text = st.text_area("Enter Python code...", "def add_numbers(a, b):\n")
        if st.button("Generate Code"):
            tokens = np.random.randint(0, 100, (1, 20)) 
            pred = model.predict(tokens)
            st.success("**Generated Continuation:**\n    return a + b (Dummy continuation)")
        
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info("No trained model found. Please go to the Model Analysis page to train it first.")
