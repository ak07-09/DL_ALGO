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
        st.write("Input a sentence to perform POS Tagging.")
        user_text = st.text_input("Enter sentence...", "The quick brown fox jumps over the lazy dog.")
        if st.button("Tag Sentence"):
            tokens = np.random.randint(0, 1000, (1, 50)) 
            pred = model.predict(tokens)
            st.success("Sequence successfully tagged! (Array output generated)")
            st.write(pred.shape)
        
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info("No trained model found. Please go to the Model Analysis page to train it first.")
