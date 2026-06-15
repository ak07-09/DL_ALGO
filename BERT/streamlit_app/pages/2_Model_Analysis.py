import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import os
import subprocess
from src.utils.config import BASE_DIR, MODEL_PATH

st.title("⚙️ Model Analysis & Retraining")

epochs = st.slider("Epochs", min_value=1, max_value=100, value=10)
batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)

if st.button("🚀 Retrain Model Now"):
    with st.spinner("Training model locally (fully offline)..."):
        train_script = os.path.join(BASE_DIR, "src", "modeling", "train.py")
        result = subprocess.run(["python", "-m", "src.modeling.train"], cwd=BASE_DIR, capture_output=True, text=True)
        st.success("Model trained successfully!")
        st.code(result.stdout)
        
if os.path.exists(MODEL_PATH):
    st.success(f"Model serialized at: {MODEL_PATH}")
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.metric("Model File Size", f"{size_mb:.2f} MB")
else:
    st.warning("Model has not been trained yet.")
