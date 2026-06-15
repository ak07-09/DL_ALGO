import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import os

st.set_page_config(page_title="GRU Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌟 GRU Deep Learning Dashboard")
