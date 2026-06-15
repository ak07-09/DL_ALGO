import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import os
from src.utils.config import DATA_RAW

st.title("📊 Dataset Explorer")
st.write("View the raw data loaded directly from the offline `data/` directory.")

if os.path.exists(DATA_RAW):
    if DATA_RAW.endswith('.csv'):
        df = pd.read_csv(DATA_RAW)
        st.dataframe(df.head(100), use_container_width=True)
        st.write(f"**Shape:** {df.shape}")
    else:
        with open(DATA_RAW, 'r') as f:
            st.text(f.read()[:1000])
else:
    st.error("Dataset not found!")
