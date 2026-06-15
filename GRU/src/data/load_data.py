import pandas as pd
from src.utils.config import DATA_RAW

def load_raw_data():
    return pd.read_csv(DATA_RAW)
