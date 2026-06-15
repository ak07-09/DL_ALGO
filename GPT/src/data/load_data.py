import os
from src.utils.config import DATA_RAW
def load_raw_data():
    with open(DATA_RAW, "r") as f:
        return f.read()
