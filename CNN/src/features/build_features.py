import pandas as pd
import numpy as np

def preprocess(data):
    # Basic preprocessing stub
    # Returns X, y
    if isinstance(data, str):
        return np.array([[1]]), np.array([1]) # dummy for text
    
    if data.shape[1] > 0:
        data = data.select_dtypes(include=[np.number])
        if data.shape[1] > 0:
            X = data.iloc[:, 1:].values
            y = data.iloc[:, 0].values
            return X, y
    return data, None
