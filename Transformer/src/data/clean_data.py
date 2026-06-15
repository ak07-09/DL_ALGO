import pandas as pd

def clean_data(df):
    return df.dropna().reset_index(drop=True)
