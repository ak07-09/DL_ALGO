import pandas as pd

def clean_data(df):
    if isinstance(df, str): return df
    return df.dropna().reset_index(drop=True)
