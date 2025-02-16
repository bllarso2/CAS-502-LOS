import pandas as pd

def preprocess_data(df):
    """Clean and preprocess the DataFrame."""
    df = df.dropna()  # Remove missing values
    df["Stay (in days)"] = df["Stay (in days)"].astype(float)  # Convert column type
    return df

