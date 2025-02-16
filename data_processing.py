import pandas as pd

def preprocess_data(df):
    """Clean and preprocess the DataFrame."""
    df = df.dropna()  # Remove missing values
    df["Stay (in days)"] = df["Stay (in days)"].astype(float)  # Convert column type
    return df


def transform_data(df):
    """Convert categorical variables into numerical format."""
    df = df.copy()  # Ensure we're modifying a copy of the DataFrame
    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})  # Encode Gender: M → 0, F → 1
    return df
