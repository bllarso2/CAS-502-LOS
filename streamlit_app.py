import streamlit as st
import pandas as pd

def load_data(file_path=None):
    """Loads a CSV file into a Pandas DataFrame and renames columns if needed."""
    file_path = file_path or "Hospital-LOS.csv"
    df = pd.read_csv(file_path)

    # Rename "Stay (in days)" column to "LOS" for consistency
    if "Stay (in days)" in df.columns:
        df.rename(columns={"Stay (in days)": "LOS"}, inplace=True)

    return df

def main():
    """Runs the Streamlit app."""
    # Title
    st.title("Hospital Length of Stay Prediction")

    # Header
    st.header("Welcome to the Hospital LOS Dashboard")

    # Simple input
    patient_age = st.number_input("Enter Patient Age", min_value=0, max_value=120, step=1)
    hospital_days = st.slider("Estimated Days in Hospital", 1, 30, 5)

    # Display results
    st.write(f"Patient Age: {patient_age}")
    st.write(f"Estimated Length of Stay: {hospital_days} days")

    # Footer
    st.text("Powered by Streamlit")

# Ensures Streamlit runs only when the script is executed directly
if __name__ == "__main__":
    main()

