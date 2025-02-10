import streamlit as st

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

