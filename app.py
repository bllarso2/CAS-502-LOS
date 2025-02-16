import streamlit as st
import pandas as pd
import model

st.title("Hospital Length of Stay Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.write(df.head())

    # Ensure data matches expected format
    X, _ = model.load_data("Hospital-LOS.csv")  # Use this to get expected feature format
    df = df[X.columns]  # Match columns

    # Make predictions
    predictions = model.predict_study_length(df)
    df['Predicted Stay (days)'] = predictions

    st.write("### Predictions:")
    st.write(df)

    # Visualization
    st.line_chart(df['Predicted Stay (days)'])
