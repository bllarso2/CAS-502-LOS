import streamlit as st
import pandas as pd

def main():
    st.title("Hospital Length of Stay Prediction")
    st.subheader("Welcome to the Hospital LOS Dashboard")

    # 1. Read the CSV file
    df = pd.read_csv("Hospital_LOS_with_Comorbidities.csv")
    
    # 2. Create dropdowns for each categorical column
    #    We'll use .unique() to get unique values in each column
    department_list = df["Department"].unique()
    age_list = df["Age"].unique()
    gender_list = df["gender"].unique()
    admission_list = df["Type of Admission"].unique()
    severity_list = df["Severity of Illness"].unique()
    condition_list = df["health_conditions"].unique()
    
    selected_department = st.selectbox("Select Department", department_list)
    selected_age = st.selectbox("Select Age Range", age_list)
    selected_gender = st.selectbox("Select Gender", gender_list)
    selected_admission = st.selectbox("Select Type of Admission", admission_list)
    selected_severity = st.selectbox("Select Severity of Illness", severity_list)
    selected_condition = st.selectbox("Select Condition", condition_list)
    
    # 3. Filter the DataFrame based on user selections
    #    This will create a subset of rows matching all the chosen values
    filtered_df = df[
        (df["Department"] == selected_department) &
        (df["Age"] == selected_age) &
        (df["Gender"] == selected_gender) &
        (df["Type of Admission"] == selected_admission) &
        (df["Severity of Illness"] == selected_severity) &
        (df["Condition"] == selected_condition)
    ]
    
    # 4. Calculate the average stay for the filtered subset
    #    If no rows match, the mean will be NaN, so we handle that case
    if not filtered_df.empty:
        average_stay = filtered_df["Stay (in Days)"].mean()
        st.write(f"### Average Stay (in Days): {average_stay:.2f}")
    else:
        st.write("### No matching records found.")
    
    st.write("Powered by Streamlit")

if __name__ == "__main__":
    main()
