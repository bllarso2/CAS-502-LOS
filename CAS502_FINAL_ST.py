#!/usr/bin/env python
# coding: utf-8

# # Adding a source for Random Forest Regressor and combinations: Machine Learning with PyTorch and Scikit-Learn: 
# 
# Raschka, Sebastian; Liu, Yuxi (Hayden); Mirjalili, Vahid. Machine Learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python (p. ). (Function). Kindle Edition. 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[2]:


# Load hospital stay dataset
data = pd.read_csv('Hospital-LOS.csv')

# Define target variable (length of stay in days) and feature set
X = data.drop(columns=['Stay (in days)', 'patientid']) # Drop target and unnecessary identifier
y = data['Stay (in days)'] # This is the target variable (dependent variable)

# Identify categorical and numerical features of the dataset
categorical_cols = X.select_dtypes(include=['object']).columns # Categorical variables
numeric_cols = X.select_dtypes(include=['number']).columns # Numeric variables

# Preprocessing: Encode categorical variables and standardize numerical variables
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols), # One-hot encode categorical features
    ('num', StandardScaler(), numeric_cols) # Standardize numeric features
])


# In[3]:


# # Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


# Define a pipeline with preprocessing and Random Forest regression model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate model performance using standard regression metrics
y_pred_no_comorbidities = model.predict(X_test) 
print("MAE:", mean_absolute_error(y_test, y_pred_no_comorbidities))
print("MSE:", mean_squared_error(y_test, y_pred_no_comorbidities))
print("R2 Score:", r2_score(y_test, y_pred_no_comorbidities)) # R2 score to show model accuracy


# In[5]:


# Define a new patient profile for prediction
new_patient = {
    'Available Extra Rooms in Hospital': 3,
    'Department': 'gynecology',
    'Ward_Facility_Code': 'C',
    'doctor_name': 'Dr. Oliva',
    'staff_available': 15,
    'Age': '31-40',
    'gender': 'Female',
    'Type of Admission': 'Emergency',
    'Severity of Illness': 'Extreme',
    'health_conditions': 'diabetes',
    'Visitors with Patient': 2,
    'Insurance': 'yes'
}


# In[6]:


# Convert patient data into a DataFrame for model input
new_patient_df = pd.DataFrame([new_patient])

# Predict length of hospital stay for new patient
predicted_stay = model.predict(new_patient_df)

print("Predicted Length of Stay (in days):", predicted_stay[0])


# # The model above is the original model--it does not factor in co-morbidities. The model below was the first attempt to factor in co-morbidities to see how they would impact length of stay (LOS). 

# In[7]:


# This is mostly cut/paste from above, with some additions
# Load updated dataset with additional co-morbidities
data = pd.read_csv('Hospital_LOS_with_Comorbidities.csv')

# Expected increase in Length of Stay (LOS) in days for patients with specific co-morbidities
# Based on external medical research. (Caveat: single study)
los_increase = {
    "Septicemia": 7,
    "CHF": 5,
    "Pneumonia": 4,
    "COPD_Bronchiectasis": 3,
    "Cardiac_Dysrhythmias": 2,
    "Acute_Cerebrovascular_Disease": 6,
    "Acute_Renal_Failure": 5,
    "Skin_Infections": 2,
    "UTI": 1
}

# Compute the expected LOS impact based on the presence of co-morbidities
# Assumes that each condition is represented as a binary (0/1) variable in the dataset
data["Comorbidity_LOS_Impact"] = sum([data[condition] * los_increase[condition] for condition in los_increase
])

from itertools import combinations # Pulled directly from source. Trying to force the model to recognize interaction effects
                                    # of co-morbitities. 

for combo in combinations(los_increase.keys(), 2):
    col_name = f"{combo[0]}_{combo[1]}_interaction"
    data[col_name] = data[combo[0]] * data[combo[1]]  # Multiply binary variables


# Scaling up Comorbidity_LOS_Impact. This just doesn't seem to be important to the model.
data["Comorbidity_LOS_Impact"] *= 10  # Trying 10x increase (can be ajusted as needed)

# Define features (X) and target variable (y)
# 'Stay (in days)' is the prediction target
# 'patientid' is dropped as it is a unique identifier with no predictive value

X = data.drop(columns=['Stay (in days)', 'patientid'])  # Target variable removed
y = data['Stay (in days)']  # Target variable

# Identify categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Preprocessing pipeline:
# - OneHotEncoder for categorical variables (handles unseen categories)
# - StandardScaler for numerical features to normalize values (Note: this could make co-morbidities too small
# which is why the model is not accounting for them)

# Update preprocessor to handle new features
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model pipeline:
# - Preprocesses data using OneHotEncoder (categorical) and StandardScaler (numerical)
# - Uses RandomForestRegressor, chosen for its ability to handle non-linear relationships and interactions

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the updated model
model.fit(X_train, y_train)

# Evaluate the trained model using the test set
y_pred_with_comorbidities = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred_with_comorbidities))
print("MSE:", mean_squared_error(y_test, y_pred_with_comorbidities))
print("R2 Score:", r2_score(y_test, y_pred_with_comorbidities))


# In[8]:


# Adding a "severity multiplier" from severity of illness to force the model to account for co-morbities. 
# Define severity multipliers
severity_multiplier = {"Minor": 1.0, "Moderate": 1.5, "Extreme": 2.0} # These numbers can be adjusted

# Define a new patient profile with additional co-morbidities
new_patient = {
    'Available Extra Rooms in Hospital': 3,
    'Department': 'gynecology',
    'Ward_Facility_Code': 'C',
    'doctor_name': 'Dr. Oliva',
    'staff_available': 15,
    'Age': '31-40',
    'gender': 'Female',
    'Type of Admission': 'Emergency',
    'Severity of Illness': 'Extreme',
    'health_conditions': 'diabetes',
    'Visitors with Patient': 2,
    'Insurance': 'yes',
    'Septicemia': 0,
    'CHF': 0,
    'Pneumonia': 1,
    'COPD_Bronchiectasis': 0,
    'Cardiac_Dysrhythmias': 0,
    'Acute_Cerebrovascular_Disease': 1,
    'Acute_Renal_Failure': 0,
    'Skin_Infections': 0,
    'UTI': 1

}


# # For some reason, the model is not factoring in co-morbidities. Maybe the correlations are not strong enough. I don't have the coding skill right now to figure out an elegant programming solution to this. So I'm going with brute force: I'm summing the co-morbidities and adding a penalty for multiple co-morbidities. 

# In[10]:


# Count the total number of co-morbidities present in the new patient
# Assumes conditions are stored as binary indicators (0 = absent, 1 = present)
new_patient["Total_CoMorbidities"] = sum([new_patient[condition] for condition in los_increase])

# Apply a penalty based on the total number of co-morbidities
# Each additional co-morbidity beyond the first increases LOS by 10%
# Ensures a minimum multiplier of 1 (i.e., no penalty if only one or zero conditions)
comorbidity_penalty = 1 + (3.89 * max(0, new_patient["Total_CoMorbidities"] - 1)) # I averaged LOS_increase (3.89) and used that as the
                                                                                    # penalty

# Calculate the baseline LOS impact based on the presence of co-morbidities
# Multiplies each condition's LOS increase by its presence (binary 0/1)
# Applies the comorbidity penalty to adjust for multiple conditions
new_patient["Comorbidity_LOS_Impact"] = sum([new_patient[condition] * los_increase[condition] for condition in los_increase]) * comorbidity_penalty

# Adjust the LOS impact based on the severity of illness
# Assumes `Severity of Illness` is a valid key in `severity_multiplier`

new_patient["Comorbidity_LOS_Impact"] *= severity_multiplier[new_patient["Severity of Illness"]]

# Ensuring all interaction terms exist in the new patient data
for combo in combinations(los_increase.keys(), 2):
    col_name = f"{combo[0]}_{combo[1]}_interaction"
    new_patient[col_name] = new_patient.get(combo[0], 0) * new_patient.get(combo[1], 0)

# Convert the new patient’s data into a DataFrame
# Required because the model expects tabular input, not a dictionary
new_patient_df = pd.DataFrame([new_patient])

# Predict length of stay
predicted_stay = model.predict(new_patient_df)

print("Predicted Length of Stay (in days):", predicted_stay[0])


# # Granted, there is a long way to go on this model. We went with a Random Forest Regression model--maybe that is the problem. While it's useful for non-linear relationships, maybe in standardizing numerical values, it's reducing them to zero or close to it. I don't know. I do know that the average cost of a day in a US hospital is approx $3000. A model that is able to better predict length of stay, even slightly better (principle of non-linearity), could have significant impacts:
# - Reduce costs by millions annually
# - Improve patient throughput, especially in high-demand hospitals
# - Decrease readmissions by ensuring adequate in-hospital care
# - Enhance operational efficiency through better staffing and resource allocation
# 

# In[34]:


# Streamlit-based dashboard for interactive hospital stay predictions
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_study_length  # Example function from your model file

# Set up dashboard title
st.title("Length of Study Prediction Dashboard")

# File uploader for users to input CSV files
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Generate predictions for uploaded dataset
    predictions = predict_study_length(df)  # This should be a function from model.py
    df['Predictions'] = predictions
    st.write(df)

    # Display predictions using line chart
    st.line_chart(df['Predictions'])


# In[ ]:




