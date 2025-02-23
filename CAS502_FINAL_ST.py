#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[21]:


# Load data
data = pd.read_csv('Hospital-LOS.csv')

# Define target and features
X = data.drop(columns=['Stay (in days)', 'patientid']) # independent variables - this is our known
y = data['Stay (in days)'] # this is our independent variable - or our target variable. Python extracted from the known and put into 
                            # the unknown or what we're trying to solve

# Preprocess data
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])


# In[23]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Build pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# In[29]:


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


# In[31]:


# Convert new patient data to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Predict length of stay
predicted_stay = model.predict(new_patient_df)

print("Predicted Length of Stay (in days):", predicted_stay[0])


# In[33]:


# Load updated dataset with co-morbidities
data = pd.read_csv('Hospital_LOS_with_Comorbidities.csv')

# Define target and new feature set including co-morbidities
X = data.drop(columns=['Stay (in days)', 'patientid'])  # Target variable removed
y = data['Stay (in days)']  # Target variable

# Preprocess the data
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Update preprocessor to handle new features
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the updated model pipeline
updated_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the updated model
updated_model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# In[35]:


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


# In[37]:


# Convert new patient data to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Predict length of stay
predicted_stay = model.predict(new_patient_df)

print("Predicted Length of Stay (in days):", predicted_stay[0])


# this was the stopping point. I have not made any changes to the code, so it's still not factoring in co-morbidities. 

# In[4]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_study_length  # Example function from your model file

st.title("Length of Study Prediction Dashboard")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Make predictions
    predictions = predict_study_length(df)  # This should be a function from model.py
    df['Predictions'] = predictions
    st.write(df)

    # Visualization example
    st.line_chart(df['Predictions'])

# In[11]:




# In[ ]:




