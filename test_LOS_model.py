#!/usr/bin/env python
# coding: utf-8

# # I originally tried to do this in the shell, but it just wasn't working for me. I took the format for "definitions" from Real Python and used them here in my code, since it's the same format. I also used https://docs.pytest.org/ as a source--particularly for @pytest.fixture function. I'm using our second set of data as a "fixture," so it will be called into my tests. 

# In[6]:


import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# load, preprocess, split, train, as in the original 
@pytest.fixture # from docs.pytest.org
def test_data():
    """Fixture to load and preprocess sample dataset"""
    data = pd.read_csv('Hospital_LOS_with_Comorbidities.csv')
    X = data.drop(columns=['Stay (in days)', 'patientid'])
    y = data['Stay (in days)']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    model.fit(X_train, y_train)

    return model, preprocessor, X_test, y_test # cuts down on code, obviously, but I'm using "return" in case we change the model.
                                                # Right now, it's not working as planned. Not sure if it's because I'm using 
                                                # Random Forest regression or not. I'll figure it out later. 

# ---- Test Cases. KISS ... making sure model returns a number not a string, gives a reasonable range, and is consistent ----

def test_output_type(test_data): # From Real Python: every argument has to start with "test." Pytest picks up on that term.
    """Test if the model returns a numerical output"""
    model, preprocessor, X_test, y_test = test_data # Same as original basically: Random Forest, columntransformer, test data, target
    sample_input = X_test.iloc[:1, :] # I selected first row, all columns to create a dataframe 
    prediction = model.predict(sample_input) # Predict LOS from the dataframe
    assert isinstance(prediction[0], (int, float, np.number)), "Model should return a number" # docs.pytest.org recommended assert function
            # to check if conditions are true. I'm using just the 1st element of array.
            # If this returns a string, the test should fail. 
    

def test_output_range(test_data):
    """Test if model predictions are within a reasonable range"""
    model, preprocessor, X_test, y_test = test_data 
    sample_input = X_test.iloc[:10, :]
    predictions = model.predict(sample_input) 
    assert all(0 <= p <= 100 for p in predictions), "Predicted LOS should be within a reasonable range" # I picked 0-100 arbitrarily
    

def test_output_consistency(test_data):
    """Test if values for the same input are consistent"""
    model, preprocessor, X_test, y_test = test_data
    sample_input = X_test.iloc[:1, :]
    pred1 = model.predict(sample_input) # this should run the exact same number twice
    pred2 = model.predict(sample_input)
    assert np.isclose(pred1, pred2, atol=1e-6).all(), "Predictions should be consistent for the same input"
            # When I use the isclose function in Numpy, I use atol=1e-6. I can't really say why--it popped up somewhere over the last
            # 3 years of CAS, so I continue to use it. This program probably doesn't need to be that exact, but it hasn't steered me 
            # wrong thus far:) 
    


# In[ ]:




