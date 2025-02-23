```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

# Explanations (for Brad)
- test_train_split is a cool feature (module) of this library (sklearn). It splits the data into training and testing--usually an 80/20 split. 
- I'm familiar with what I've done in class - L1/L2 regressions. But I use random forests on Dataiku when dealing with non-linear regressions. I pulled this code from a book I frequently use: Machine Learning with PyTorch and Scikit_Learn, which offers links to great tutorials.
- Mean Absolute Error gives us the absolute difference between actual length of stay and the predicted. This model's prediction is off by 1.24 days.
- Mean Squared Error gives us the average of the squared difference between actual and predicted. It essentially penalizes large errors because they are squared. With this method, the model is off by 3.43 days. We could improve these numbers, but it could get complicated. We could also drop some of the columns that don't matter, like doctors' names. But when I look at this data visually, it's good to see the docs' names, because it could give us some insight into which doctors have the longest LOS. I'm less concerned with accuracy than I am with building a workable model. 
- R2 tells us how well the model explains variance in the target variable. The closer to 1, the better the model explains the variance. This model's R2 is .94, which is pretty good--the model fits the data. It explains 94% of the variance, essentially.
- OneHotEncoder and StandardScaler: I'm not all that familiar with these modules inside sklearn. I pulled them from the same book I mentioned above. OneHotEncoding takes all of our categorical variables (e.g., doctors' names) and converts them into binary numbers and places each in a new category. Moreover, it ignores unknowns, like if we put a doctor's name in that it doesn't recognize--it will just ignore it. 'cat' (for category) was the label used in the book, so I stuck with it--cut and pasted, essentially.
- StandardScaler: in the same "pre-processing" vein that sklearn does for us, StandardScaler transforms number columns ('num') to binary numbers as well, where 0 is the mean and 1 is the standard deviation. Basically, when numbers/integers are scaled like this, they all contribute evenly to the model. ColumnTransformer, then, is the function we call to pre-process categorical variables and numbers. It's pretty cool, actually.
- Pipeline puts it all together for us. It pre-processes our data and then does the random forest regression on both training and test data. Very slick.


```python
# Load data
data = pd.read_csv('/Users/davecooper/Documents/ASU/CAS502/Project/Hospital LOS.csv')

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
```

- x = data.drop: stay in days and patientid. We needed to drop stay in days to make it a dependent variable--the target. When I first built the model (2 years ago), I left patientid alone, so the model used the values, which increased the errors. I could have gone back to the data and just deleted the column, but this is an easy fix.
- y = data(stay in days): this is our target--our dependent variable. 


```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- I've set the size of the test data to 20% of the data with test_size. The random state just ensures the same data split every time we run the model, or we'd get different results each time. The "42" is just arbitrary. In Python, it's also a joke. If you've ever read Hitchhiker's Guide to the Galaxy, 42 is the answer to life, the universe, and everything--LOL!  


```python
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
```

    MAE: 1.2415595039469387
    MSE: 3.4250369082537997
    R² Score: 0.9446304584950684



```python
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

```


```python
# Convert new patient data to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Predict length of stay
predicted_stay = model.predict(new_patient_df)

print("Predicted Length of Stay (in days):", predicted_stay[0])

```

    Predicted Length of Stay (in days): 8.234666666666667


- Data frames: this used to be troubling for me. The new_patient data above is a "dictionary." In a dictionary, keys (e.g., age) map to values (e.g., 31-40). Pandas takes dictionary data and transforms it into a data frame (e.g., excel spreadsheet). Sklearn requires data to be in a data frame format. If we didn't have that line of code in there, we'd get an error that tells us we have to convert the data to a data frame.
- That's it. This is a pretty simple model--no for loops! 

# here goes nothing ... taking the study Brad found and combining it with initial dataset


```python
import numpy as np

# The original dataset based on client data
data = pd.read_csv('/Users/davecooper/Documents/ASU/CAS502/Project/Hospital LOS.csv')

# Define co-morbidities and their probabilities (based on study results from Predicting Patients at Risk for Prolonged Hospital Stays)
comorbidities = {
    "Septicemia": 0.086,
    "CHF": 0.056,
    "Pneumonia": 0.051,
    "COPD_Bronchiectasis": 0.038,
    "Cardiac_Dysrhythmias": 0.035,
    "Acute_Cerebrovascular_Disease": 0.035,
    "Acute_Renal_Failure": 0.032,
    "Skin_Infections": 0.032,
    "UTI": 0.031
}

# Randomly assign co-morbidities based on probabilities
for condition, prob in comorbidities.items():
    data[condition] = np.random.choice([1, 0], size=len(data), p=[prob, 1 - prob])

# Save the updated dataset
data.to_csv('/Users/davecooper/Documents/ASU/CAS502/Project/Hospital_LOS_with_Comorbidities.csv', index=False)

# Display first few rows
print(data.head())
```

       Available Extra Rooms in Hospital    Department Ward_Facility_Code  \
    0                                  2    gynecology                  F   
    1                                  4    gynecology                  D   
    2                                  3  radiotherapy                  E   
    3                                  3  radiotherapy                  E   
    4                                  4    gynecology                  F   
    
      doctor_name  staff_available  patientid    Age  gender Type of Admission  \
    0   Dr Olivia                9      23287  41-50  Female            Trauma   
    1   Dr Sophia                0     103955  21-30  Female            Trauma   
    2      Dr Sam                4      16412  21-30    Male            Trauma   
    3      Dr Sam                2      43812   0-10    Male            Trauma   
    4   Dr Olivia                2      49799  21-30  Female            Trauma   
    
      Severity of Illness  ... Stay (in days)  Septicemia CHF  Pneumonia  \
    0            Moderate  ...              6           0   0          0   
    1            Moderate  ...              9           0   0          0   
    2               Minor  ...             16           0   1          0   
    3            Moderate  ...             23           0   0          0   
    4            Moderate  ...              7           0   0          0   
    
       COPD_Bronchiectasis  Cardiac_Dysrhythmias  Acute_Cerebrovascular_Disease  \
    0                    0                     0                              0   
    1                    0                     0                              0   
    2                    0                     0                              0   
    3                    0                     0                              0   
    4                    0                     0                              0   
    
       Acute_Renal_Failure  Skin_Infections  UTI  
    0                    0                0    0  
    1                    0                0    0  
    2                    0                0    0  
    3                    0                0    0  
    4                    0                0    0  
    
    [5 rows x 23 columns]


Ok, it worked ... with 5 tries at the for loop! 

I'm basically cutting and pasting the code I wrote above into a single cell to re-run the model and with new data


```python
# Load updated dataset with co-morbidities
data = pd.read_csv('/Users/davecooper/Documents/ASU/CAS502/Project/Hospital_LOS_with_Comorbidities.csv')

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


```

    MAE: 1.2415595039469387
    MSE: 3.4250369082537997
    R² Score: 0.9446304584950684



```python
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
```


```python
# Convert new patient data to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Predict length of stay
predicted_stay = model.predict(new_patient_df)

print("Predicted Length of Stay (in days):", predicted_stay[0])

```

    Predicted Length of Stay (in days): 8.234666666666667


Ok, so ... that did absolutely nothing--LOL! I'll figure it out. It's as if the co-morbidities didn't figure in. It could be something with the Random Forest regression. This is a first time using it. The 0s and 1s I used could be throwing the standardscaler off--not sure. It makes for a nice "issue," anyway: Why doesn't the model work as expected???? 
