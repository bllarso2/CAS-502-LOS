User Documentation: Computational Modeling System
Overview
This documentation provides a comprehensive guide to the computational modeling system developed for predicting hospital length of stay (LOS). The system consists of:
1.	Python Model – A machine learning-based approach using Random Forest Regression
2.	NetLogo Model – An agent-based simulation representing patient LOS with comorbidities
3.	Streamlit Dashboard – A user-friendly interface to interact with the Python model

1. Python Model
1.1 Description
The Python model is a Random Forest Regression model designed to predict hospital length of stay (LOS) based on various patient attributes. The second half of the model is an extension that facors in comorbidities and their impact on LOS.
1.2 Features
•	Uses OneHotEncoder and StandardScaler for preprocessing
•	Implements Random Forest Regression for predicting LOS
•	Accounts for comorbidities and their interactions
•	Applies a severity multiplier for illness severity
•	Outputs MAE, MSE, and R2 Score for performance evaluation
•	Supports new patient predictions

1.3 How to Use
1.3.1 Installation
Ensure you have Python and the required dependencies installed:
pip install pandas scikit-learn
1.3.2 Running the Model
1.	Load the dataset: 
2.	data = pd.read_csv('Hospital-LOS.csv')
3.	Preprocess the data: 
4.	X = data.drop(columns=['Stay (in days)', 'patientid'])
5.	y = data['Stay (in days)']
6.	Train the model: 
7.	model.fit(X_train, y_train)
8.	Make predictions: 
9.	predicted_stay = model.predict(new_patient_df)
1.3.3 Evaluating Performance
After training, the model outputs:
MAE: 1.23
MSE: 3.38
R² Score: 0.94
These metrics indicate how well the model predicts LOS.
1.3.4 Predicting for a New Patient
A new patient’s LOS can be predicted using the following:
new_patient = pd.DataFrame([{...patient data...}])
predicted_stay = model.predict(new_patient)
print("Predicted Length of Stay (in days):", predicted_stay[0])

2. NetLogo Model
2.1 Description
The NetLogo model is an agent-based simulation that represents hospital patients with comorbidities and their impact on LOS.
2.2 Features
•	Each patient (turtle) has an assigned LOS based on comorbiditie
•	Comorbidities influence LOS via a quadratic regression formula
•	Global settings allow for user-defined comorbidity adjustments
•	Real-time histograms visualize LOS distribution
2.3 How to Use
2.3.1 Setup
1.	Open the NetLogo model (version 6.4).
2.	Click Setup to initialize 100 patients.
3.	Click Go to simulate patient LOS over time.
2.3.2 Adjusting Comorbidities
Modify the global variables (septicemia?, chf?, etc.) to toggle comorbidities on or off for all patients.
2.3.3 Understanding Outputs
•	avg-los: Average LOS of all patients
•	Histogram: Displays the LOS distribution
2.3.4 Modifying the Model
To change the LOS formula, edit the compute-los function:
to-report compute-los [n]
  let base-los 5
  let penalty (comorbidity-penalty * n)
  let random-variation random-float 5
  let los base-los + penalty + (0.5 * n ^ 2) + random-variation
  report los
end

3. Streamlit Dashboard
3.1 Description
The Streamlit dashboard provides an interactive interface to:
•	Input patient details
•	Run the Python model for predictions
•	Display the predicted LOS in real time
3.2 How to Use
3.2.1 Installation
Ensure Streamlit is installed:
pip install streamlit
3.2.2 Running the Dashboard
1.	Navigate to the directory with the Streamlit script.
2.	Run: 
3.	streamlit run dashboard.py
4.	Enter patient details in the UI
5.	View the predicted LO
3.3 Features
•	User-friendly interface
•	Real-time prediction updates
•	Visualizes patient risk factors

4. Summary
This computational modeling system provides an end-to-end solution for predicting hospital LOS:
•	The Python Model offers a machine-learning approach
•	The NetLogo Model simulates agent-based patient LOS
•	The Streamlit Dashboard makes predictions accessible via a UI
For further improvements, consider:
•	Enhancing feature engineering in the Python model
•	Increasing patient population in the NetLogo model
•	Improving UI elements in Streamlit for better visualization
•	Using a different regression model to improve predictions
Next Steps
•	Integrate real-world patient data
•	Integrate real-world data on specific co-morbidities and the impact to LOS (a statistical approach)
•	Test the models with larger datasets
•	Improve model explainability for medical professionals
•	Expand co-morbidities

