import pytest
import pandas as pd
from itertools import combinations
from CAS502_FINAL_ST import model  # Must match script's filename!

def test_model_prediction():
    # Define test patient data
    test_patient = {
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

    # Expected increase in Length of Stay (LOS) in days for co-morbidities
    los_increase = {
        "Septicemia": 7, "CHF": 5, "Pneumonia": 4, "COPD_Bronchiectasis": 3,
        "Cardiac_Dysrhythmias": 2, "Acute_Cerebrovascular_Disease": 6,
        "Acute_Renal_Failure": 5, "Skin_Infections": 2, "UTI": 1
    }

    # Generate missing interaction terms
    for combo in combinations(los_increase.keys(), 2):
        col_name = f"{combo[0]}_{combo[1]}_interaction"
        test_patient[col_name] = test_patient.get(combo[0], 0) * test_patient.get(combo[1], 0)

    # Ensure Comorbidity_LOS_Impact is included
    test_patient["Comorbidity_LOS_Impact"] = sum(
        [test_patient[condition] * los_increase[condition] for condition in los_increase]
    )

    # Convert test patient data into a DataFrame
    test_patient_df = pd.DataFrame([test_patient])

    # Ensure preprocessing is applied before prediction
    try:
        test_patient_transformed = model.named_steps['preprocessor'].transform(test_patient_df)
    except Exception as e:
        pytest.fail(f"Preprocessing failed with error: {e}")

    # Make prediction
    try:
        predicted_stay = model.predict(test_patient_df)
    except Exception as e:
        pytest.fail(f"Model prediction failed with error: {e}")

    # Validate output
    assert predicted_stay[0] > 0, "Predicted LOS should be positive"
    print(f"Predicted Length of Stay: {predicted_stay[0]}")

# Added so that pytest does not run test functions
if __name__ == "__main__":
    pytest.main()

