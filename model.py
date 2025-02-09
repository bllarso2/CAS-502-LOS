import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Loads hospital data from a CSV file and splits it into features (X) and target (y)."""
    df = pd.read_csv(file_path)

    # Define features and target variable
    X = df.drop(columns=['Stay (in days)', 'patientid'], errors='ignore')  # Features
    y = df['Stay (in days)'] if 'Stay (in days)' in df.columns else None  # Target

    return X, y  # ✅ FIXED: Now returns both X and y

# Function to train and save the model
def train_model(data_path="Hospital-LOS.csv"):
    """Trains a RandomForest model and saves it as a .pkl file."""
    X, y = load_data(data_path)  # ✅ Uses fixed load_data function

    # Preprocessing steps
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])
      
    # Build the pipeline
    model = Pipeline([  
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
      
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "length_of_stay_model.pkl")
    return model
    
# Function to load the trained model and make predictions
def predict_study_length(df):
    """Loads the trained model and predicts the Length of Stay for new data."""
    model = joblib.load("length_of_stay_model.pkl")
    return model.predict(df)

