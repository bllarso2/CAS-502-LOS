import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Caching data loading so that it doesn't reload on every interaction.
@st.cache_data
def load_data(file_path="Hospital-LOS.csv"):
    df = pd.read_csv(file_path)
    # Rename the target column if necessary for consistency
    if "Stay (in days)" in df.columns:
        df.rename(columns={"Stay (in days)": "LOS"}, inplace=True)
    return df

# Cache the model training so that the model is trained only once.
@st.cache_resource
def train_model(df):
    # Define features and target
    # We drop "LOS" (target) and "patientid" (assuming it's an identifier) from features.
    X = df.drop(columns=["LOS", "patientid"])
    y = df["LOS"]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["number"]).columns

    # Create preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    # Split data into training and test sets (here test set is used only for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the pipeline with preprocessing and the regressor
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return pipeline, mae, mse, r2, X  # also return X (features) to build the user input form

def main():
    st.title("Hospital Length of Stay Prediction Dashboard")
    st.header("Predict LOS using a trained model")

    # Load data
    df = load_data()
    
    # Optionally display the raw data
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # Train the model (this happens only once thanks to caching)
    with st.spinner("Training model..."):
        pipeline, mae, mse, r2, X_features = train_model(df)
    st.success("Model trained successfully!")
    
    # Display model evaluation metrics
    st.subheader("Model Evaluation Metrics")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

    st.subheader("Enter Patient Details for LOS Prediction")
    user_input = {}
    # Build a dynamic form based on the features in the dataset
    for col in X_features.columns:
        if X_features[col].dtype == "object":
            # For categorical variables, provide a selectbox with unique values.
            unique_values = list(X_features[col].unique())
            user_input[col] = st.selectbox(f"{col}", unique_values)
        else:
            # For numeric features, set sensible min, max, and default (mean) values.
            min_val = int(X_features[col].min())
            max_val = int(X_features[col].max())
            mean_val = int(X_features[col].mean())
            user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

    if st.button("Predict LOS"):
        # Convert user input to a DataFrame so that it matches the pipeline's expected format.
        input_df = pd.DataFrame([user_input])
        prediction = pipeline.predict(input_df)
        st.success(f"Predicted Length of Stay: {prediction[0]:.2f} days")
    
    st.text("Powered by Streamlit")

if __name__ == "__main__":
    main()

