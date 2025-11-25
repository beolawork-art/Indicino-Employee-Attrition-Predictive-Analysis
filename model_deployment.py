joblib.dump(model, 'attrition_model.pkl')
joblib.dump(scaler, 'min_max_scaler.pkl')

print("Model and Scaler saved successfully.")

# Define the paths for the saved model and scaler
MODEL_PATH = 'attrition_model.pkl'
SCALER_PATH = 'min_max_scaler.pkl'

# Define the feature names the model expects (must be in the exact order used for training!)
# NOTE: This list needs to be exactly the 44 features that remained after dropping
# the 5 columns (Attrition, EmployeeCount, StandardHours, Over18, EmployeeNumber)
# and the 1 column we added for EDA (Age_Band).
# This list is based on the column names present in X_scaled.columns.
FEATURE_COLUMNS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Married',
    'MaritalStatus_Single'
]


def load_assets():
    """Loads the pre-trained Logistic Regression model and MinMaxScaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Assets loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        print("Error: Model or Scaler file not found. Ensure 'attrition_model.pkl' and 'min_max_scaler.pkl' are in the directory.")
        return None, None

def preprocess_new_data(new_employee_features: dict, scaler):
    """
    Converts raw feature data into the scaled format expected by the model.
    """
    # 1. Convert dictionary to DataFrame
    data_df = pd.DataFrame([new_employee_features], columns=FEATURE_COLUMNS)

    # 2. Ensure data types are correct (especially for the boolean columns)
    for col in data_df.columns:
        if data_df[col].dtype == 'object' and (data_df[col].isin([0, 1, True, False]).all()):
            data_df[col] = data_df[col].astype(np.int64)

    # 3. Apply the saved MinMaxScaler
    # The scaler expects a DataFrame with the same columns used during training
    scaled_data = scaler.transform(data_df)

    return scaled_data

def predict_attrition(new_employee_features: dict):
    """
    Makes a prediction on the probability of attrition for a single employee.

    Args:
        new_employee_features: A dictionary containing all 44 feature values
                               for the new employee.

    Returns:
        A dictionary containing the predicted class (0 or 1) and probability.
    """
    model, scaler = load_assets()
    if model is None or scaler is None:
        return {"error": "Prediction failed due to missing model assets."}

    # 1. Preprocess and Scale the data
    scaled_input = preprocess_new_data(new_employee_features, scaler)

    # 2. Predict the class (0=Stay, 1=Leave)
    prediction_class = model.predict(scaled_input)[0]

    # 3. Predict the probability
    # model.predict_proba returns [[Prob_Stay, Prob_Leave]]
    prediction_proba = model.predict_proba(scaled_input)[0][1] # Get the probability of 1 (Leaving)

    result = {
        "predicted_class": int(prediction_class),
        "probability_of_leaving": float(f"{prediction_proba:.4f}")
    }

    return result

# --- Example Usage (How to test the script) ---
if __name__ == "__main__":
    # NOTE: You MUST replace this with actual 0/1 values for ALL 44 features.
    # This example is just illustrative and will likely crash if run without
    # the exact scaled data structure or real data.

    example_employee_data = {
        'Age': 35,
        'DailyRate': 1000,
        'DistanceFromHome': 5,
        'Education': 3,
        'EnvironmentSatisfaction': 2,
        'Gender': 1, # Male
        # ... (rest of the features must be here!)
        'OverTime': 1, # Yes
        'JobRole_Sales Representative': 1, # Yes
        'JobLevel': 1,
        'TotalWorkingYears': 10,
        # ...
    }

    print("\nAttempting to make prediction for a new employee...")
    # Replace example_employee_data with a dictionary containing all 44 feature values
    # prediction = predict_attrition(example_employee_data)
    # print(prediction)
    print("Script execution finished. To run a real test, fill the 'example_employee_data' dictionary with all 44 features.")
