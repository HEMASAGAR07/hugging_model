import joblib
import os

# Define the paths relative to the current script's directory
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'svm_diabetes_model.pkl')
scaler_path = os.path.join(base_path, 'standard_scaler.pkl')

# Load the SVM model and StandardScaler
svm_model = joblib.load(model_path)
standard_scaler = joblib.load(scaler_path)

# Define a function to predict diabetes
def predict(input_features):
    # Standardize the input features
    standardized_features = standard_scaler.transform([input_features])
    # Predict using the SVM model
    prediction = svm_model.predict(standardized_features)
    return prediction[0]

# Example usage:
if __name__ == "__main__":
    input_features = [5, 116, 74, 0, 0, 25.6, 0.201, 30]
    prediction = predict(input_features)
    if prediction == 0:
        print("Person is not having diabetes")
    else:
        print("The person is having diabetes")
