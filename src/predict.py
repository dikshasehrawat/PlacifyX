import pandas as pd
import numpy as np
import shap
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_data, preprocess_data, split_data

def load_model():
    model = joblib.load("models/model.pkl")
    return model

def predict_single(model, input_data):
    """
    input_data: dictionary with student details
    """
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical columns
    df_input['ExtracurricularActivities'] = 1 if input_data['ExtracurricularActivities'] == 'Yes' else 0
    df_input['PlacementTraining'] = 1 if input_data['PlacementTraining'] == 'Yes' else 0
    
    # Predict
    prob = model.predict_proba(df_input)[0][1]
    prediction = "Placed" if prob >= 0.5 else "NotPlaced"
    
    return prediction, prob

def get_shap_explanation(model, input_data, X_train):
    df_input = pd.DataFrame([input_data])
    df_input['ExtracurricularActivities'] = 1 if input_data['ExtracurricularActivities'] == 'Yes' else 0
    df_input['PlacementTraining'] = 1 if input_data['PlacementTraining'] == 'Yes' else 0
    
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(df_input)
    
    # Create explanation dataframe
    explanation = pd.DataFrame({
        'Feature': df_input.columns,
        'Value': df_input.values[0],
        'SHAP Value': shap_values[0]
    }).sort_values('SHAP Value', ascending=False)
    
    return explanation

if __name__ == "__main__":
    # Test with a sample student
    sample_student = {
        'CGPA': 8.5,
        'Internships': 2,
        'Projects': 3,
        'Workshops/Certifications': 2,
        'AptitudeTestScore': 85,
        'SoftSkillsRating': 4.5,
        'ExtracurricularActivities': 'Yes',
        'PlacementTraining': 'Yes',
        'SSC_Marks': 78,
        'HSC_Marks': 82
    }
    
    model = load_model()
    prediction, probability = predict_single(model, sample_student)
    
    print(f"Prediction: {prediction}")
    print(f"Placement Probability: {probability*100:.2f}%")
    
    # Load training data for SHAP
    df = load_data()
    df = preprocess_data(df)
    X_train, _, _, _ = split_data(df)
    
    explanation = get_shap_explanation(model, sample_student, X_train)
    print("\n=== SHAP Explanation ===")
    print(explanation)