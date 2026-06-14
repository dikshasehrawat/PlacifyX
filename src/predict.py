import pandas as pd
import numpy as np
import shap
import joblib

def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def prepare_input(input_data, scaler):
    df_input = pd.DataFrame([input_data])
    df_input['ExtracurricularActivities'] = 1 if input_data['ExtracurricularActivities'] == 'Yes' else 0
    df_input['PlacementTraining'] = 1 if input_data['PlacementTraining'] == 'Yes' else 0
    
    df_scaled = scaler.transform(df_input)
    df_scaled = pd.DataFrame(df_scaled, columns=df_input.columns)
    return df_scaled

def predict_single(model, input_data, scaler):
    df_scaled = prepare_input(input_data, scaler)
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = "Placed" if prob >= 0.5 else "NotPlaced"
    return prediction, prob

def get_shap_explanation(model, input_data, X_train_scaled, scaler):
    df_scaled = prepare_input(input_data, scaler)
    
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(df_scaled)
    
    explanation = pd.DataFrame({
        'Feature': df_scaled.columns,
        'Value': pd.DataFrame([input_data])[df_scaled.columns].values[0],
        'SHAP Value': shap_values[0]
    }).sort_values('SHAP Value', ascending=False)
    
    return explanation

if __name__ == "__main__":
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
    
    model, scaler = load_model()
    prediction, probability = predict_single(model, sample_student, scaler)
    
    print(f"Prediction: {prediction}")
    print(f"Placement Probability: {probability*100:.2f}%")