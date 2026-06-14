import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"{name} trained successfully")
    
    return trained

def evaluate_models(trained_models, X_test, y_test):
    print("\n=== Model Evaluation ===")
    results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        results[name] = {'Accuracy': acc, 'F1 Score': f1, 'ROC-AUC': roc}
        
        print(f"\n{name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc:.4f}")
    
    return results

def save_best_model(trained_models, results):
    best_model_name = max(results, key=lambda x: results[x]['ROC-AUC'])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, "models/model.pkl")
    print(f"\nBest model: {best_model_name}")
    print(f"Saved to models/model.pkl")
    return best_model_name

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    print("\nTraining models...")
    trained_models = train_models(X_train_scaled, y_train)
    
    print("\nEvaluating models...")
    results = evaluate_models(trained_models, X_test_scaled, y_test)
    
    save_best_model(trained_models, results)