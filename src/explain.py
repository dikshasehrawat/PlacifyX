import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_data, preprocess_data, split_data

def explain_model(model, X_train, X_test):
    print("Generating SHAP explanations...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global feature importance plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("images/shap_global.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Global SHAP plot saved to images/shap_global.png")
    
    # SHAP summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot", fontweight='bold')
    plt.tight_layout()
    plt.savefig("images/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("SHAP summary plot saved to images/shap_summary.png")
    
    return explainer, shap_values

if __name__ == "__main__":
    # Load model
    model = joblib.load("models/model.pkl")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Explain
    explainer, shap_values = explain_model(model, X_train, X_test)
    
    print("\nSHAP explanation complete!")