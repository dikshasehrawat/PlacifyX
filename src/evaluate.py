import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_data, preprocess_data, split_data

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['NotPlaced','Placed'],
                yticklabels=['NotPlaced','Placed'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to images/confusion_matrix.png")

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='#2ecc71', linewidth=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], color='gray', linestyle='--')
    plt.title('ROC Curve', fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/roc_curve.png', dpi=150)
    plt.show()
    print("ROC curve saved to images/roc_curve.png")

def print_classification_report(y_test, y_pred):
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred,
                                target_names=['NotPlaced','Placed']))

if __name__ == "__main__":
    # Load model
    model = joblib.load("models/model.pkl")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    _, X_test, _, y_test = split_data(df)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)