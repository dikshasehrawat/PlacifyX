import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    df = pd.read_csv("data/raw/placementdata.csv")
    return df

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['StudentID'])
    
    # Encode binary categorical columns
    df['ExtracurricularActivities'] = (df['ExtracurricularActivities'] == 'Yes').astype(int)
    df['PlacementTraining'] = (df['PlacementTraining'] == 'Yes').astype(int)
    
    # Encode target column
    df['PlacementStatus'] = (df['PlacementStatus'] == 'Placed').astype(int)
    
    return df

def split_data(df):
    X = df.drop(columns=['PlacementStatus'])
    y = df['PlacementStatus']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for use in app
    joblib.dump(scaler, "models/scaler.pkl")
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Scaling data...")
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    print(f"\nTraining set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    print("\nPreprocessing complete!")