import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/raw/placementdata.csv")
    return df

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['StudentID'])
    
    # Encode binary categorical columns
    le = LabelEncoder()
    df['ExtracurricularActivities'] = le.fit_transform(df['ExtracurricularActivities'])
    df['PlacementTraining'] = le.fit_transform(df['PlacementTraining'])
    
    # Encode target column
    df['PlacementStatus'] = le.fit_transform(df['PlacementStatus'])
    # NotPlaced = 0, Placed = 1
    
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

def save_processed_data(df):
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Processed data saved to data/processed/cleaned_data.csv")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Saving processed data...")
    save_processed_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"\nTarget distribution in training set:")
    print(y_train.value_counts())
    print("\nPreprocessing complete!")