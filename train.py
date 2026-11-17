import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from app.simulation.generator import generate_training_dataset

MODEL_PATH = "./models/random_forest_model.joblib"
SCALER_PATH = "./models/scaler.joblib"
UPLOADED_DATA_PATH = "./data/uploaded_dataset.csv"

def train_model():
    os.makedirs("./models", exist_ok=True)
    
    # Use uploaded dataset if available, otherwise generate synthetic data
    if os.path.exists(UPLOADED_DATA_PATH):
        df = pd.read_csv(UPLOADED_DATA_PATH)
        print(f"Using uploaded dataset with {len(df)} samples")
    else:
        df = generate_training_dataset(n_samples=1000)
        print(f"Using synthetic dataset with {len(df)} samples")
    
    # Prepare features and target
    feature_cols = [
        'encoder_s1', 'encoder_s2', 'encoder_s3', 'encoder_s4',
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'max_acceleration', 'rms_acceleration', 'event_duration'
    ]
    
    X = df[feature_cols]
    y = df['predicted_class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Calculate accuracy
    accuracy = model.score(X_test_scaled, y_test)
    
    return {
        "accuracy": accuracy,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "feature_count": len(feature_cols)
    }
