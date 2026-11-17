import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from app.simulation.generator import generate_training_dataset

MODEL_PATH = "./models/random_forest_model.joblib"
SCALER_PATH = "./models/scaler.joblib"

def evaluate_model():
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Generate test data
    df = generate_training_dataset(n_samples=200)
    
    feature_cols = [
        'encoder_s1', 'encoder_s2', 'encoder_s3', 'encoder_s4',
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'max_acceleration', 'rms_acceleration', 'event_duration'
    ]
    
    X = df[feature_cols]
    y = df['predicted_class']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred).tolist()
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    
    # Class distribution
    class_dist = df['predicted_class'].value_counts().to_dict()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "training_samples": 0,
        "testing_samples": len(df),
        "class_distribution": class_dist
    }
