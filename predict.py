import numpy as np
import joblib
from app.schemas.models import PredictionRequest, PredictionResponse
from datetime import datetime

MODEL_PATH = "./models/random_forest_model.joblib"
SCALER_PATH = "./models/scaler.joblib"

def predict_event(request: PredictionRequest) -> PredictionResponse:
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Prepare feature vector
    features = np.array([[
        request.encoder_state.encoder_s1,
        request.encoder_state.encoder_s2,
        request.encoder_state.encoder_s3,
        request.encoder_state.encoder_s4,
        request.features.acceleration_x,
        request.features.acceleration_y,
        request.features.acceleration_z,
        request.features.max_acceleration,
        request.features.rms_acceleration,
        request.features.event_duration
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get class labels
    classes = model.classes_
    
    # Create probability dictionary
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities))
    
    return PredictionResponse(
        predicted_class=prediction,
        confidence=confidence,
        probabilities=prob_dict,
        timestamp=datetime.utcnow(),
        model_used="RandomForestClassifier"
    )
