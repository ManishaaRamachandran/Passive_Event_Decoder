import pandas as pd
import numpy as np
from typing import Tuple

def validate_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    required_columns = [
        'encoder_s1', 'encoder_s2', 'encoder_s3', 'encoder_s4',
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'max_acceleration', 'rms_acceleration', 'event_duration'
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    return True, "Validation successful"

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean())

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        'encoder_s1', 'encoder_s2', 'encoder_s3', 'encoder_s4',
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'max_acceleration', 'rms_acceleration', 'event_duration'
    ]
    
    return df[feature_columns]

def prepare_training_data(df: pd.DataFrame, target_column: str = 'predicted_class'):
    df_clean = handle_missing_values(df)
    
    is_valid, message = validate_columns(df_clean)
    if not is_valid:
        raise ValueError(message)
    
    X = extract_features(df_clean)
    y = df_clean[target_column]
    
    return X, y
