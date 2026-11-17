import numpy as np
from typing import Dict

def calculate_acceleration_magnitude(x: float, y: float, z: float) -> float:
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_mean_acceleration(x: float, y: float, z: float) -> float:
    return (x + y + z) / 3

def calculate_rms(values: list) -> float:
    return np.sqrt(np.mean(np.array(values)**2))

def calculate_standard_deviation(values: list) -> float:
    return np.std(values)

def calculate_peak_to_peak(values: list) -> float:
    return np.max(values) - np.min(values)

def create_feature_vector(encoder_state: Dict, acceleration_data: Dict) -> Dict:
    x = acceleration_data.get('x', 0)
    y = acceleration_data.get('y', 0)
    z = acceleration_data.get('z', 0)
    
    magnitude = calculate_acceleration_magnitude(x, y, z)
    mean = calculate_mean_acceleration(x, y, z)
    
    return {
        'encoder_s1': encoder_state.get('s1', 0),
        'encoder_s2': encoder_state.get('s2', 0),
        'encoder_s3': encoder_state.get('s3', 0),
        'encoder_s4': encoder_state.get('s4', 0),
        'acceleration_x': x,
        'acceleration_y': y,
        'acceleration_z': z,
        'acceleration_magnitude': magnitude,
        'max_acceleration': acceleration_data.get('max', magnitude),
        'mean_acceleration': mean,
        'rms_acceleration': acceleration_data.get('rms', magnitude),
        'standard_deviation': acceleration_data.get('std', 0),
        'peak_to_peak': acceleration_data.get('peak_to_peak', 0),
        'event_duration': acceleration_data.get('duration', 0)
    }
