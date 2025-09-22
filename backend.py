import pandas as pd
import numpy as np
import threading
import time
from models import predictors

# Load and filter data to numeric columns
eeg_df_raw = pd.read_csv('data/EEG.csv', low_memory=False, on_bad_lines='skip')
gsr_df_raw = pd.read_csv('data/GSR.csv', low_memory=False, on_bad_lines='skip')
eye_df_raw = pd.read_csv('data/EYE.csv', low_memory=False, on_bad_lines='skip')
facial_df_raw = pd.read_csv('data/TIVA.csv', low_memory=False, on_bad_lines='skip')

eeg_df = eeg_df_raw.select_dtypes(include=[np.number])
gsr_df = gsr_df_raw.select_dtypes(include=[np.number])
eye_df = eye_df_raw.select_dtypes(include=[np.number])
facial_df = facial_df_raw.select_dtypes(include=[np.number])

num_rows = min(len(eeg_df), len(gsr_df), len(eye_df), len(facial_df))

def extract_features(row):
    vals = row.values.astype(float)
    return np.array([np.mean(vals), np.std(vals)])

shared_data = {
    'workload_level': 'low',
    'workload_value': 0,
    'accuracy': 0.0,
    'emotion': 'neutral',
    'time_idx': 0,
    'emotion_history': []
}

def data_stream_simulator():
    last_emotion = 'neutral'
    shared_data['emotion_history'].append((0, None, last_emotion))
    for i in range(1, num_rows):
        eeg_features = extract_features(eeg_df.iloc[i])
        gsr_features = extract_features(gsr_df.iloc[i])
        eye_features = extract_features(eye_df.iloc[i])
        facial_features = extract_features(facial_df.iloc[i])

        multimodal_features = np.concatenate([eeg_features, gsr_features, eye_features, facial_features])

        workload = predictors.predict_workload(eeg_features)
        accuracy = predictors.predict_accuracy(multimodal_features)
        emotion = predictors.predict_emotion(facial_features, eeg_features)

        if emotion != last_emotion:
            shared_data['emotion_history'].append((i, last_emotion, emotion))
        else:
            shared_data['emotion_history'].append((i, emotion, emotion))
        last_emotion = emotion

        workload_map = {'low': 30, 'medium': 60, 'high': 90}
        workload_val = workload_map.get(workload, 0)

        shared_data.update({
            'workload_level': workload,
            'workload_value': workload_val,
            'accuracy': accuracy,
            'emotion': emotion,
            'time_idx': i
        })

        time.sleep(1)

threading.Thread(target=data_stream_simulator, daemon=True).start()
