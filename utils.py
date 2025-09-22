# app/utils.py
import joblib
import numpy as np
from pathlib import Path

# Get absolute path of the models directory relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODELS_DIR = BASE_DIR / "models"

workload_model_path = MODELS_DIR / "workload_model.pkl"
accuracy_model_path = MODELS_DIR / "accuracy_model.pkl"
emotion_model_path = MODELS_DIR / "emotion_model.pkl"

workload_model, workload_le = joblib.load(workload_model_path)
accuracy_model = joblib.load(accuracy_model_path)
emotion_model, emotion_le = joblib.load(emotion_model_path)

def predict_workload(eeg_features):
    eeg_features = np.array(eeg_features).reshape(1, -1)
    pred_class = workload_model.predict(eeg_features)[0]
    pred_label = workload_le.inverse_transform([pred_class])[0]
    return pred_label

def predict_accuracy(multimodal_features):
    multimodal_features = np.array(multimodal_features).reshape(1, -1)
    proba = accuracy_model.predict_proba(multimodal_features)[0, 1]
    return proba

def predict_emotion(facial_features, eeg_features):
    combined = np.array(facial_features + eeg_features).reshape(1, -1)
    pred_class = emotion_model.predict(combined)[0]
    pred_label = emotion_le.inverse_transform([pred_class])[0]
    return pred_label
