# train_dummy_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

# Mental workload model (3-class)
X_workload = np.random.rand(1000, 10)
y_workload = np.random.choice(['low', 'medium', 'high'], 1000)
le_workload = LabelEncoder()
y_workload_enc = le_workload.fit_transform(y_workload)
workload_model = RandomForestClassifier(n_estimators=50, random_state=42)
workload_model.fit(X_workload, y_workload_enc)
joblib.dump((workload_model, le_workload), "models/workload_model.pkl")

# Accuracy model (binary)
X_accuracy = np.random.rand(1000, 20)
y_accuracy = np.random.choice([0, 1], 1000)
accuracy_model = LogisticRegression()
accuracy_model.fit(X_accuracy, y_accuracy)
joblib.dump(accuracy_model, "models/accuracy_model.pkl")

# Emotion model (multi-class)
X_emotion = np.random.rand(1000, 15)
y_emotion = np.random.choice(['engaged', 'confused', 'neutral', 'happy', 'sad'], 1000)
le_emotion = LabelEncoder()
y_emotion_enc = le_emotion.fit_transform(y_emotion)
emotion_model = RandomForestClassifier(n_estimators=50, random_state=42)
emotion_model.fit(X_emotion, y_emotion_enc)
joblib.dump((emotion_model, le_emotion), "models/emotion_model.pkl")

