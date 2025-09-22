import streamlit as st
import numpy as np
from collections import deque
import plotly.graph_objects as go
from utils import predict_workload, predict_accuracy, predict_emotion
from streamlit_autorefresh import st_autorefresh

# Page config
st.set_page_config(layout="wide", page_title="Mental State Monitoring Dashboard")

# Constants
HISTORY_LENGTH = 60

# Initialize or retrieve history
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = deque(maxlen=HISTORY_LENGTH)
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=HISTORY_LENGTH)
if 'workload_history' not in st.session_state:
    st.session_state.workload_history = deque(maxlen=HISTORY_LENGTH)

def make_gauge_chart(level):
    level_map = {"low": 30, "medium": 60, "high": 90}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=level_map.get(level, 0),
        title={'text': "Mental Workload"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 40], 'color': "lightgreen"},
                         {'range': [40, 70], 'color': "yellow"},
                         {'range': [70, 100], 'color': "red"}],
               'threshold': {'line': {'color': "black", 'width':4},
                             'thickness': 0.75,
                             'value': level_map.get(level, 0)}}))
    return fig

def make_accuracy_chart(history):
    time_steps = list(range(-len(history)+1, 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=history, mode='lines+markers', name='Accuracy Probability'))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Threshold 0.5", annotation_position="top right")
    fig.update_layout(
        title="Accuracy Probability Over Time",
        xaxis_title="Time (seconds ago)",
        yaxis_title="Probability",
        yaxis_range=[0, 1]
    )
    return fig

def make_emotion_timeline(history):
    emotion_colors = {
        "engaged": "green",
        "confused": "orange",
        "neutral": "grey",
        "happy": "yellow",
        "sad": "blue",
    }
    colors = [emotion_colors.get(e, "lightgrey") for e in history]
    time_steps = list(range(-len(history)+1, 1))
    fig = go.Figure(go.Bar(
        x=time_steps,
        y=[1]*len(history),
        marker_color=colors,
        showlegend=False))
    fig.update_layout(
        title="Emotion Transitions Over Time",
        yaxis=dict(showticklabels=False),
        xaxis_title="Time (seconds ago)"
    )
    return fig

def simulate_features():
    eeg_feat = list(np.random.rand(10))
    multimodal_feat = list(np.random.rand(20))
    facial_feat = list(np.random.rand(7))
    return eeg_feat, multimodal_feat, facial_feat, eeg_feat[:8]

# Auto-refresh every 1000ms (1s)
count = st_autorefresh(interval=1000, limit=None, key="autorefresh")

# Simulate one step and update session state history
eeg_feat, multimodal_feat, facial_feat, emotion_eeg_feat = simulate_features()
workload = predict_workload(eeg_feat)
accuracy_prob = predict_accuracy(multimodal_feat)
emotion = predict_emotion(facial_feat, emotion_eeg_feat)

st.session_state.workload_history.append(workload)
st.session_state.accuracy_history.append(accuracy_prob)
st.session_state.emotion_history.append(emotion)

# Dashboard layout
st.title("Mental State Monitoring Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    fig_gauge = make_gauge_chart(workload)
    st.plotly_chart(fig_gauge, use_container_width=True, key="workload_gauge")

with col2:
    fig_accuracy = make_accuracy_chart(list(st.session_state.accuracy_history))
    st.plotly_chart(fig_accuracy, use_container_width=True, key="accuracy_chart")

with col3:
    fig_emotion = make_emotion_timeline(list(st.session_state.emotion_history))
    st.plotly_chart(fig_emotion, use_container_width=True, key="emotion_timeline")

st.markdown(f"""
### Participant Snapshot
- Current workload: **{workload.capitalize()}**
- Current accuracy probability: **{accuracy_prob:.2%}**
- Current emotion: **{emotion.capitalize()}**
""")
