# Mental State Monitoring Dashboard

## Overview
Real-time interactive dashboard for visualizing mental workload levels, accuracy probabilities, and emotion transitions from simulated multimodal sensor data and pretrained models.

## Folder Structure
- data/: CSV sensor data files
- models/: Pretrained workload, accuracy, emotion models (.pkl)
- app/: Backend data simulation and Dash-based dashboard frontend
- README.md: This documentation

## Setup
1. Create and activate Python environment
2. Install dependencies:
   pip install dash pandas numpy plotly scikit-learn
3. Add your CSV data files in data/
4. Add pretrained models in models/ (optional)
5. Run dashboard:
   python -m app.dashboard

## Features
- Mental workload gauge (Low, Medium, High)
- Accuracy probability time-series chart
- Sankey diagram for emotion state transitions
- Simulated live data streaming integration

## Notes
- Dummy predictors used if pretrained models are missing.
- Adapt model loading in models/predictors.py as needed.
