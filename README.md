# Real-Time Heart Rate Anomaly Detector

## Problem Statement
The goal of this project is to detect abnormal heart rate conditions from patient heart rate readings and classify them into Bradycardia, Normal, or Tachycardia categories.

## Dataset
HeartRate-Mat.csv

## Algorithm
Random Forest Classifier

## Features Used
- Heart Rate
- ECG Signal
- SPO2
- Respiration Rate
- Stress Level
- Activity Type

## Output Classes
- Bradycardia (Heart Rate < 60 bpm)
- Normal (Heart Rate 60–100 bpm)
- Tachycardia (Heart Rate > 100 bpm)

## Model Accuracy
XX%

## Files
- load_data.py
- eda.py
- clean_data.py
- train_model.py
- predict.py
- charts.py
- model.pkl

## How to Run

1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python train_model.py
python predict.py

---

### Step 4: Save the File

Press:

```text
Ctrl + S