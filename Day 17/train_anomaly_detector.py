import os
import pickle
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# Path definitions
csv_path = 'HeartRate_Cleaned.csv'
assets_dir = 'model_assets'
model_out_path = os.path.join(assets_dir, 'anomaly_model.pkl')

print("Starting Isolation Forest training process...")

# 1. Load data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}")
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Load scaler and encoders
print("Loading preprocessing assets...")
scaler = joblib.load(os.path.join(assets_dir, 'scaler.pkl'))
activity_encoder = joblib.load(os.path.join(assets_dir, 'activity_encoder.pkl'))
feature_cols = pickle.load(open(os.path.join(assets_dir, 'feature_columns.pkl'), 'rb'))

print(f"Features for modeling: {feature_cols}")

# 3. Preprocess data
df_proc = df.copy()
df_proc['activity_type'] = activity_encoder.transform(df_proc['activity_type'])

# Reorder columns to match feature_columns
X = df_proc[feature_cols]

# Scale features
X_scaled = scaler.transform(X)

# 4. Train Isolation Forest
print("Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.04,
    random_state=42
)
iso_forest.fit(X_scaled)

# Evaluate on training data
preds = iso_forest.predict(X_scaled)
anomaly_count = (preds == -1).sum()
normal_count = (preds == 1).sum()
print(f"Training Complete!")
print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(preds)*100:.2f}%) and {normal_count} normal points.")

# 5. Save the trained model
print(f"Saving model to {model_out_path}...")
joblib.dump(iso_forest, model_out_path)
print("Model saved successfully!")
