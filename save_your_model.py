import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Features and target
X = [[i] for i in range(len(df))]
y = df["heart_rate"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("heart_rate_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as heart_rate_model.pkl")