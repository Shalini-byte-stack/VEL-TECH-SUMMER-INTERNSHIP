import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Features and target
X = [[i] for i in range(len(df))]
y = df["heart_rate"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

plt.title("Actual vs Predicted Heart Rate")
plt.xlabel("Actual Heart Rate")
plt.ylabel("Predicted Heart Rate")

plt.grid(True)
plt.show()
