import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Features and target
X = [[i] for i in range(len(df))]
y = df["heart_rate"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

print("Heart Rate Prediction Mini Project")
print("----------------------------------")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Mean Absolute Error:", mae)

# Predict future heart rate
future_sample = [[len(df)]]
future_prediction = model.predict(future_sample)

print("Predicted Future Heart Rate:", future_prediction[0])
