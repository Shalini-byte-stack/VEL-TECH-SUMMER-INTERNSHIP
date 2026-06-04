import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Feature (sample number)
X = [[i] for i in range(len(df))]

# Target (heart rate)
y = df["heart_rate"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
predictions = model.predict(X_test)

# Display results
print("First 5 Predictions:")
for i in range(5):
    print("Actual:", y_test.iloc[i], "Predicted:", predictions[i])