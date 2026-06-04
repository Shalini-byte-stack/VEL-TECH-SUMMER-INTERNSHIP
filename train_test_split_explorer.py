import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Features and target
X = [[i] for i in range(len(df))]
y = df["heart_rate"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display results
print("Total Samples:", len(df))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nFirst 5 Training Values:")
print(y_train.head())

print("\nFirst 5 Testing Values:")
print(y_test.head())