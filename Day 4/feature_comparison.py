import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Create sample numbers
samples = range(len(df))

# Heart rate values
heart_rate = df["heart_rate"]

# Feature Comparison Plot
plt.figure(figsize=(8, 5))
plt.scatter(samples, heart_rate)

plt.title("Feature Comparison: Sample vs Heart Rate")
plt.xlabel("Sample Number")
plt.ylabel("Heart Rate")

plt.grid(True)
plt.show()
