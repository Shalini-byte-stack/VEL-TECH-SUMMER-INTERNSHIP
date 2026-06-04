import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Select heart rate column
heart_rate = df["heart_rate"]

# Line Chart
plt.figure(figsize=(6,4))
plt.plot(heart_rate)
plt.title("Heart Rate Line Chart")
plt.xlabel("Sample")
plt.ylabel("Heart Rate")
plt.show()

# Bar Chart
plt.figure(figsize=(6,4))
plt.bar(range(10), heart_rate[:10])
plt.title("Heart Rate Bar Chart")
plt.xlabel("Sample")
plt.ylabel("Heart Rate")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(range(len(heart_rate)), heart_rate)
plt.title("Heart Rate Scatter Plot")
plt.xlabel("Sample")
plt.ylabel("Heart Rate")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(heart_rate, bins=10)
plt.title("Heart Rate Histogram")
plt.xlabel("Heart Rate")
plt.ylabel("Frequency")
plt.show()
