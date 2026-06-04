import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heartrate-mat.csv")

# Heart rate column
heart_rate = df["heart_rate"]

# Create chart
plt.figure(figsize=(10, 5))

plt.plot(
    heart_rate,
    marker='o',
    linestyle='--',
    linewidth=2
)

plt.title("Custom Styled Heart Rate Chart", fontsize=16)
plt.xlabel("Sample Number", fontsize=12)
plt.ylabel("Heart Rate", fontsize=12)

plt.grid(True)
plt.legend(["Heart Rate"])

plt.show()
