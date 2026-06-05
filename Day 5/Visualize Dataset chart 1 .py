import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HeartRate-Mat.csv")

plt.hist(df["heart_rate"], bins=20)
plt.title("Heart Rate Distribution")
plt.savefig("heart_rate_distribution.png")
plt.show()
