import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HeartRate-Mat.csv")

plt.scatter(df["heart_rate"], df["ecg_signal"])
plt.xlabel("Heart Rate")
plt.ylabel("ecg_signal")
plt.title("Heart Rate vs ecg signal")
plt.savefig("heart_rate_stress.png")
plt.show()
