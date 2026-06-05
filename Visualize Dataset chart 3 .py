import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HeartRate-Mat.csv")

df.hist(figsize=(10,8))

plt.show()