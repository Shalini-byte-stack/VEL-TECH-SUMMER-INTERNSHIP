import pandas as pd

df = pd.read_csv("HeartRate-Mat.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

print(df.isnull().sum())