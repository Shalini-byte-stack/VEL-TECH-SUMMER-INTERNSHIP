import pandas as pd

df = pd.read_csv("HeartRate-Mat.csv")

print(df.describe())

print("\nNull Values")
print(df.isnull().sum())

print("\nActivity Type Count")
print(df["activity_type"].value_counts())

print("\nFall Detection Count")
print(df["fall_detected"].value_counts())