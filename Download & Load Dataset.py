import pandas as pd

df = pd.read_csv("HeartRate-Mat.csv")

print("Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)