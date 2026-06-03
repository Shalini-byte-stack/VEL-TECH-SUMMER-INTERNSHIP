import pandas as pd

df = pd.read_csv("student-mat.csv", sep="\t")

print("First 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nInternet Usage:")
print(df["internet"].value_counts())
