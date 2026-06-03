import pandas as pd

df = pd.read_csv("student-mat.csv", sep="\t")

print("Average G3 by Study Time")
print(df.groupby("studytime")["G3"].mean())

print("\nAverage G3 by Gender")
print(df.groupby("sex")["G3"].mean())

print("\nTop 5 Students by G3")
print(df.nlargest(5, "G3")[["sex", "studytime", "G3"]])