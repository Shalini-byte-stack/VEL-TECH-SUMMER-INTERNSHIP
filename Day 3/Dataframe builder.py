import pandas as pd

data = {
    "Name": ["shalini", "Priya", "Ravi", "Kiran", "Divya"],
    "Age": [19, 20, 18, 21, 20],
    "City": ["Chennai", "Madurai", "Coimbatore", "Salem", "Trichy"],
    "Marks": [85, 92, 76, 88, 95]
}

df = pd.DataFrame(data)

df["Result"] = df["Marks"].apply(
    lambda x: "Pass" if x >= 50 else "Fail"
)

print(df)
print("\nShape:", df.shape)
print("\nData Types:")
print(df.dtypes)
