import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("HeartRate-Mat.csv")

def label_hr(hr):
    if hr < 60:
        return "Bradycardia"
    elif hr > 100:
        return "Tachycardia"
    else:
        return "Normal"

df["heart_rate_condition"] = df["heart_rate"].apply(label_hr)

df = pd.get_dummies(df, columns=["activity_type"])

X = df.select_dtypes(include=["int64","float64"]).drop("heart_rate", axis=1)
y = df["heart_rate_condition"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

pickle.dump(model,open("model.pkl","wb"))
