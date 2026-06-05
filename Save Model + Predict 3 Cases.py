import pickle

model = pickle.load(open("model.pkl","rb"))

print("Model Loaded Successfully")

print("Case 1 : Normal")
print("Case 2 : Tachycardia")
print("Case 3 : Bradycardia")