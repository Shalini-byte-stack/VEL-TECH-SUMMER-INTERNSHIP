import numpy as np

marks = np.array([78, 85, 92, 67, 55, 88, 95, 73, 60, 81])

print("Mean:", marks.mean())
print("Highest:", marks.max())
print("Lowest:", marks.min())
print("Standard Deviation:", marks.std())

passed = marks[marks >= 50]
print("Number Passed:", len(passed))