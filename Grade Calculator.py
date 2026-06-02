def get_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 75:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 45:
        return "D"
    else:
        return "F"

for i in range(5):
    marks = int(input(f"Enter marks for student {i+1}: "))
    print("Grade:", get_grade(marks))