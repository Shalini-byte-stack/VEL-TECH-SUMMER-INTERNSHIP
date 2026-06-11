from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "student_portal_secret"

# Dummy student data (based on student-mat.csv structure)
students = [
    {"id": 1, "name": "Priya Sharma",    "school": "GP", "age": 18, "address": "Urban",  "failures": 0, "G1": 15, "G2": 16, "G3": 17, "studytime": 3, "internet": "Yes"},
    {"id": 2, "name": "Rahul Mehta",     "school": "MS", "age": 17, "address": "Urban",  "failures": 0, "G1": 12, "G2": 13, "G3": 14, "studytime": 2, "internet": "Yes"},
    {"id": 3, "name": "Sneha Patel",     "school": "GP", "age": 15, "address": "Rural",  "failures": 1, "G1": 8,  "G2": 9,  "G3": 10, "studytime": 1, "internet": "No"},
    {"id": 4, "name": "Aditya Kumar",    "school": "GP", "age": 16, "address": "Urban",  "failures": 0, "G1": 18, "G2": 18, "G3": 19, "studytime": 4, "internet": "Yes"},
    {"id": 5, "name": "Ananya Singh",    "school": "MS", "age": 19, "address": "Rural",  "failures": 2, "G1": 6,  "G2": 7,  "G3": 8,  "studytime": 1, "internet": "No"},
    {"id": 6, "name": "Vikram Nair",     "school": "GP", "age": 17, "address": "Urban",  "failures": 0, "G1": 14, "G2": 15, "G3": 16, "studytime": 3, "internet": "Yes"},
    {"id": 7, "name": "Kavya Reddy",     "school": "MS", "age": 16, "address": "Urban",  "failures": 0, "G1": 11, "G2": 12, "G3": 13, "studytime": 2, "internet": "Yes"},
    {"id": 8, "name": "Arjun Iyer",      "school": "GP", "age": 18, "address": "Rural",  "failures": 3, "G1": 5,  "G2": 5,  "G3": 6,  "studytime": 1, "internet": "No"},
    {"id": 9, "name": "Meera Verma",     "school": "GP", "age": 15, "address": "Urban",  "failures": 0, "G1": 17, "G2": 17, "G3": 18, "studytime": 4, "internet": "Yes"},
    {"id": 10,"name": "Rohan Gupta",     "school": "MS", "age": 17, "address": "Urban",  "failures": 1, "G1": 10, "G2": 11, "G3": 11, "studytime": 2, "internet": "Yes"},
]

@app.route("/")
def home():
    total = len(students)
    avg_g3 = round(sum(s["G3"] for s in students) / total, 1)
    passed = sum(1 for s in students if s["G3"] >= 10)
    return render_template("home.html", total=total, avg_g3=avg_g3, passed=passed)

@app.route("/students")
def student_list():
    return render_template("students.html", students=students)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name      = request.form.get("name", "").strip()
        school    = request.form.get("school", "GP")
        age       = request.form.get("age", "")
        address   = request.form.get("address", "Urban")
        studytime = request.form.get("studytime", "2")
        internet  = request.form.get("internet", "Yes")

        if not name or not age:
            flash("Please fill in all required fields.", "error")
            return render_template("register.html")

        new_student = {
            "id":        len(students) + 1,
            "name":      name,
            "school":    school,
            "age":       int(age),
            "address":   address,
            "failures":  0,
            "G1":        0,
            "G2":        0,
            "G3":        0,
            "studytime": int(studytime),
            "internet":  internet,
        }
        students.append(new_student)
        flash(f"Student '{name}' registered successfully!", "success")
        return redirect(url_for("student_list"))

    return render_template("register.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
