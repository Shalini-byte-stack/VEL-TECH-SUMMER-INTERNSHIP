from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os

app = Flask(__name__)
DB = "students.db"

# ── Database Setup ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT    NOT NULL,
            roll  TEXT    NOT NULL UNIQUE,
            email TEXT    NOT NULL,
            dept  TEXT    NOT NULL,
            year  TEXT    NOT NULL
        )
    ''')
    # Insert dummy data only if table is empty
    c.execute("SELECT COUNT(*) FROM students")
    if c.fetchone()[0] == 0:
        dummy = [
            ("Yuvaraju",     "22AD001", "yuvaraju@veltech.edu.in", "AI & Data Science", "2nd Year"),
            ("Shalini",      "22AD002", "shalini@veltech.edu.in",  "AI & Data Science", "2nd Year"),
            ("Akshara",      "22AD003", "akshara@veltech.edu.in",  "AI & Data Science", "2nd Year"),
            ("Rahul Kumar",  "22CS010", "rahul@veltech.edu.in",    "CSE",               "2nd Year"),
            ("Priya Sharma", "22EC021", "priya@veltech.edu.in",    "ECE",               "2nd Year"),
        ]
        c.executemany(
            "INSERT INTO students (name, roll, email, dept, year) VALUES (?,?,?,?,?)",
            dummy
        )
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row   # lets us use row["name"] instead of row[0]
    return conn

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    conn.close()
    return render_template("home.html", total=total)

@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    error   = None
    if request.method == "POST":
        name  = request.form["name"].strip()
        roll  = request.form["roll"].strip()
        email = request.form["email"].strip()
        dept  = request.form["dept"]
        year  = request.form["year"]
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO students (name, roll, email, dept, year) VALUES (?,?,?,?,?)",
                (name, roll, email, dept, year)
            )
            conn.commit()
            conn.close()
            message = f"Student '{name}' registered successfully!"
        except sqlite3.IntegrityError:
            error = f"Roll number '{roll}' already exists! Please use a different roll number."
    return render_template("register.html", message=message, error=error)

@app.route("/students")
def students_list():
    conn     = get_db()
    students = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("students.html", students=students)

@app.route("/delete/<int:sid>")
def delete_student(sid):
    conn = get_db()
    conn.execute("DELETE FROM students WHERE id = ?", (sid,))
    conn.commit()
    conn.close()
    return redirect(url_for("students_list"))

@app.route("/about")
def about():
    return render_template("about.html")

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
