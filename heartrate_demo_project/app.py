import sqlite3
import csv
import os
from flask import Flask, render_template, request, redirect, url_for, flash, g

app = Flask(__name__)
app.secret_key = "heart_portal_2026"
DATABASE = "patients.db"

# ── DB Helpers ─────────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                name                TEXT    NOT NULL,
                gender              TEXT    NOT NULL,
                phone               TEXT    NOT NULL,
                address             TEXT    NOT NULL,
                heart_rate          REAL    NOT NULL,
                blood_pressure_sys  REAL    NOT NULL,
                blood_pressure_dia  REAL    NOT NULL,
                blood_glucose       REAL    NOT NULL,
                ecg_signal          REAL    NOT NULL,
                spo2_level          REAL,
                activity_type       TEXT,
                stress_level_index  REAL,
                condition           TEXT,
                registered_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        count = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        if count == 0:
            # Seed from real CSV
            csv_path = os.path.join(os.path.dirname(__file__), "HeartRate_Cleaned.csv")
            if os.path.exists(csv_path):
                names   = ["Priya Sharma","Rahul Mehta","Sneha Patel","Aditya Kumar",
                           "Ananya Singh","Vikram Nair","Kavya Reddy","Arjun Iyer",
                           "Meera Verma","Rohan Gupta","Divya Nair","Suresh Kumar",
                           "Pooja Singh","Amit Shah","Rekha Verma"]
                genders = ["Female","Male","Female","Male","Female",
                           "Male","Female","Male","Female","Male",
                           "Female","Male","Female","Male","Female"]
                phones  = [f"98765{43200+i}" for i in range(15)]
                cities  = ["Mumbai","Delhi","Bengaluru","Chennai","Hyderabad",
                           "Pune","Kolkata","Jaipur","Ahmedabad","Lucknow",
                           "Surat","Nagpur","Indore","Bhopal","Visakhapatnam"]

                with open(csv_path, newline='') as f:
                    rows = list(csv.DictReader(f))

                seed_rows = rows[:15]
                for i, row in enumerate(seed_rows):
                    hr  = float(row['heart_rate'])
                    if hr < 60:
                        cond = "Bradycardia"
                    elif hr > 100:
                        cond = "Tachycardia"
                    else:
                        cond = "Normal"

                    db.execute("""
                        INSERT INTO patients
                        (name,gender,phone,address,heart_rate,blood_pressure_sys,
                         blood_pressure_dia,blood_glucose,ecg_signal,spo2_level,
                         activity_type,stress_level_index,condition)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        names[i], genders[i], phones[i],
                        f"{100+i} Hospital Road, {cities[i]}",
                        hr,
                        float(row['blood_pressure_sys']),
                        float(row['blood_pressure_dia']),
                        float(row['blood_glucose']),
                        float(row['ecg_signal']),
                        float(row['spo2_level']),
                        row['activity_type'],
                        float(row['stress_level_index']),
                        cond
                    ))
                db.commit()

def classify(hr):
    hr = float(hr)
    if hr < 60:   return "Bradycardia"
    if hr > 100:  return "Tachycardia"
    return "Normal"

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    db = get_db()
    total    = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    normal   = db.execute("SELECT COUNT(*) FROM patients WHERE condition='Normal'").fetchone()[0]
    brady    = db.execute("SELECT COUNT(*) FROM patients WHERE condition='Bradycardia'").fetchone()[0]
    tachy    = db.execute("SELECT COUNT(*) FROM patients WHERE condition='Tachycardia'").fetchone()[0]
    avg_hr   = db.execute("SELECT ROUND(AVG(heart_rate),1) FROM patients").fetchone()[0] or 0
    latest   = db.execute("SELECT name FROM patients ORDER BY id DESC LIMIT 1").fetchone()
    latest_name = latest["name"] if latest else "—"
    return render_template("home.html",
        total=total, normal=normal, brady=brady, tachy=tachy,
        avg_hr=avg_hr, latest_name=latest_name)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name      = request.form.get("name","").strip()
        gender    = request.form.get("gender","").strip()
        phone     = request.form.get("phone","").strip()
        address   = request.form.get("address","").strip()
        hr        = request.form.get("heart_rate","").strip()
        bp_sys    = request.form.get("blood_pressure_sys","").strip()
        bp_dia    = request.form.get("blood_pressure_dia","").strip()
        glucose   = request.form.get("blood_glucose","").strip()
        ecg       = request.form.get("ecg_signal","").strip()
        spo2      = request.form.get("spo2_level","").strip()
        activity  = request.form.get("activity_type","").strip()
        stress    = request.form.get("stress_level_index","").strip()

        if not all([name, gender, phone, address, hr, bp_sys, bp_dia, glucose, ecg]):
            flash("All required fields must be filled in.", "error")
            return render_template("register.html")

        cond = classify(hr)
        db = get_db()
        db.execute("""
            INSERT INTO patients
            (name,gender,phone,address,heart_rate,blood_pressure_sys,
             blood_pressure_dia,blood_glucose,ecg_signal,spo2_level,
             activity_type,stress_level_index,condition)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (name, gender, phone, address,
              float(hr), float(bp_sys), float(bp_dia),
              float(glucose), float(ecg),
              float(spo2) if spo2 else None,
              activity or None,
              float(stress) if stress else None,
              cond))
        db.commit()
        flash(f"✅ Patient '{name}' registered! Condition: {cond}", "success")
        return redirect(url_for("patient_list"))

    return render_template("register.html")

@app.route("/patients")
def patient_list():
    db = get_db()
    patients = db.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    return render_template("patients.html", patients=patients)

@app.route("/edit/<int:pid>", methods=["GET","POST"])
def edit(pid):
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for("patient_list"))

    if request.method == "POST":
        name    = request.form.get("name","").strip()
        gender  = request.form.get("gender","").strip()
        phone   = request.form.get("phone","").strip()
        address = request.form.get("address","").strip()
        hr      = request.form.get("heart_rate","").strip()
        bp_sys  = request.form.get("blood_pressure_sys","").strip()
        bp_dia  = request.form.get("blood_pressure_dia","").strip()
        glucose = request.form.get("blood_glucose","").strip()
        ecg     = request.form.get("ecg_signal","").strip()
        spo2    = request.form.get("spo2_level","").strip()
        activity= request.form.get("activity_type","").strip()
        stress  = request.form.get("stress_level_index","").strip()
        cond    = classify(hr)

        db.execute("""
            UPDATE patients SET
            name=?,gender=?,phone=?,address=?,heart_rate=?,blood_pressure_sys=?,
            blood_pressure_dia=?,blood_glucose=?,ecg_signal=?,spo2_level=?,
            activity_type=?,stress_level_index=?,condition=?
            WHERE id=?
        """, (name, gender, phone, address,
              float(hr), float(bp_sys), float(bp_dia),
              float(glucose), float(ecg),
              float(spo2) if spo2 else None,
              activity or None,
              float(stress) if stress else None,
              cond, pid))
        db.commit()
        flash(f"✅ Patient '{name}' updated! Condition: {cond}", "success")
        return redirect(url_for("patient_list"))

    return render_template("edit.html", patient=patient)

@app.route("/delete/<int:pid>", methods=["POST"])
def delete(pid):
    db = get_db()
    p = db.execute("SELECT name FROM patients WHERE id=?", (pid,)).fetchone()
    if p:
        db.execute("DELETE FROM patients WHERE id=?", (pid,))
        db.commit()
        flash(f"🗑️ Patient '{p['name']}' deleted.", "success")
    return redirect(url_for("patient_list"))

@app.route("/about")
def about():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    return render_template("about.html", total=total)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
