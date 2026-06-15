import os
import io
import pickle
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, make_response, jsonify)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR  = os.path.join(BASE_DIR, 'model_assets')
DEFAULT_CSV = os.path.join(BASE_DIR, 'HeartRate_Cleaned.csv')
DB_PATH     = os.path.join(BASE_DIR, 'heartsync.db')

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = 'heartsync_secret_2026'

# ── Load ML assets ─────────────────────────────────────────────────────
print('Loading model assets…')
scaler          = joblib.load(os.path.join(ASSETS_DIR, 'scaler.pkl'))
activity_enc    = joblib.load(os.path.join(ASSETS_DIR, 'activity_encoder.pkl'))
label_enc       = joblib.load(os.path.join(ASSETS_DIR, 'label_encoder.pkl'))
rf_model        = joblib.load(os.path.join(ASSETS_DIR, 'final_model.pkl'))
iso_model       = joblib.load(os.path.join(ASSETS_DIR, 'anomaly_model.pkl'))
feature_cols    = pickle.load(open(os.path.join(ASSETS_DIR, 'feature_columns.pkl'), 'rb'))
print('All models loaded.')

# ═══════════════════════════════════════════════════════════════════════
# SQLite helpers
# ═══════════════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create table and seed 10 demo rows if DB is empty."""
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id          TEXT,
            timestamp           TEXT,
            heart_rate          REAL,
            spo2_level          REAL,
            ecg_signal          REAL,
            respiration_rate    REAL,
            body_temperature    REAL,
            blood_pressure_sys  REAL,
            blood_pressure_dia  REAL,
            blood_glucose       REAL,
            eeg_alpha_power     REAL,
            eeg_beta_power      REAL,
            emg_signal_strength REAL,
            fall_detected       INTEGER,
            activity_type       TEXT,
            step_count          REAL,
            ambient_temperature REAL,
            stress_level_index  REAL,
            predicted_stress_level TEXT,
            heart_condition     TEXT,
            is_anomaly          INTEGER,
            anomaly_score       REAL
        )
    ''')
    conn.commit()

    # Seed from CSV if DB is empty
    if conn.execute('SELECT COUNT(*) FROM readings').fetchone()[0] == 0:
        if os.path.exists(DEFAULT_CSV):
            df = pd.read_csv(DEFAULT_CSV).head(10)   # seed exactly 10 rows
            processed = process_dataset(df)
            rows_to_insert = processed.to_dict(orient='records')
            _bulk_insert(conn, rows_to_insert)
            conn.commit()
            print(f'Seeded {len(rows_to_insert)} demo records into SQLite.')
    conn.close()

def _bulk_insert(conn, records):
    cols = ['patient_id','timestamp','heart_rate','spo2_level','ecg_signal',
            'respiration_rate','body_temperature','blood_pressure_sys',
            'blood_pressure_dia','blood_glucose','eeg_alpha_power',
            'eeg_beta_power','emg_signal_strength','fall_detected',
            'activity_type','step_count','ambient_temperature',
            'stress_level_index','predicted_stress_level',
            'heart_condition','is_anomaly','anomaly_score']
    placeholders = ','.join(['?'] * len(cols))
    sql = f"INSERT INTO readings ({','.join(cols)}) VALUES ({placeholders})"
    for r in records:
        vals = [r.get(c, None) for c in cols]
        # Coerce booleans to int
        vals[cols.index('is_anomaly')] = int(bool(r.get('is_anomaly', False)))
        conn.execute(sql, vals)

def load_all_readings():
    conn = get_db()
    rows = conn.execute('SELECT * FROM readings ORDER BY id DESC').fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ═══════════════════════════════════════════════════════════════════════
# ML Processing
# ═══════════════════════════════════════════════════════════════════════
def classify_heart(hr):
    try:
        hr = float(hr)
        if hr < 60:   return 'Bradycardia'
        if hr > 100:  return 'Tachycardia'
        return 'Normal'
    except:
        return 'Normal'

def process_dataset(df):
    df = df.copy()
    df['heart_condition'] = df['heart_rate'].apply(classify_heart)

    known = list(activity_enc.classes_)
    df['activity_type'] = df['activity_type'].apply(
        lambda a: str(a).strip().lower() if str(a).strip().lower() in known else 'resting')

    df_tmp = df.copy()
    df_tmp['activity_type'] = activity_enc.transform(df_tmp['activity_type'])

    missing = [c for c in feature_cols if c not in df_tmp.columns]
    for c in missing:
        df_tmp[c] = 0

    X       = df_tmp[feature_cols]
    X_sc    = scaler.transform(X)
    stress  = rf_model.predict(X_sc)
    anomaly = iso_model.predict(X_sc)
    score   = iso_model.decision_function(X_sc)

    df['predicted_stress_level'] = label_enc.inverse_transform(stress)
    df['is_anomaly']   = (anomaly == -1)
    df['anomaly_score']= score

    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].astype(str)
    return df

# ═══════════════════════════════════════════════════════════════════════
# Initialise DB on startup
# ═══════════════════════════════════════════════════════════════════════
init_db()

# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════
def _summary(records):
    if not records:
        return dict(total=0, anomalies=0, tachy=0, brady=0,
                    avg_hr=0.0, avg_spo2=0.0)
    total    = len(records)
    anomalies= sum(1 for r in records if r['is_anomaly'])
    tachy    = sum(1 for r in records if r['heart_condition'] == 'Tachycardia')
    brady    = sum(1 for r in records if r['heart_condition'] == 'Bradycardia')
    avg_hr   = round(sum(r['heart_rate']  for r in records) / total, 1)
    avg_spo2 = round(sum(r['spo2_level'] for r in records) / total, 1)
    return dict(total=total, anomalies=anomalies, tachy=tachy,
                brady=brady, avg_hr=avg_hr, avg_spo2=avg_spo2)


@app.route('/')
def index():
    records = load_all_readings()
    s = _summary(records)
    return render_template('index.html',
        readings=records,
        total_records=s['total'],
        anomaly_count=s['anomalies'],
        tachycardia_count=s['tachy'],
        bradycardia_count=s['brady'],
        avg_heart_rate=s['avg_hr'],
        avg_spo2=s['avg_spo2'])


@app.route('/add-patient', methods=['GET'])
def add_patient_page():
    return render_template('add_patient.html')


@app.route('/input-manual', methods=['POST'])
def input_manual():
    try:
        row = {
            'patient_id':         request.form.get('patient_id', 'P001'),
            'heart_rate':         float(request.form.get('heart_rate', 75)),
            'spo2_level':         float(request.form.get('spo2_level', 98.0)),
            'ecg_signal':         float(request.form.get('ecg_signal', 0.05)),
            'respiration_rate':   int(request.form.get('respiration_rate', 15)),
            'body_temperature':   float(request.form.get('body_temperature', 36.6)),
            'blood_pressure_sys': int(request.form.get('blood_pressure_sys', 120)),
            'blood_pressure_dia': int(request.form.get('blood_pressure_dia', 80)),
            'blood_glucose':      float(request.form.get('blood_glucose', 100.0)),
            'stress_level_index': int(request.form.get('stress_level_index', 50)),
            'fall_detected':      1 if request.form.get('fall_detected') else 0,
            'step_count':         int(request.form.get('step_count', 0)),
            'activity_type':      request.form.get('activity_type', 'resting'),
            'eeg_alpha_power':    9.0,
            'eeg_beta_power':     6.0,
            'emg_signal_strength':0.5,
            'ambient_temperature':27.0,
            'latitude':           12.9716,
            'longitude':          77.5946,
            'timestamp':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        processed = process_dataset(pd.DataFrame([row]))
        conn = get_db()
        _bulk_insert(conn, processed.to_dict(orient='records'))
        conn.commit()
        conn.close()
        flash(f"Reading for {row['patient_id']} saved successfully!", 'success')
    except Exception as e:
        flash(f'Error: {e}', 'error')
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('index'))
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        flash('Only CSV files are supported.', 'error')
        return redirect(url_for('index'))
    try:
        df = pd.read_csv(f)
        required = ['heart_rate', 'spo2_level', 'ecg_signal', 'activity_type']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            flash(f'Missing columns: {", ".join(missing)}', 'error')
            return redirect(url_for('index'))
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
        processed = process_dataset(df)
        conn = get_db()
        _bulk_insert(conn, processed.to_dict(orient='records'))
        conn.commit()
        conn.close()
        flash(f'Uploaded {len(df)} records from {f.filename}.', 'success')
    except Exception as e:
        flash(f'Error processing file: {e}', 'error')
    return redirect(url_for('index'))


@app.route('/reset')
def reset():
    conn = get_db()
    conn.execute('DELETE FROM readings')
    conn.commit()
    conn.close()
    # Re-seed 10 demo rows
    if os.path.exists(DEFAULT_CSV):
        df = pd.read_csv(DEFAULT_CSV).head(10)
        processed = process_dataset(df)
        conn = get_db()
        _bulk_insert(conn, processed.to_dict(orient='records'))
        conn.commit()
        conn.close()
    flash('Dashboard reset to 10 demo records.', 'success')
    return redirect(url_for('index'))


@app.route('/delete/<int:rid>', methods=['POST'])
def delete_reading(rid):
    conn = get_db()
    conn.execute('DELETE FROM readings WHERE id = ?', (rid,))
    conn.commit()
    conn.close()
    flash('Record deleted.', 'success')
    return redirect(url_for('index'))


@app.route('/download-template')
def download_template():
    headers = ['patient_id','heart_rate','spo2_level','ecg_signal',
               'respiration_rate','body_temperature','blood_pressure_sys',
               'blood_pressure_dia','blood_glucose','eeg_alpha_power',
               'eeg_beta_power','emg_signal_strength','fall_detected',
               'activity_type','step_count','ambient_temperature',
               'stress_level_index','timestamp']
    rows = [
        ['P100',72,98.2,0.05,14,36.6,120,80,110,10,6,0.5,0,'resting',200,27,'45','2026-06-14 08:00:00'],
        ['P101',115,97.5,0.12,18,37.1,130,85,125,9,7.5,0.7,0,'walking',1500,28,'80','2026-06-14 08:01:00'],
        ['P102',48,96.0,-0.05,11,35.9,110,70,95,11.5,4,0.3,0,'sleeping',0,27,'20','2026-06-14 08:02:00'],
    ]
    df = pd.DataFrame(rows, columns=headers)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    resp = make_response(buf.getvalue())
    resp.headers['Content-Disposition'] = 'attachment; filename=heartsync_template.csv'
    resp.headers['Content-type'] = 'text/csv'
    return resp


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5005)
