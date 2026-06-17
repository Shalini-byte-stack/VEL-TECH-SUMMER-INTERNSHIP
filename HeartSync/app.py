import os
import io
import re
import math
import pickle
import sqlite3
import joblib
import pandas as pd
import numpy as np
from functools import wraps
from collections import Counter
from datetime import datetime
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, make_response, jsonify, session)
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
except ImportError:               # twilio package not installed
    TwilioClient = None
    class TwilioRestException(Exception):
        pass

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR  = os.path.join(BASE_DIR, 'model_assets')
DEFAULT_CSV = os.path.join(BASE_DIR, 'HeartRate_Cleaned.csv')
DB_PATH     = os.path.join(BASE_DIR, 'heartsync.db')

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
# In production, set a strong, random SECRET_KEY via environment variable.
app.secret_key = os.environ.get('SECRET_KEY', 'heartsync_secret_2026')

# ── Twilio (SMS caretaker alerts) configuration ─────────────────────────
# Set these as environment variables before running the app, e.g.:
#   export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#   export TWILIO_AUTH_TOKEN="your_auth_token"
#   export TWILIO_FROM_NUMBER="+15551234567"
TWILIO_ACCOUNT_SID  = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN   = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER  = os.environ.get('TWILIO_FROM_NUMBER')

TWILIO_CONFIGURED = bool(TwilioClient and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER)
_twilio_client = None
if TWILIO_CONFIGURED:
    try:
        _twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        print(f'Twilio client init failed: {e}')
        TWILIO_CONFIGURED = False
else:
    print('Twilio not configured — caretaker SMS alerts will be simulated/logged only. '
          'Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER to enable real SMS.')

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
    """Create tables and seed 10 demo rows if DB is empty."""
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

    # ── Authentication: caregiver / doctor accounts ──
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT UNIQUE NOT NULL,
            email           TEXT,
            password_hash   TEXT NOT NULL,
            created_at      TEXT
        )
    ''')

    # ── Patient profiles: contact + caretaker info ──
    conn.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id      TEXT PRIMARY KEY,
            full_name       TEXT,
            phone_number    TEXT,
            caretaker_name  TEXT,
            caretaker_phone TEXT,
            created_by      INTEGER,
            created_at      TEXT,
            updated_at      TEXT
        )
    ''')

    # ── Caretaker SMS alert log ──
    conn.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      TEXT,
            reading_id      INTEGER,
            caretaker_name  TEXT,
            caretaker_phone TEXT,
            reasons         TEXT,
            message         TEXT,
            status          TEXT,
            error           TEXT,
            created_at      TEXT
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
    inserted_ids = []
    for r in records:
        vals = [r.get(c, None) for c in cols]
        # Coerce booleans to int
        vals[cols.index('is_anomaly')] = int(bool(r.get('is_anomaly', False)))
        cur = conn.execute(sql, vals)
        inserted_ids.append(cur.lastrowid)
    return inserted_ids

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
# Patient Profiles  (contact number + caretaker info)
# ═══════════════════════════════════════════════════════════════════════
def get_patient(patient_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def load_all_patients():
    conn = get_db()
    rows = conn.execute('SELECT * FROM patients ORDER BY patient_id').fetchall()
    conn.close()
    return [dict(r) for r in rows]

def upsert_patient(patient_id, full_name='', phone_number='', caretaker_name='',
                    caretaker_phone='', created_by=None):
    """Create or update a patient's profile. Blank fields keep the existing value."""
    if not patient_id:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db()
    existing = conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
    if existing:
        full_name       = full_name.strip()       or existing['full_name']
        phone_number    = phone_number.strip()    or existing['phone_number']
        caretaker_name  = caretaker_name.strip()  or existing['caretaker_name']
        caretaker_phone = caretaker_phone.strip() or existing['caretaker_phone']
        conn.execute('''
            UPDATE patients
               SET full_name = ?, phone_number = ?, caretaker_name = ?,
                   caretaker_phone = ?, updated_at = ?
             WHERE patient_id = ?
        ''', (full_name, phone_number, caretaker_name, caretaker_phone, now, patient_id))
    else:
        conn.execute('''
            INSERT INTO patients (patient_id, full_name, phone_number, caretaker_name,
                                   caretaker_phone, created_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, full_name.strip(), phone_number.strip(), caretaker_name.strip(),
              caretaker_phone.strip(), created_by, now, now))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Caretaker SMS Alerts  (Twilio)
# ═══════════════════════════════════════════════════════════════════════
# Conditions considered serious enough to notify a patient's caretaker.
CRITICAL_SPO2          = 92     # SpO2 below this %     → danger
CRITICAL_BP_SYS        = 140    # Systolic BP at/above  → danger (hypertensive)
CRITICAL_BP_DIA        = 90     # Diastolic BP at/above → danger (hypertensive)

def check_critical_conditions(row):
    """Return a list of human-readable reasons this reading warrants a caretaker alert."""
    reasons = []
    if row.get('is_anomaly'):
        reasons.append('Unusual vital-sign pattern flagged by the anomaly-detection model')
    if row.get('fall_detected'):
        reasons.append('Fall detected')
    try:
        spo2 = float(row.get('spo2_level'))
        if spo2 < CRITICAL_SPO2:
            reasons.append(f'Critically low SpO2 ({spo2:g}%)')
    except (TypeError, ValueError):
        pass
    try:
        sys_bp = float(row.get('blood_pressure_sys'))
        dia_bp = float(row.get('blood_pressure_dia'))
        if sys_bp >= CRITICAL_BP_SYS or dia_bp >= CRITICAL_BP_DIA:
            reasons.append(f'Hypertensive blood pressure ({sys_bp:g}/{dia_bp:g} mmHg)')
    except (TypeError, ValueError):
        pass
    return reasons


def send_sms_alert(to_number, body):
    """
    Send an SMS via Twilio. Returns (status, error) where status is one of:
    'sent', 'simulated', 'skipped', 'failed'.
    """
    if not to_number:
        return 'skipped', 'No caretaker phone number on file for this patient.'

    if not TWILIO_CONFIGURED:
        print(f'[SIMULATED SMS to {to_number}]: {body}')
        return 'simulated', ('Twilio is not configured — set TWILIO_ACCOUNT_SID, '
                              'TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER to send real SMS.')

    try:
        _twilio_client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=to_number)
        return 'sent', None
    except TwilioRestException as e:
        return 'failed', str(e)
    except Exception as e:
        return 'failed', str(e)


def trigger_caretaker_alert(row, reading_id):
    """
    Check a (processed) reading for critical conditions and, if found,
    send/simulate an SMS alert to the patient's caretaker and log it.
    Returns the alert dict if one was triggered, otherwise None.
    """
    reasons = check_critical_conditions(row)
    if not reasons:
        return None

    patient_id = row.get('patient_id') or 'Unknown'
    patient = get_patient(patient_id) or {}
    caretaker_name  = patient.get('caretaker_name')  or 'Caretaker'
    caretaker_phone = patient.get('caretaker_phone') or ''
    patient_name    = patient.get('full_name') or patient_id

    reasons_text = '; '.join(reasons)
    message = (
        f"HeartSync ALERT: Patient {patient_name} ({patient_id}) needs attention. "
        f"{reasons_text}. HR {row.get('heart_rate'):g} bpm, SpO2 {row.get('spo2_level'):g}%, "
        f"BP {row.get('blood_pressure_sys'):g}/{row.get('blood_pressure_dia'):g} mmHg "
        f"at {row.get('timestamp')}. Please check on them."
    )

    status, error = send_sms_alert(caretaker_phone, message)

    conn = get_db()
    conn.execute('''
        INSERT INTO alerts (patient_id, reading_id, caretaker_name, caretaker_phone,
                             reasons, message, status, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, reading_id, caretaker_name, caretaker_phone,
          reasons_text, message, status, error, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

    return dict(patient_id=patient_id, reasons=reasons_text, message=message,
                 status=status, error=error, caretaker_phone=caretaker_phone)


# ═══════════════════════════════════════════════════════════════════════
# Authentication helpers
# ═══════════════════════════════════════════════════════════════════════
def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'error')
            return redirect(url_for('login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_alert_count():
    """Make a pending-alert badge count available to every template."""
    try:
        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE status IN ('failed','skipped')"
        ).fetchone()[0]
        conn.close()
    except Exception:
        count = 0
    return dict(pending_alert_count=count, twilio_configured=TWILIO_CONFIGURED)


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


# ═══════════════════════════════════════════════════════════════════════
# Heart-Rate Analysis  (activity-aware accurate classification)
# ═══════════════════════════════════════════════════════════════════════
# Expected resting/active heart-rate ranges per activity type (bpm)
HR_EXPECTED_RANGES = {
    'sleeping': (45, 65),
    'resting':  (60, 100),
    'walking':  (70, 115),
    'running':  (100, 170),
}

def hr_zone_info(hr, activity):
    """Return (zone_label, expected_low, expected_high) for a HR + activity."""
    try:
        hr = float(hr)
    except (TypeError, ValueError):
        hr = 0.0
    activity = (activity or 'resting').strip().lower()
    lo, hi = HR_EXPECTED_RANGES.get(activity, HR_EXPECTED_RANGES['resting'])
    if hr < lo:
        return 'Below Expected', lo, hi
    if hr > hi:
        return 'Above Expected', lo, hi
    return 'Within Expected', lo, hi


def analyze_heart_rate(records):
    """Detailed, activity-aware heart-rate analysis across all readings."""
    if not records:
        return dict(avg=0, min=0, max=0, std=0, trend='No Data',
                     variability='No Data', latest_hr=0, latest_activity='resting',
                     latest_zone='No Data', expected_range='60-100',
                     latest_condition='Normal', latest_is_anomaly=False,
                     resting_avg=0, active_avg=0)

    hr_values = [float(r['heart_rate']) for r in records]
    n = len(hr_values)
    avg = sum(hr_values) / n
    mn, mx = min(hr_values), max(hr_values)
    variance = sum((x - avg) ** 2 for x in hr_values) / n
    std = math.sqrt(variance)

    # records are ordered DESC by id (newest first) -> reverse for chronological order
    chrono = list(reversed(records))
    half = max(1, n // 2)
    first_half_avg  = sum(float(r['heart_rate']) for r in chrono[:half]) / half
    second_half_avg = sum(float(r['heart_rate']) for r in chrono[-half:]) / half
    diff = second_half_avg - first_half_avg
    if abs(diff) < 2:
        trend = 'Stable'
    elif diff > 0:
        trend = 'Rising'
    else:
        trend = 'Falling'

    if std < 8:
        variability = 'Low'
    elif std < 18:
        variability = 'Moderate'
    else:
        variability = 'High'

    latest = records[0]
    zone, lo, hi = hr_zone_info(latest['heart_rate'], latest.get('activity_type'))

    resting_vals = [float(r['heart_rate']) for r in records
                     if (r.get('activity_type') or '').lower() in ('resting', 'sleeping')]
    active_vals   = [float(r['heart_rate']) for r in records
                     if (r.get('activity_type') or '').lower() in ('walking', 'running')]
    resting_avg = round(sum(resting_vals) / len(resting_vals), 1) if resting_vals else 0
    active_avg  = round(sum(active_vals) / len(active_vals), 1) if active_vals else 0

    return dict(
        avg=round(avg, 1), min=mn, max=mx, std=round(std, 1),
        trend=trend, variability=variability,
        latest_hr=latest['heart_rate'], latest_activity=(latest.get('activity_type') or 'resting'),
        latest_zone=zone, expected_range=f'{lo}–{hi}',
        latest_condition=latest['heart_condition'],
        latest_is_anomaly=bool(latest['is_anomaly']),
        resting_avg=resting_avg, active_avg=active_avg,
    )


# ═══════════════════════════════════════════════════════════════════════
# Health Recommendations Engine
# ═══════════════════════════════════════════════════════════════════════
def generate_recommendations(records, summary, hr_analysis):
    """Produce a prioritized list of health recommendation cards."""
    if not records:
        return [dict(level='info', icon='fa-circle-info', title='No Data Yet',
                      message='Add a patient reading or upload a CSV to receive '
                              'personalized health recommendations.')]

    recs = []
    latest = records[0]

    # ── Heart rate / cardiac condition ──
    hr   = latest['heart_rate']
    cond = latest['heart_condition']
    activity = (latest.get('activity_type') or 'resting').title()
    if cond == 'Tachycardia':
        recs.append(dict(level='warning', icon='fa-gauge-high', title='Elevated Heart Rate',
            message=f'Latest reading shows {hr:g} bpm (Tachycardia) during {activity.lower()}. '
                    f'Try slow, deep breathing, sit down and rest, and limit caffeine or '
                    f'stimulants. If a high resting heart rate persists, consult a doctor.'))
    elif cond == 'Bradycardia':
        recs.append(dict(level='warning', icon='fa-gauge-simple', title='Low Heart Rate',
            message=f'Latest reading shows {hr:g} bpm (Bradycardia). This can be normal for '
                    f'well-conditioned individuals, but if it comes with dizziness, fatigue '
                    f'or fainting, seek medical evaluation.'))
    elif hr_analysis['latest_zone'] == 'Within Expected':
        recs.append(dict(level='success', icon='fa-heart-pulse', title='Heart Rate Normal',
            message=f'Latest heart rate of {hr:g} bpm is within the expected range '
                    f'({hr_analysis["expected_range"]} bpm) for {activity.lower()} activity. Keep it up!'))
    else:
        below_above = 'below' if hr_analysis['latest_zone'] == 'Below Expected' else 'above'
        recs.append(dict(level='info', icon='fa-heart-pulse', title='Heart Rate Normal, Unusual for Activity',
            message=f'Latest heart rate of {hr:g} bpm is within the general healthy range, but '
                    f'{below_above} the typical {hr_analysis["expected_range"]} bpm window for '
                    f'{activity.lower()} activity. This can be fine occasionally — keep an eye '
                    f'on the pattern.'))

    # ── Variability / trend ──
    if hr_analysis['variability'] == 'High':
        recs.append(dict(level='info', icon='fa-wave-square', title='High Heart-Rate Variability',
            message=f'Heart rate fluctuated noticeably across recent readings '
                    f'(σ ≈ {hr_analysis["std"]:g} bpm). Maintain consistent sleep, hydration '
                    f'and activity routines, and keep an eye on any sharp spikes.'))
    if hr_analysis['trend'] == 'Rising':
        recs.append(dict(level='info', icon='fa-arrow-trend-up', title='Rising Heart-Rate Trend',
            message='Average heart rate has been trending upward in recent readings. '
                    'Consider checking hydration, stress, sleep quality and caffeine intake.'))
    elif hr_analysis['trend'] == 'Falling':
        recs.append(dict(level='info', icon='fa-arrow-trend-down', title='Falling Heart-Rate Trend',
            message='Average heart rate has been trending downward in recent readings — '
                    'often a sign of improved rest or recovery, but monitor for excessive fatigue.'))

    # ── SpO2 ──
    spo2 = latest['spo2_level']
    if spo2 < 92:
        recs.append(dict(level='danger', icon='fa-lungs', title='Critically Low Oxygen Saturation',
            message=f'SpO2 at {spo2:g}% is below the safe threshold of 92%. '
                    f'Seek medical attention promptly.'))
    elif spo2 < 95:
        recs.append(dict(level='warning', icon='fa-lungs', title='Low Oxygen Saturation',
            message=f'SpO2 at {spo2:g}% is slightly below the ideal 95%+ range. '
                    f'Practice slow diaphragmatic breathing and re-check shortly.'))

    # ── Stress ──
    stress_label = latest.get('predicted_stress_level', 'Low')
    if stress_label == 'High':
        recs.append(dict(level='warning', icon='fa-brain', title='High Stress Level Predicted',
            message='The model predicts a High stress level. Try a short walk, guided '
                    'breathing or meditation, and reduce screen time before bed.'))
    elif stress_label == 'Medium':
        recs.append(dict(level='info', icon='fa-brain', title='Moderate Stress Level',
            message='Stress level is Medium. Short breaks, hydration and light stretching '
                    'can help keep stress from building up.'))

    # ── Blood pressure ──
    sys_bp = latest['blood_pressure_sys']
    dia_bp = latest['blood_pressure_dia']
    if sys_bp >= 140 or dia_bp >= 90:
        recs.append(dict(level='danger', icon='fa-droplet', title='High Blood Pressure',
            message=f'Blood pressure {sys_bp:g}/{dia_bp:g} mmHg is in the hypertensive range. '
                    f'Reduce sodium intake, avoid stimulants, and consult a healthcare '
                    f'provider if it remains elevated.'))
    elif sys_bp >= 130 or dia_bp >= 85:
        recs.append(dict(level='info', icon='fa-droplet', title='Elevated Blood Pressure',
            message=f'Blood pressure {sys_bp:g}/{dia_bp:g} mmHg is slightly elevated. '
                    f'Monitor regularly and maintain a balanced, low-sodium diet.'))

    # ── Glucose ──
    glucose = latest['blood_glucose']
    if glucose >= 140:
        recs.append(dict(level='warning', icon='fa-droplet-slash', title='Elevated Blood Glucose',
            message=f'Blood glucose of {glucose:g} mg/dL is above the typical fasting range '
                    f'(70–100 mg/dL). Consider reviewing recent meals and consult a physician '
                    f'if readings stay high.'))
    elif glucose < 70:
        recs.append(dict(level='warning', icon='fa-droplet-slash', title='Low Blood Glucose',
            message=f'Blood glucose of {glucose:g} mg/dL is below the normal fasting range. '
                    f'Consider a small healthy snack and monitor for symptoms of hypoglycemia.'))

    # ── Anomalies & falls ──
    if summary['anomalies'] > 0:
        recs.append(dict(level='danger', icon='fa-triangle-exclamation', title='Anomalies Detected',
            message=f'The Isolation Forest model flagged {summary["anomalies"]} reading(s) as '
                    f'anomalous out of {summary["total"]}. Review the flagged rows in the log '
                    f'for unusual vital-sign combinations.'))

    if latest.get('fall_detected'):
        recs.append(dict(level='danger', icon='fa-person-falling', title='Fall Detected',
            message='A fall was detected in the most recent reading. Check on the patient '
                    'immediately and assess for injury.'))

    # ── Overall positive note if nothing concerning ──
    if not any(r['level'] in ('warning', 'danger') for r in recs):
        recs.append(dict(level='success', icon='fa-circle-check', title='Overall Status: Good',
            message='All key vitals look healthy. Keep up regular activity, balanced '
                    'nutrition, and a consistent sleep schedule.'))

    # Sort: danger > warning > info > success
    order = {'danger': 0, 'warning': 1, 'info': 2, 'success': 3}
    recs.sort(key=lambda r: order.get(r['level'], 4))
    return recs


# ═══════════════════════════════════════════════════════════════════════
# Analytics / Distributions  (for interactive charts)
# ═══════════════════════════════════════════════════════════════════════
def get_distributions(records):
    stress_counts    = Counter(r.get('predicted_stress_level', 'Low') for r in records)
    activity_counts  = Counter((r.get('activity_type') or 'resting') for r in records)
    condition_counts = Counter(r.get('heart_condition', 'Normal') for r in records)

    patient_hr = {}
    for r in records:
        pid = r.get('patient_id') or 'Unknown'
        patient_hr.setdefault(pid, []).append(float(r['heart_rate']))
    patient_avg_hr = {pid: round(sum(v) / len(v), 1) for pid, v in patient_hr.items()}
    # keep at most 15 patients with most readings for readability
    top_patients = sorted(patient_hr.items(), key=lambda kv: len(kv[1]), reverse=True)[:15]
    patient_avg_hr = {pid: patient_avg_hr[pid] for pid, _ in top_patients}

    return dict(
        stress=dict(stress_counts),
        activity=dict(activity_counts),
        condition=dict(condition_counts),
        patient_avg_hr=patient_avg_hr,
    )


# ═══════════════════════════════════════════════════════════════════════
# Chatbot — rule-based HeartSync health assistant
# ═══════════════════════════════════════════════════════════════════════
def chatbot_reply(message, records, summary, hr_analysis, recommendations):
    msg = (message or '').strip().lower()
    if not msg:
        return "I didn't catch that — could you type your question again?"

    # ── Greetings ──
    if re.search(r'\b(hi|hello|hey|yo|good\s*(morning|afternoon|evening))\b', msg):
        return ("Hello! 👋 I'm your HeartSync health assistant. I can tell you about the "
                "current dashboard stats, the latest heart-rate analysis, health "
                "recommendations, or explain terms like *tachycardia* and *anomaly*. "
                "What would you like to know?")

    if 'thank' in msg:
        return "You're welcome! Stay healthy and keep monitoring your vitals. 💙"

    if any(w in msg for w in ['help', 'what can you do', 'commands']):
        return ("Here's what I can help with:\n"
                "• \"What's the latest heart rate?\" — current HR analysis\n"
                "• \"Give me recommendations\" — personalized health tips\n"
                "• \"Show me a summary\" — dashboard overview\n"
                "• \"Any anomalies?\" — anomaly detection status\n"
                "• \"What's my stress level?\" — stress prediction\n"
                "• \"Check blood pressure / SpO2 / glucose\"\n"
                "• \"What is tachycardia / bradycardia / anomaly?\" — explanations\n"
                "• \"Show patient P019\" — lookup a specific patient's latest reading")

    if summary['total'] == 0:
        return ("There's no data in the dashboard yet. Add a patient reading via "
                "'Add Patient' or upload a CSV file to get started — then ask me again!")

    latest = records[0]

    # ── Specific patient lookup ──
    m = re.search(r'\bp[\s\-]?\d{2,4}\b', msg)
    if m:
        pid = m.group().upper().replace(' ', '').replace('-', '')
        match = next((r for r in records if (r.get('patient_id') or '').upper() == pid), None)
        if match:
            status = 'an Outlier ⚠️' if match['is_anomaly'] else 'Normal ✅'
            return (f"Latest reading for {match['patient_id']} ({match['timestamp']}):\n"
                    f"• Heart Rate: {match['heart_rate']:g} bpm ({match['heart_condition']})\n"
                    f"• SpO2: {match['spo2_level']:g}%\n"
                    f"• BP: {match['blood_pressure_sys']:g}/{match['blood_pressure_dia']:g} mmHg\n"
                    f"• Stress: {match['predicted_stress_level']}\n"
                    f"• Activity: {match['activity_type']}\n"
                    f"• Status: {status}")
        else:
            return f"I couldn't find a record for patient {pid} in the current dashboard data."

    # ── Heart rate analysis ──
    if re.search(r'\b(heart\s*rate|hr|pulse|bpm)\b', msg):
        return (f"💓 Heart-Rate Analysis:\n"
                f"• Latest: {hr_analysis['latest_hr']:g} bpm during "
                f"{hr_analysis['latest_activity']} → {hr_analysis['latest_zone']} "
                f"(expected {hr_analysis['expected_range']} bpm)\n"
                f"• Condition: {hr_analysis['latest_condition']}\n"
                f"• Average across all readings: {hr_analysis['avg']:g} bpm "
                f"(range {hr_analysis['min']:g}–{hr_analysis['max']:g})\n"
                f"• Variability: {hr_analysis['variability']} (σ ≈ {hr_analysis['std']:g} bpm)\n"
                f"• Trend: {hr_analysis['trend']}\n"
                f"• Resting avg: {hr_analysis['resting_avg']:g} bpm · "
                f"Active avg: {hr_analysis['active_avg']:g} bpm")

    # ── Recommendations ──
    if re.search(r'\b(recommend\w*|advi[sc]\w*|suggest\w*|tip\w*|should i)\b', msg):
        top = recommendations[:3]
        lines = [f"{'🔴' if r['level']=='danger' else '🟠' if r['level']=='warning' else '🔵' if r['level']=='info' else '🟢'} "
                 f"{r['title']}: {r['message']}" for r in top]
        return "Here are your top health recommendations:\n\n" + "\n\n".join(lines)

    # ── Anomalies ──
    if re.search(r'\b(anomal\w*|outlier\w*|unusual\w*|abnormal\w*)\b', msg):
        if summary['anomalies'] == 0:
            return ("No anomalies detected ✅ — all readings fall within the patterns "
                    "the Isolation Forest model considers normal.")
        return (f"⚠️ The Isolation Forest model has flagged {summary['anomalies']} out of "
                f"{summary['total']} reading(s) as anomalous. Check the 'Patient Readings "
                f"Log' table — rows highlighted in red are outliers, and the latest "
                f"reading is {'an outlier' if latest['is_anomaly'] else 'normal'}.")

    # ── Stress ──
    if 'stress' in msg:
        return (f"🧠 The latest predicted stress level is **{latest['predicted_stress_level']}** "
                f"(stress index input: {latest['stress_level_index']:g}). "
                f"This is predicted by a Random-Forest model trained on vitals such as "
                f"heart rate, SpO2 and activity.")

    # ── Blood pressure ──
    if re.search(r'\b(blood\s*pressure|bp|systolic|diastolic)\b', msg):
        sys_bp, dia_bp = latest['blood_pressure_sys'], latest['blood_pressure_dia']
        category = ('Hypertensive' if sys_bp >= 140 or dia_bp >= 90 else
                     'Elevated' if sys_bp >= 130 or dia_bp >= 85 else 'Normal')
        return (f"🩸 Latest blood pressure: {sys_bp:g}/{dia_bp:g} mmHg — {category}. "
                f"Normal is roughly below 120/80 mmHg.")

    # ── SpO2 ──
    if re.search(r'\b(spo2|oxygen|saturation)\b', msg):
        spo2 = latest['spo2_level']
        category = 'Critical' if spo2 < 92 else 'Low' if spo2 < 95 else 'Normal'
        return f"🫁 Latest SpO2 is {spo2:g}% — {category}. Healthy SpO2 is typically 95–100%."

    # ── Glucose ──
    if re.search(r'\b(glucose|sugar)\b', msg):
        glucose = latest['blood_glucose']
        category = ('High' if glucose >= 140 else 'Low' if glucose < 70 else 'Normal')
        return f"🍬 Latest blood glucose is {glucose:g} mg/dL — {category} (normal fasting: 70–100 mg/dL)."

    # ── Dashboard summary ──
    if re.search(r'\b(summary|overview|stats|statistics|dashboard|how many)\b', msg):
        return (f"📊 Dashboard Summary:\n"
                f"• Total readings: {summary['total']}\n"
                f"• Anomalies: {summary['anomalies']}\n"
                f"• Tachycardia cases: {summary['tachy']}\n"
                f"• Bradycardia cases: {summary['brady']}\n"
                f"• Avg heart rate: {summary['avg_hr']:g} bpm\n"
                f"• Avg SpO2: {summary['avg_spo2']:g}%")

    # ── Term explanations ──
    if 'tachycardia' in msg:
        return ("Tachycardia means a heart rate above 100 bpm at rest. It can be caused "
                "by exercise, stress, caffeine, fever or, less commonly, an underlying "
                "cardiac issue. Persistent resting tachycardia should be checked by a doctor.")
    if 'bradycardia' in msg:
        return ("Bradycardia means a heart rate below 60 bpm. It's common in athletes and "
                "during sleep, but if paired with dizziness, fatigue or fainting it may "
                "need medical attention.")
    if 'anomaly' in msg or 'isolation forest' in msg:
        return ("HeartSync uses an Isolation Forest model — an unsupervised machine-learning "
                "algorithm — to flag readings whose combination of vitals looks statistically "
                "unusual compared to the rest of the data. These flagged 'outliers' don't "
                "automatically mean something is wrong, but they're worth a closer look.")

    # ── Fallback ──
    return ("I'm not sure I understood that. You can ask me about your heart rate, SpO2, "
            "blood pressure, glucose, stress level, anomalies, or type 'help' to see what "
            "I can do.")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email    = (request.form.get('email') or '').strip()
        password = request.form.get('password') or ''
        confirm  = request.form.get('confirm_password') or ''

        if not username or not password:
            flash('Username and password are required.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        else:
            conn = get_db()
            existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            if existing:
                flash('That username is already taken.', 'error')
                conn.close()
            else:
                conn.execute(
                    'INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
                    (username, email, generate_password_hash(password),
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
                conn.commit()
                conn.close()
                flash('Account created — please log in.', 'success')
                return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))

        flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    records = load_all_readings()
    s = _summary(records)
    hr_analysis     = analyze_heart_rate(records)
    recommendations = generate_recommendations(records, s, hr_analysis)
    return render_template('index.html',
        active_page='dashboard',
        readings=records,
        total_records=s['total'],
        anomaly_count=s['anomalies'],
        tachycardia_count=s['tachy'],
        bradycardia_count=s['brady'],
        avg_heart_rate=s['avg_hr'],
        avg_spo2=s['avg_spo2'],
        hr_analysis=hr_analysis,
        recommendations=recommendations)


@app.route('/add-patient', methods=['GET'])
@login_required
def add_patient_page():
    return render_template('add_patient.html', active_page='add_patient', patient=None)


@app.route('/input-manual', methods=['POST'])
@login_required
def input_manual():
    try:
        patient_id = request.form.get('patient_id', 'P001')
        row = {
            'patient_id':         patient_id,
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

        # ── Save / update patient contact + caretaker info ──
        upsert_patient(
            patient_id,
            full_name=request.form.get('patient_name', ''),
            phone_number=request.form.get('patient_phone', ''),
            caretaker_name=request.form.get('caretaker_name', ''),
            caretaker_phone=request.form.get('caretaker_phone', ''),
            created_by=session.get('user_id'),
        )

        processed = process_dataset(pd.DataFrame([row]))
        conn = get_db()
        inserted_ids = _bulk_insert(conn, processed.to_dict(orient='records'))
        conn.commit()
        conn.close()

        flash(f"Reading for {patient_id} saved successfully!", 'success')

        # ── Check for critical conditions and alert the caretaker ──
        saved_row = processed.to_dict(orient='records')[0]
        alert = trigger_caretaker_alert(saved_row, inserted_ids[0])
        if alert:
            if alert['status'] == 'sent':
                flash(f"Caretaker alert SMS sent to {alert['caretaker_phone']} "
                      f"for {patient_id}: {alert['reasons']}", 'success')
            elif alert['status'] == 'simulated':
                flash(f"Critical reading for {patient_id} ({alert['reasons']}) — "
                      f"SMS alert simulated (Twilio not configured). See Alerts page.", 'error')
            elif alert['status'] == 'skipped':
                flash(f"Critical reading for {patient_id} ({alert['reasons']}) — "
                      f"no caretaker phone number on file. Add one on the Add Patient page.", 'error')
            else:
                flash(f"Critical reading for {patient_id} ({alert['reasons']}) — "
                      f"failed to send caretaker SMS: {alert['error']}", 'error')

    except Exception as e:
        flash(f'Error: {e}', 'error')
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
@login_required
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
        records   = processed.to_dict(orient='records')
        conn = get_db()
        inserted_ids = _bulk_insert(conn, records)
        conn.commit()
        conn.close()
        flash(f'Uploaded {len(df)} records from {f.filename}.', 'success')

        # ── Check each uploaded row for critical conditions ──
        alert_count = 0
        for row, rid in zip(records, inserted_ids):
            alert = trigger_caretaker_alert(row, rid)
            if alert:
                alert_count += 1
        if alert_count:
            flash(f'{alert_count} reading(s) triggered a caretaker alert — see the Alerts page.', 'error')
    except Exception as e:
        flash(f'Error processing file: {e}', 'error')
    return redirect(url_for('index'))


@app.route('/reset')
@login_required
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
@login_required
def delete_reading(rid):
    conn = get_db()
    conn.execute('DELETE FROM readings WHERE id = ?', (rid,))
    conn.commit()
    conn.close()
    flash('Record deleted.', 'success')
    return redirect(url_for('index'))


@app.route('/download-template')
@login_required
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


# ═══════════════════════════════════════════════════════════════════════
# Analytics page — Interactive Charts
# ═══════════════════════════════════════════════════════════════════════
@app.route('/analytics')
@login_required
def analytics():
    records = load_all_readings()
    s = _summary(records)
    hr_analysis     = analyze_heart_rate(records)
    recommendations = generate_recommendations(records, s, hr_analysis)
    dist            = get_distributions(records)
    return render_template('analytics.html',
        active_page='analytics',
        readings=records,
        total_records=s['total'],
        anomaly_count=s['anomalies'],
        tachycardia_count=s['tachy'],
        bradycardia_count=s['brady'],
        avg_heart_rate=s['avg_hr'],
        avg_spo2=s['avg_spo2'],
        hr_analysis=hr_analysis,
        recommendations=recommendations,
        distributions=dist)


# ═══════════════════════════════════════════════════════════════════════
# Caretaker Alerts — log page
# ═══════════════════════════════════════════════════════════════════════
@app.route('/alerts')
@login_required
def alerts_page():
    conn = get_db()
    rows = conn.execute('SELECT * FROM alerts ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('alerts.html', active_page='alerts',
                            alerts=[dict(r) for r in rows])


# ═══════════════════════════════════════════════════════════════════════
# Chatbot API
# ═══════════════════════════════════════════════════════════════════════
@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data    = request.get_json(silent=True) or {}
    message = data.get('message', '')

    records = load_all_readings()
    s = _summary(records)
    hr_analysis     = analyze_heart_rate(records)
    recommendations = generate_recommendations(records, s, hr_analysis)

    try:
        reply = chatbot_reply(message, records, s, hr_analysis, recommendations)
    except Exception as e:
        reply = f"Sorry, I ran into an error processing that: {e}"

    return jsonify(reply=reply)


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5005)
