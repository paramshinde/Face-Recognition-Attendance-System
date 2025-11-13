# Additional backend logic for Admin Dashboard enhancements
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import json, os, time, shutil, sqlite3, csv
import psutil
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'

# ---- Load Config Files ----
USERS_FILE = "users.json"
NOTIFY_FILE = "notification.json"
UNKNOWN_DIR = os.path.join("static", "unknown_faces")
PROXY_LOG = "proxy_log.csv"
UNKNOWN_LOG = "unknown_log.csv"
TRAIN_LOG = "train_log.txt"

# ---- Load Users from JSON ----
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ---- Notification Settings ----
def load_notifications():
    if os.path.exists(NOTIFY_FILE):
        with open(NOTIFY_FILE) as f:
            return json.load(f)
    return {"notify_spoof": True, "notify_phone": True}

def save_notifications(settings):
    with open(NOTIFY_FILE, "w") as f:
        json.dump(settings, f, indent=4)

# ---- System Status ----
def get_system_status():
    boot_time = psutil.boot_time()
    face_rec_running = False

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if isinstance(cmdline, list) and '1update.py' in ' '.join(cmdline):
                face_rec_running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue

    return {
        "uptime": round(time.time() - boot_time),
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "running": face_rec_running
    }


# ---- Unknown Faces ----
def load_unknown_faces():
    faces = []
    if os.path.exists(UNKNOWN_LOG):
        with open(UNKNOWN_LOG) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    faces.append({"timestamp": row[0], "filename": row[1]})
    return faces

# ---- Training Logs ----
def load_train_logs():
    if os.path.exists(TRAIN_LOG):
        with open(TRAIN_LOG) as f:
            return f.read()
    return "No training logs available."

# ---- Download Logs ZIP ----
@app.route("/download_logs")
def download_logs():
    shutil.make_archive("logs_backup", "zip", root_dir=".", base_dir=".", verbose=0,
                        logger=None, dry_run=False, files=[PROXY_LOG, UNKNOWN_LOG])
    return send_file("logs_backup.zip", as_attachment=True)

# ---- Add User ----
@app.route("/add_user", methods=["POST"])
def add_user():
    username = request.form["username"].strip()
    password = request.form["password"].strip()
    role = request.form["role"].strip()

    users = load_users()
    if username in users:
        flash("⚠️ User already exists.")
    else:
        users[username] = {"password": password, "role": role}
        save_users(users)
        flash(f"✅ Added user: {username}")
    return redirect("/admin_dashboard")

# ---- Delete User ----
@app.route("/delete_user/<username>")
def delete_user(username):
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        flash(f"🗑 Deleted user: {username}")
    else:
        flash("⚠️ User not found.")
    return redirect("/admin_dashboard")

# ---- Set Notification Settings ----
@app.route("/set_notifications", methods=["POST"])
def set_notifications():
    settings = {
        "notify_spoof": 'notify_spoof' in request.form,
        "notify_phone": 'notify_phone' in request.form
    }
    save_notifications(settings)
    flash("🔔 Notification settings updated.")
    return redirect("/admin_dashboard")

# ---- Admin Dashboard Route ----
@app.route("/admin_dashboard")
def admin_dashboard():
    users_json = load_users()
    users_list = [{"username": u, "role": users_json[u]["role"]} for u in users_json]
    unknown_faces = load_unknown_faces()
    training_logs = load_train_logs()
    system_status = get_system_status()
    notifications = load_notifications()

    return render_template("admin_dashboard.html",
        users=users_list,
        unknown_faces=unknown_faces,
        training_logs=training_logs,
        system_status=system_status,
        notifications=notifications
    )
