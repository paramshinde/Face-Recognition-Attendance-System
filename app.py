from flask import Flask, render_template, request, redirect, session, url_for, flash, Response
from functools import wraps
import subprocess
import psutil
from flask_cors import CORS
import json
...
import json
import sqlite3
import pandas as pd
import queue
import time




attendance_queue = queue.Queue()


app = Flask(__name__)
app.secret_key = 'your_super_secret_key'  # Required for session management

# Dummy user database
import json

USERS_FILE = 'users.json'

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)


# Decorator to protect routes
def login_required(role=None):
    def wrapper(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if role and session.get('role') != role:
                return "Unauthorized", 403
            return f(*args, **kwargs)
        return decorated_function
    return wrapper

@app.route('/')
def index():
    return redirect('/login')

@app.route('/analytics')
@login_required(role='admin')
def analytics():
    return render_template("analytics.html")


@app.route('/analytics_data')
@login_required(role='admin')
def analytics_data():
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql_query("SELECT username, timestamp, status FROM attendance", conn)
    conn.close()

    if df.empty:
        return {"daily": [], "weekly": [], "monthly": [], "pie": {"Present": 0, "Absent": 0}}

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Daily trend
    daily = df.groupby(df['timestamp'].dt.date).size().reset_index(name='count')

    # Weekly trend
    weekly = df.groupby(df['timestamp'].dt.isocalendar().week).size().reset_index(name='count')

    # Monthly trend
# Monthly trend
    monthly = df.groupby(df['timestamp'].dt.to_period("M")).size().reset_index(name='count')
    monthly['month'] = monthly['timestamp'].astype(str)   # convert Period → string
    monthly = monthly[['month', 'count']]

    # Pie chart
    pie_data = df['status'].value_counts().to_dict()

    return {
        "daily": daily.to_dict(orient="records"),
        "weekly": weekly.to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "pie": pie_data
    }


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form['role']

        users = load_users()
        user = users.get(username)

        if user and user['password'] == password and user['role'] == role:
            session['user'] = username
            session['role'] = role
            return redirect(f'/{role}_dashboard')
        else:
            flash("Invalid credentials or role mismatch")
            return redirect('/login')

    return render_template('1login.html')

# In 1app.py, confirm the log_attendance route looks like this:

@app.route('/review_unknowns')
@login_required(role='admin')
def review_unknowns():
    import os
    unknown_dir = "unknown_faces"
    images = []
    if os.path.exists(unknown_dir):
        for img in os.listdir(unknown_dir):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                images.append(img)
    return render_template("review_unknowns.html", images=images)

@app.route('/assign_unknown', methods=['POST'])
@login_required(role='admin')
def assign_unknown():
    import os, shutil

    filename = request.form['filename']
    name = request.form['name'].strip()
    unknown_dir = "unknown_faces"
    images_dir = "E:/Final Year-1/images"  # your training images folder

    src = os.path.join(unknown_dir, filename)
    dst_dir = os.path.join(images_dir, name)
    os.makedirs(dst_dir, exist_ok=True)

    # Move file
    dst = os.path.join(dst_dir, filename)
    shutil.move(src, dst)

    flash(f" Unknown face '{filename}' assigned to {name}.")
    return redirect('/review_unknowns')

from flask import send_from_directory

@app.route('/unknown_faces/<filename>')
@login_required(role='admin')
def unknown_file(filename):
    return send_from_directory('unknown_faces', filename)



@app.route('/log_attendance', methods=['POST'])
def log_attendance():
    data = request.json
    name = data.get('name')
    status = data.get('status')
    timestamp = data.get('timestamp')
    if name and status and timestamp:
        attendance_queue.put({'name': name, 'status': status, 'timestamp': timestamp})
        return "Logged successfully", 200
    return "Invalid data", 400

def generate_attendance_events():
    while True:
        if not attendance_queue.empty():
            data = attendance_queue.get()
            event = f"data: {json.dumps(data)}\n\n"
            yield event
        time.sleep(1)

@app.route('/attendance_stream')
def attendance_stream():
    return Response(generate_attendance_events(), mimetype='text/event-stream')




@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/delete_unknown/<path:filename>', methods=['POST'])
@login_required(role='admin')
def delete_unknown(filename):
    import os
    filepath = os.path.join("unknown_faces", filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        flash(f" Deleted unknown face: {filename}")
    else:
        flash(" File not found.")
    return redirect('/review_unknowns')


@app.route('/register_face', methods=['GET', 'POST'])
@login_required(role='admin')
def register_face():
    if request.method == 'POST':
        name = request.form['name'].strip()
        try:
            subprocess.Popen(['python', '1register_face.py', name])
            flash(f"Registering face for: {name}")
        except Exception as e:
            flash(f"Error starting registration: {e}")
        return redirect('/admin_dashboard')
    return render_template('register_face.html')

@app.route('/delete_all_unknowns', methods=['POST'])
@login_required(role='admin')
def delete_all_unknowns():
    import os, glob
    files = glob.glob(os.path.join("unknown_faces", "*"))
    for f in files:
        try:
            os.remove(f)
        except:
            pass
    flash(" All unknown faces deleted successfully.")
    return redirect('/review_unknowns')


@app.route('/retrain_model')
@login_required(role='admin')
def retrain_model():
    try:
        subprocess.run(['python', '1class.py'], check=True)
        flash("Model retrained successfully!")
    except Exception as e:
        flash(f"Error retraining model: {e}")
    return redirect('/admin_dashboard')


from flask import send_file
import subprocess

@app.route('/export_attendance')
@login_required(role='admin')
def export_attendance():
    try:
        # Run export script
        subprocess.run(['python', '1export_to_excel.py'], check=True)
        flash("Attendance exported successfully!")
    except Exception as e:
        flash(f"Failed to export attendance: {e}")
        return redirect('/admin_dashboard')

    # Send the file to download
    return send_file("attendance.xlsx", as_attachment=True)

@app.route('/manage_users')
@login_required(role='admin')
def manage_users():
    users = load_users()
    return render_template('manage_users.html', users=users)

@app.route('/add_user', methods=['POST'])
@login_required(role='admin')
def add_user():
    username = request.form['username'].strip()
    password = request.form['password'].strip()
    role = request.form['role'].strip()

    users = load_users()
    if username in users:
        flash("User already exists.")
    else:
        users[username] = {"password": password, "role": role}
        save_users(users)
        flash(f"User '{username}' added successfully.")
    return redirect('/manage_users')

@app.route('/delete_user/<username>')
@login_required(role='admin')
def delete_user(username):
    users = load_users()
    if username in users and username != 'admin':
        users.pop(username)
        save_users(users)
        flash(f"User '{username}' deleted.")
    else:
        flash("Cannot delete admin or unknown user.")
    return redirect('/manage_users')

@app.route('/update_user/<username>', methods=['POST'])
@login_required(role='admin')
def update_user(username):
    new_password = request.form['password'].strip()
    new_role = request.form['role'].strip()

    users = load_users()
    if username in users:
        users[username]['password'] = new_password
        users[username]['role'] = new_role
        save_users(users)
        flash(f" User '{username}' updated.")
    return redirect('/manage_users')




@app.route('/teacher_dashboard')
@login_required(role='teacher')
def teacher_dashboard():
    import sqlite3
    name_filter = request.args.get('name', '').strip()
    date_filter = request.args.get('date', '').strip()
    status_filter = request.args.get('status', '').strip()

    query = "SELECT username, timestamp, status FROM attendance WHERE 1=1"
    params = []

    if name_filter:
        query += " AND username LIKE ?"
        params.append(f"%{name_filter}%")
    if date_filter:
        query += " AND DATE(timestamp) = ?"
        params.append(date_filter)
    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()

    return render_template("teacher_dashboard.html", records=records)




@app.route('/export_filtered')
@login_required(role='teacher')
def export_filtered():
    import pandas as pd
    import sqlite3

    name = request.args.get('name', '')
    start = request.args.get('start', '')
    end = request.args.get('end', '')

    conn = sqlite3.connect("attendance.db")
    query = "SELECT * FROM attendance WHERE 1=1"
    params = []

    if name:
        query += " AND username LIKE ?"
        params.append(f"%{name}%")
    if start and end:
        query += " AND DATE(timestamp) BETWEEN ? AND ?"
        params.extend([start, end])

    df = pd.read_sql_query(query, conn, params=params)
    df.to_excel("filtered_attendance.xlsx", index=False)
    conn.close()

    return send_file("filtered_attendance.xlsx", as_attachment=True)






@app.route('/admin_dashboard')
@login_required(role='admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/student_dashboard')
@login_required(role='student')
def student_dashboard():
    import sqlite3
    student_name = session['user']
    date_filter = request.args.get('date', '').strip()

    query = "SELECT * FROM attendance WHERE username = ?"
    params = [student_name]

    if date_filter:
        query += " AND DATE(timestamp) = ?"
        params.append(date_filter)

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(query, params)
    records = cursor.fetchall()
    present_count = len(records)
    conn.close()

    return render_template("student_dashboard.html", records=records, present_count=present_count)

@app.route('/set_interval', methods=['POST'])
@login_required(role='admin')
def set_interval():
    try:
        minutes = int(request.form.get("interval"))
        if minutes < 1 or minutes > 1440:
            flash("Please enter a valid interval between 1 and 1440 minutes.")
            return redirect("/admin_dashboard")

        with open("config.json", "r") as f:
            config = json.load(f)
        config["attendance_interval_minutes"] = minutes
        with open("config.json", "w") as f:
            json.dump(config, f)

        flash(f"Attendance logging interval set to {minutes} minutes.")
    except Exception as e:
        flash(f"Failed to update interval: {e}")

    return redirect("/admin_dashboard")



@app.route('/start_cctv')
@login_required(role='admin')
def start_cctv():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/run_cctv')
@login_required(role='admin')
def run_cctv():
    # Check if 1update.py is already running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'python' in proc.info['name'] and '1update.py' in ' '.join(proc.info['cmdline']):
            flash(" Face Recognition is already running.")
            return redirect('/admin_dashboard')

    try:
        subprocess.Popen(['python', '1update.py'])
        flash(" Face Recognition started successfully!")
    except Exception as e:
        flash(f" Failed to start recognition: {e}")
    return redirect('/admin_dashboard')

@app.route('/stop_cctv')
@login_required(role='admin')
def stop_cctv():
    import psutil
    stopped = False

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and '1update.py' in ' '.join(proc.info['cmdline']):
                proc.terminate()
                stopped = True
                flash(" Face Recognition stopped successfully.")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not stopped:
        flash(" No running Face Recognition process found.")
    
    return redirect('/admin_dashboard')





@app.route('/start_cctv_view')
@login_required(role='admin')
def start_cctv_view():
    return render_template('start_cctv.html')

@app.route('/start_and_view_cctv')
@login_required(role='admin')
def start_and_view_cctv():
    # Check if 1update.py is already running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'python' in proc.info['name'] and '1update.py' in ' '.join(proc.info['cmdline']):
            flash(" Face Recognition is already running.")
            return render_template('live_attendance.html')

    try:
        subprocess.Popen(['python', '1update.py'])
        flash(" Face Recognition started successfully!")
    except Exception as e:
        flash(f" Failed to start recognition: {e}")

    return render_template('live_attendance.html')

@app.route('/stop_cctv_and_redirect')
@login_required(role='admin')
def stop_cctv_and_redirect():
    stopped = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and '1update.py' in ' '.join(proc.info['cmdline']):
                proc.terminate()
                stopped = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if stopped:
        flash(" Face Recognition stopped successfully.")
    else:
        flash(" No running Face Recognition process found.")

    return redirect(url_for('admin_dashboard'))


import webbrowser
import threading

if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)

