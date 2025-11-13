import cv2 as cv
from datetime import datetime, timedelta
import numpy as np
import os
import csv
from datetime import datetime, timedelta
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import sqlite3
import mediapipe as mp
import json
from tensorflow.keras.models import load_model
import cvzone
from flask import Flask, render_template, request, redirect, session, url_for, flash, Response
from functools import wraps
import subprocess
import psutil
from flask_cors import CORS
import json
import sqlite3
import pandas as pd
import queue
import time
import requests

# Global queue to hold attendance data
attendance_queue = queue.Queue()

#configuration
UNKNOWN_DIR = "unknown_faces"
os.makedirs(UNKNOWN_DIR, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.8
UNKNOWN_SAVE_LIMIT = 5
UNKNOWN_COOLDOWN = timedelta(seconds=5)
unknown_last_saved = datetime.min
unknown_face_count = 0
# Keep track of last logged time per user
recently_logged = {}  # {username: datetime}
last_status_messages = {}

#database
def get_attendance_interval():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        return int(config.get("attendance_interval_minutes", 5))
    except:
        return 5

def get_last_logged_time(username):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM attendance WHERE username=? ORDER BY timestamp DESC LIMIT 1", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
    return None

def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Present')''')
    conn.commit()
    conn.close()


# In 1update.py

def mark_attendance_sql(username):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    interval_seconds = get_attendance_interval()
    
    log_status = {}  # This dictionary will contain the final status to send to the web app

    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        cursor.execute('SELECT timestamp FROM attendance WHERE username=? ORDER BY timestamp DESC LIMIT 1', (username,))
        row = cursor.fetchone()

        # Case 1: New log. Log the user and prepare the "Present" message.
        if row is None or now - datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S') > timedelta(minutes=interval_seconds):
            cursor.execute('INSERT INTO attendance (username, timestamp, status) VALUES (?, ?, ?)',
                           (username, timestamp, "Present"))
            print(f"[DB] Logged {username} at {timestamp}")
            log_status = {"name": username, "status": "Present", "timestamp": timestamp}
        
        # Case 2: Already logged. Get the last log time and prepare the "Already Logged" message.
        else:
            last_logged_time = row[0]
            print(f"[DB] {username} already logged recently")
            log_status = {"name": username, "status": "Already Logged", "timestamp": last_logged_time}

        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] {e}")
        log_status = {} # Return an empty dictionary on error
    finally:
        conn.close()
    
    return log_status

def mark_attendance_csv(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, now, "Present"])
    print(f"[CSV] Logged {name} at {now}")

def save_unknown_face(image):
    global unknown_face_count, unknown_last_saved
    now = datetime.now()
    if unknown_face_count < UNKNOWN_SAVE_LIMIT and (now - unknown_last_saved) > UNKNOWN_COOLDOWN:
        unknown_last_saved = now
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        filename = f"unknown_{unknown_face_count+1}_{timestamp}.jpg"
        filepath = os.path.join(UNKNOWN_DIR, filename)
        cv.imwrite(filepath, image)
        print(f"[UNKNOWN] Saved unknown face: {filepath}")
        unknown_face_count += 1

# model loading
print("[INFO] Loading models...")

encoder = pickle.load(open('label_encoder.pkl', 'rb'))
model_svm = pickle.load(open('svm_model_160x160.pkl', 'rb'))
facenet = FaceNet()

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# loading trained model
liveness_model = load_model("liveness_model.h5")

#liveness check
def predict_liveness(face_roi):
    try:
        face = cv.resize(face_roi, (224, 224)) / 255.0
        face = np.expand_dims(face, axis=0)
        pred = liveness_model.predict(face, verbose=0)[0]
        label_idx = np.argmax(pred)  # 0 = spoof 1 = real
        label = "REAL" if label_idx == 1 else "FAKE"
        return label
    except Exception as e:
        print(f"[LIVENESS ERROR] {e}")
        return "REAL"  # assume real if error

#face detect
def detect_faces_mediapipe(frame):
    results = mp_face.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    detections = []
    if results.detections:
        h, w, c = frame.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            x, y = max(0, x), max(0, y)
            detections.append((x, y, width, height))
    return detections

# In 1update.py

# In 1update.py, confirm the send function looks like this:

def send_attendance_to_web(data):
    try:
        requests.post("http://127.0.0.1:5000/log_attendance", json=data)
    except Exception as e:
        print(f"[WEB API ERROR] Failed to send data: {e}")
#main
def main():
    init_db()
    #cap = cv.VideoCapture("http://192.168.0.106:4747/video")
    cap = cv.VideoCapture(0)
    

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
    print("[INFO] Starting Face Recognition + Liveness... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_mediapipe(frame)
        processed_in_frame = set()  # track faces processed in this frame

        for (x, y, w_box, h_box) in faces:
            face_roi = frame[y:y+h_box, x:x+w_box]
            if face_roi.size == 0:
                continue

            # Default values (avoid NameError)
            color = (0, 0, 255)   # Red = default
            label_text = "UNKNOWN"
            name = "Unknown"

            # STEP 1: LIVENESS CHECK
            liveness_label = predict_liveness(face_roi)
            if liveness_label == "FAKE":
                label_text = "FAKE"
                cvzone.cornerRect(frame, (x, y, w_box, h_box), l=20, rt=4,
                                colorC=color, colorR=color)
                cvzone.putTextRect(frame, label_text, (x, y - 15),
                                scale=1.8, thickness=3,
                                colorR=color, colorB=(0, 0, 0), offset=10)
                continue

            # STEP 2: FACE RECOGNITION
            resized_face = cv.resize(face_roi, (160, 160))
            embedding = facenet.embeddings(np.expand_dims(resized_face.astype('float32'), axis=0))
            y_proba = model_svm.predict_proba(embedding)[0]
            max_prob = np.max(y_proba)
            predicted_index = np.argmax(y_proba)

            if max_prob >= CONFIDENCE_THRESHOLD:
                name = encoder.inverse_transform([predicted_index])[0]
                color = (0, 255, 0)   # Green = recognized
                label_text = "REAL"
            else:
                save_unknown_face(cv.cvtColor(resized_face, cv.COLOR_RGB2BGR))
                # skip logging for unknowns
                continue  

            # STEP 3: Attendance Logging
            if name not in processed_in_frame:
                processed_in_frame.add(name)
                last_logged = get_last_logged_time(name)
                now = datetime.now()
                interval_seconds = get_attendance_interval()

                if last_logged is None or (now - last_logged) > timedelta(minutes=interval_seconds):
                    log_status = mark_attendance_sql(name)
                    if log_status:
                        send_attendance_to_web(log_status)   # ✅ send "Present"
                    recently_logged[name] = now
                    last_status_messages[name] = "Present"
                else:
                    log_status = {
                        "name": name,
                        "status": "Already Logged",
                        "timestamp": last_logged.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # ✅ send "Already Logged" to website
                    send_attendance_to_web(log_status)
                    last_status_messages[name] = "Already Logged"


            # STEP 4: Draw box & labels
            cvzone.cornerRect(frame, (x, y, w_box, h_box), l=20, rt=4,
                            colorC=color, colorR=color)
            cvzone.putTextRect(frame, label_text, (x, y - 15),
                            scale=1.8, thickness=3,
                            colorR=color, colorB=(0, 0, 0), offset=10)
            cv.putText(frame, name, (x, y + h_box + 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


        time.sleep(0.5)
        cv.imshow("Face Recognition + Liveness", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()
    print("[INFO] Recognition + Liveness Stopped.")

if __name__ == "__main__":
    main()
