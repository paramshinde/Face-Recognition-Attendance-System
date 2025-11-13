import cv2 as cv
import os
import sys
import time
from mtcnn import MTCNN

# Config
SAVE_DIR = "E:/Final Year-1/images"
IMG_SIZE = (160, 160)
NUM_IMAGES = 20
detector = MTCNN()

if len(sys.argv) < 2:
    print("Usage: python 1register_face.py <name>")
    exit(1)

name = sys.argv[1].strip()
save_path = os.path.join(SAVE_DIR, name)
os.makedirs(save_path, exist_ok=True)

cap = cv.VideoCapture(0)
count = 0
print("[INFO] Capturing face images... Press 'q' to quit.")

while cap.isOpened() and count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera failure.")
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if faces:
        # Take largest face only
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_resized = cv.resize(face_img, IMG_SIZE)
        count += 1
        img_path = os.path.join(save_path, f"{name}-{count}.jpg")
        cv.imwrite(img_path, face_resized)
        print(f"[SAVED] {img_path}")

        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, f"Captured {count}/{NUM_IMAGES}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        time.sleep(0.5)  # small delay

    cv.imshow("Register New Face", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print(f"[DONE] Collected {count}/{NUM_IMAGES} images for {name}")
