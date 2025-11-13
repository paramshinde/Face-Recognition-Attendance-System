import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ---------------------- CONFIG ----------------------
IMAGE_DIR = "E:/Final Year-1/images"
EMBEDDING_FILE = "faces_embeddings_done_4classes.npz"
MODEL_FILE = "svm_model_160x160.pkl"
ENCODER_FILE = "label_encoder.pkl"
TARGET_SIZE = (160, 160)

# ---------------------- CLASS ----------------------
class FACELOADER:
    def __init__(self, directory):
        self.directory = directory
        self.detector = MTCNN()
        self.target_size = TARGET_SIZE
        self.X, self.Y = [], []

    def extract_face(self, filepath):
        img = cv.imread(filepath)
        if img is None:
            print(f"[ERROR] Could not load image: {filepath}")
            return None
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)
        if not faces:
            print(f"[WARNING] No face detected: {filepath}")
            return None
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        face_crop = img_rgb[y:y+h, x:x+w]
        return cv.resize(face_crop, self.target_size)

    def load_faces_from_class(self, class_path):
        faces = []
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            face = self.extract_face(img_path)
            if face is not None:
                faces.append(face)
        return faces

    def load_dataset(self):
        for subdir in os.listdir(self.directory):
            class_dir = os.path.join(self.directory, subdir)
            if os.path.isdir(class_dir):
                faces = self.load_faces_from_class(class_dir)
                labels = [subdir] * len(faces)
                print(f"[INFO] Loaded {len(faces)} images for class '{subdir}'")
                self.X.extend(faces)
                self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

# ---------------------- STEP 1: LOAD DATA ----------------------
print("[STEP 1] Loading dataset...")
loader = FACELOADER(IMAGE_DIR)
X_raw, Y_raw = loader.load_dataset()
print(f"[DATASET] Total images: {len(X_raw)}, Classes: {set(Y_raw)}")

# ---------------------- STEP 2: FACE EMBEDDINGS ----------------------
print("[STEP 2] Generating embeddings...")
facenet = FaceNet()
X_raw = np.asarray(X_raw).astype('float32')
embeddings = facenet.embeddings(X_raw)
normalizer = Normalizer(norm='l2')
X = normalizer.transform(embeddings)

# ---------------------- STEP 3: SAVE EMBEDDINGS ----------------------
np.savez_compressed(EMBEDDING_FILE, X=X, Y=Y_raw)
print(f"[SAVED] Embeddings saved to {EMBEDDING_FILE}")

# ---------------------- STEP 4: ENCODE LABELS ----------------------
encoder = LabelEncoder()
Y = encoder.fit_transform(Y_raw)
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(encoder, f)
print(f"[SAVED] Label encoder saved to {ENCODER_FILE}")

# ---------------------- STEP 5: STRATIFIED SPLIT ----------------------
print("[STEP 5] Stratified split for training/testing...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in sss.split(X, Y):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

# ---------------------- STEP 6: TRAIN SVM ----------------------
print("[STEP 6] Training SVM classifier...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# ---------------------- STEP 7: EVALUATION ----------------------
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

train_acc = accuracy_score(Y_train, ypreds_train)
test_acc = accuracy_score(Y_test, ypreds_test)

print(f"[ACCURACY] Train: {train_acc:.2f}, Test: {test_acc:.2f}")
print("[REPORT]\n", classification_report(
    Y_test,
    ypreds_test,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))
print("[CONFUSION MATRIX]\n", confusion_matrix(Y_test, ypreds_test))

# ---------------------- STEP 8: SAVE MODEL ----------------------
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print(f"[SAVED] SVM model saved to {MODEL_FILE}")

# ---------------------- DONE ----------------------
print("[✅ DONE] Face recognition model trained and saved successfully.")
