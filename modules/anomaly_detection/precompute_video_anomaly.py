import os
import cv2
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

from modules.anomaly_detection.prepare_sequences import create_sequences
from modules.anomaly_detection.lstm_autoencoder import LSTMAutoencoder
from modules.anomaly_detection.anomaly_scoring import compute_anomaly_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "visdrone_val.mp4")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "detection", "best.pt")
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_lstm.pt")
WINDOW_SIZE = 10

print("Loading detection model...")
det_model = YOLO(YOLO_MODEL_PATH)

print("Loading anomaly model...")
anomaly_model = LSTMAutoencoder(input_dim=7)
anomaly_model.load_state_dict(torch.load(ANOMALY_MODEL_PATH))
anomaly_model.to(device)
anomaly_model.eval()
cap = cv2.VideoCapture(VIDEO_PATH)

features = []
frame_count = 0

print("Extracting features from video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = det_model(frame, imgsz=512, device=0)

    for r in results:
        num_objects = len(r.boxes)
        avg_conf = float(r.boxes.conf.mean()) if num_objects > 0 else 0
        avg_area = float((r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).mean()) if num_objects > 0 else 0

        # Simple 7-dim feature vector (match your training format)
        feature_vector = [
            num_objects,
            avg_conf,
            avg_area,
            0, 0, 0, 0  # keep same feature size as training
        ]

        features.append(feature_vector)

    frame_count += 1

cap.release()

features = np.array(features)

print("Total frames:", frame_count)
print("Feature shape:", features.shape)

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create sequences
sequences = create_sequences(features_scaled, window_size=WINDOW_SIZE)

print("Sequence shape:", sequences.shape)

# Compute anomaly scores
scores = compute_anomaly_scores(anomaly_model, sequences)

# Pad first WINDOW_SIZE frames with 0 (since no sequence yet)
scores = np.concatenate([np.zeros(WINDOW_SIZE), scores])

np.save(os.path.join(BASE_DIR, "models", "video_anomaly_scores.npy"), scores)

print("Anomaly scores saved.")