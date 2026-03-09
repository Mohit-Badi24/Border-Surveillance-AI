import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Project Modules ===
from modules.anomaly_detection.build_timeseries import build_sequence
from modules.anomaly_detection.prepare_sequences import create_sequences
from modules.anomaly_detection.train_lstm import train_model
from modules.anomaly_detection.anomaly_scoring import compute_anomaly_scores
from modules.anomaly_detection.lstm_autoencoder import LSTMAutoencoder
from modules.threat_engine.threat_scoring import compute_threat_scores


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = "models/detection/best.pt"
IMAGE_FOLDER = "data/visdrone_val"
ANOMALY_MODEL_PATH = "models/anomaly_lstm.pt"
WINDOW_SIZE = 10
EPOCHS = 20


# ============================================================
# STEP 1 — Feature Extraction
# ============================================================

print("Extracting features...")
features_raw = build_sequence(IMAGE_FOLDER, YOLO_MODEL_PATH)
print("Feature shape:", features_raw.shape)


# ============================================================
# STEP 2 — Scaling (ONLY for LSTM)
# ============================================================

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_raw)


# ============================================================
# STEP 3 — Create Sequences
# ============================================================

sequences_scaled = create_sequences(features_scaled, WINDOW_SIZE)
sequences_raw = create_sequences(features_raw, WINDOW_SIZE)

print("Sequence shape:", sequences_scaled.shape)


# ============================================================
# STEP 4 — Load or Train LSTM
# ============================================================

if os.path.exists(ANOMALY_MODEL_PATH):
    print("Loading anomaly model...")
    model = LSTMAutoencoder(input_dim=7)
    model.load_state_dict(torch.load(ANOMALY_MODEL_PATH))
else:
    print("Training anomaly model...")
    model = train_model(sequences_scaled, epochs=EPOCHS)


# ============================================================
# STEP 5 — Compute Anomaly Scores
# ============================================================

scores = compute_anomaly_scores(model, sequences_scaled)

print("Sample anomaly scores:", scores[:10])


# ============================================================
# STEP 6 — Threshold Detection
# ============================================================

mean_score = np.mean(scores)
std_score = np.std(scores)

threshold = mean_score + 1.5 * std_score

anomalies = np.where(scores > threshold)[0]

print("Total anomalies detected:", len(anomalies))


# ============================================================
# STEP 7 — Threat Scoring (USE RAW DATA)
# ============================================================

threat_scores = compute_threat_scores(sequences_raw, scores)

print("Sample threat scores:", threat_scores[:10])


# ============================================================
# STEP 8 — Risk Classification
# ============================================================

risk_labels = []

for score in threat_scores:
    if score < 0.4:
        risk_labels.append("LOW")
    elif score < 0.7:
        risk_labels.append("MEDIUM")
    else:
        risk_labels.append("HIGH")

print("First 10 risk labels:", risk_labels[:10])


# ============================================================
# STEP 9 — Visualization
# ============================================================

plt.figure(figsize=(12,6))
plt.plot(scores, label="Anomaly Score")
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
plt.scatter(anomalies, scores[anomalies], color="red")
plt.title("Anomaly Detection Over Time")
plt.legend()
plt.show()

import os
from PIL import Image

# ============================================================
# STEP 10 — Map HIGH Risk Windows to Frames
# ============================================================

# Get sorted image filenames
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.endswith(".jpg") or f.endswith(".png")
])

print("\nDisplaying HIGH risk frames...\n")

for idx, label in enumerate(risk_labels):
    if label == "HIGH":
        frame_index = idx + WINDOW_SIZE - 1  # last frame in window

        if frame_index < len(image_files):
            frame_name = image_files[frame_index]
            frame_path = os.path.join(IMAGE_FOLDER, frame_name)

            print(f"HIGH RISK → Frame Index: {frame_index}, File: {frame_name}")

            img = Image.open(frame_path)

            plt.figure(figsize=(4,4))
            plt.imshow(img)
            plt.title(f"HIGH RISK: {frame_name}")
            plt.axis("off")
            plt.show()

import json
from datetime import datetime

# ============================================================
# STEP 11 — Group Consecutive HIGH Risk Windows
# ============================================================

high_indices = [i for i, label in enumerate(risk_labels) if label == "HIGH"]

alerts = []
current_group = []

for idx in high_indices:
    if not current_group:
        current_group.append(idx)
    elif idx == current_group[-1] + 1:
        current_group.append(idx)
    else:
        alerts.append(current_group)
        current_group = [idx]

if current_group:
    alerts.append(current_group)

print("\nGenerated Alerts:\n")

alert_data = []

for i, group in enumerate(alerts):
    start_frame = group[0] + WINDOW_SIZE - 1
    end_frame = group[-1] + WINDOW_SIZE - 1
    duration = end_frame - start_frame + 1

    alert_info = {
        "alert_id": f"ALERT_{i+1:03}",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "duration_frames": int(duration),
        "risk_level": "HIGH",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    alert_data.append(alert_info)

    print(alert_info)


# ============================================================
# STEP 12 — Save Alerts to JSON
# ============================================================

os.makedirs("alerts", exist_ok=True)

json_path = "alerts/alert_log.json"

with open(json_path, "w") as f:
    json.dump(alert_data, f, indent=4)

print(f"\nAlerts saved to {json_path}")