import os
import cv2
import numpy as np
from ultralytics import YOLO

from modules.motion_engine.motion_analyzer import compute_motion_risk
from modules.threat_engine.alert_manager import save_alert

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "models", "detection", "best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "visdrone_val.mp4")
ANOMALY_PATH = os.path.join(BASE_DIR, "models", "video_anomaly_scores.npy")

# ------------------ Fusion Weights ------------------
MOTION_WEIGHT = 2.0
ANOMALY_WEIGHT = 1.2
ANOMALY_SCALE = 5
ANOMALY_BOOST_IF_MOVING = 1.5

HIGH_THRESHOLD = 6
MEDIUM_THRESHOLD = 3

# ------------------ Border Zone ------------------
ZONE_X1, ZONE_Y1 = 350, 100
ZONE_X2, ZONE_Y2 = 900, 600

# ------------------ Class Threat Weights ------------------
CLASS_THREAT_WEIGHTS = {
    "pedestrian": 1.0,
    "people": 1.0,
    "bicycle": 1.0,
    "motor": 1.2,
    "car": 1.3,
    "van": 1.5,
    "truck": 2.0,
    "bus": 2.0,
    "tricycle": 1.2,
}

# ==========================================================
# INITIALIZATION
# ==========================================================

print("Loading detection model...")
model = YOLO(MODEL_PATH)

print("Loading anomaly scores...")
anomaly_scores = np.load(ANOMALY_PATH)

track_history = {}
frame_index = 0

print("Starting tracking...")

results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    persist=True,
    imgsz=512,
    device=0,
    stream=True
)

for r in results:

    frame = r.orig_img.copy()
    boxes = r.boxes

    # Draw border zone (VISIBLE)
    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 0, 255), 2)
    cv2.putText(frame, "BORDER ZONE", (ZONE_X1, ZONE_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    anomaly_score = anomaly_scores[frame_index] if frame_index < len(anomaly_scores) else 0

    if boxes.id is not None:

        for i, (box, obj_id) in enumerate(zip(boxes.xyxy, boxes.id)):

            obj_id = int(obj_id)
            cls_id = int(boxes.cls[i])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = box.tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            track_history.setdefault(obj_id, []).append(center)

            # ---------------- Motion ----------------
            motion_score = compute_motion_risk(track_history, obj_id)

            # ---------------- Fusion ----------------
            normalized_anomaly = anomaly_score * ANOMALY_SCALE

            motion_component = motion_score * MOTION_WEIGHT
            anomaly_component = normalized_anomaly * ANOMALY_WEIGHT

            if motion_score > 0:
                anomaly_component *= ANOMALY_BOOST_IF_MOVING

            final_score = motion_component + anomaly_component

            # ---------------- Class Weight ----------------
            class_weight = CLASS_THREAT_WEIGHTS.get(class_name, 1.0)
            final_score *= class_weight

            # ---------------- Border Zone Amplification ----------------
            in_border_zone = (
                ZONE_X1 <= center[0] <= ZONE_X2 and
                ZONE_Y1 <= center[1] <= ZONE_Y2
            )

            if in_border_zone:
                final_score *= 1.5

            # ---------------- Classification ----------------
            if final_score > HIGH_THRESHOLD:
                risk_label = "HIGH"
                color = (0, 0, 255)
            elif final_score > MEDIUM_THRESHOLD:
                risk_label = "MEDIUM"
                color = (0, 165, 255)
            else:
                risk_label = "LOW"
                color = (0, 255, 0)

            # ---------------- Draw Bounding Box ----------------
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{class_name} | {risk_label}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if risk_label == "HIGH":
                save_alert(obj_id, risk_label, frame_index)

    cv2.imshow("Border Surveillance AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cv2.destroyAllWindows()
print("Tracking finished.")