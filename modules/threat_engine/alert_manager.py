import json
import os
from datetime import datetime

def save_alert(obj_id, risk_level, frame_index):
    alert = {
        "object_id": obj_id,
        "risk_level": risk_level,
        "frame": frame_index,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs("alerts", exist_ok=True)

    file_path = "alerts/alert_log.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(alert)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print("🚨 ALERT SAVED:", alert)