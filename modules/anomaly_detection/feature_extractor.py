from ultralytics import YOLO
import numpy as np
import os

class FeatureExtractor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def extract_from_image(self, image_path):
        results = self.model(image_path, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return np.zeros(7)

        cls_ids = boxes.cls.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()

        pedestrian = 0
        vehicles = 0
        bicycle = 0
        motor = 0
        areas = []

        for box, cls in zip(xywh, cls_ids):
            area = box[2] * box[3]
            areas.append(area)

            cls = int(cls)

            if cls == 0:
                pedestrian += 1
            elif cls in [3,4,5,8]:
                vehicles += 1
            elif cls == 2:
                bicycle += 1
            elif cls == 9:
                motor += 1

        total = len(cls_ids)
        avg_area = np.mean(areas) if areas else 0
        density_flag = 1 if total > 20 else 0

        return np.array([
            pedestrian,
            vehicles,
            bicycle,
            motor,
            total,
            avg_area,
            density_flag
        ])