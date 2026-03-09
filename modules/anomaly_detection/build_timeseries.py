import os
import numpy as np
from modules.anomaly_detection.feature_extractor import FeatureExtractor

def build_sequence(image_folder, model_path):
    extractor = FeatureExtractor(model_path)

    image_files = sorted(os.listdir(image_folder))
    features = []

    for img in image_files:
        if img.endswith(".jpg") or img.endswith(".png"):
            path = os.path.join(image_folder, img)
            feat = extractor.extract_from_image(path)
            features.append(feat)

    return np.array(features)