import numpy as np

def compute_threat_scores(sequences, anomaly_scores):

    # Normalize anomaly score (0–1)
    anomaly_norm = (anomaly_scores - anomaly_scores.min()) / (
        anomaly_scores.max() - anomaly_scores.min() + 1e-8
    )

    threat_scores = []

    for i, seq in enumerate(sequences):

        # Extract window statistics
        total_objects = seq[:, 4]      # feature index 4 = total_objects
        vehicles = seq[:, 1]           # feature index 1 = vehicles

        avg_density = np.mean(total_objects)
        avg_vehicles = np.mean(vehicles)

        # Normalize density (simple scaling)
        density_score = avg_density / (avg_density + 10)
        vehicle_score = avg_vehicles / (avg_vehicles + 5)

        risk = (
            0.5 * anomaly_norm[i] +
            0.3 * density_score +
            0.2 * vehicle_score
        )

        threat_scores.append(risk)

    return np.array(threat_scores)