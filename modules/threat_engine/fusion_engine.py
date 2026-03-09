def fuse_risk(motion_score, anomaly_score):
    """
    Combine motion risk and anomaly score into final threat level.
    """

    final_score = 0.6 * motion_score + 0.4 * anomaly_score

    if final_score >= 2:
        return "HIGH"
    elif final_score >= 1:
        return "MEDIUM"
    else:
        return "LOW"