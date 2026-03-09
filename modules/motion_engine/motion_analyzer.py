def compute_motion_risk(track_history, obj_id, border_x=700):
    """
    Compute motion-based risk score for a tracked object.
    """

    history = track_history[obj_id]

    if len(history) < 2:
        return 0

    x_prev, y_prev = history[-2]
    x_curr, y_curr = history[-1]

    # Velocity
    velocity = ((x_curr - x_prev)**2 + (y_curr - y_prev)**2)**0.5

    # Direction shift
    dx = x_curr - x_prev

    risk_score = 0

    if velocity > 20:
        risk_score += 1

    if abs(dx) > 50:
        risk_score += 1

    if x_prev < border_x and x_curr >= border_x:
        risk_score += 2

    return risk_score