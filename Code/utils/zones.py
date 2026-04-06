def get_zone(y, h):
    if y > 0.6 * h:
        return "NEAR"
    elif y > 0.3 * h:
        return "MID"
    return "FAR"