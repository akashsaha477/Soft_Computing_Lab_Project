def update_signal(vehicle_count, prev_signal, high, low):
    if vehicle_count > high:
        return "GREEN"
    elif vehicle_count < low:
        return "RED"
    return prev_signal