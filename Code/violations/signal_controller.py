import time

# Global signal state
_signal_state = "RED"
_signal_timer = time.time()


def update_signal(vehicle_count, prev_signal, high, low):
    """
    Advanced traffic signal controller with timing and hysteresis.
    States: RED → GREEN → YELLOW → RED
    """
    global _signal_state, _signal_timer

    current_time = time.time()

    # GREEN phase
    if _signal_state == "GREEN":
        if current_time - _signal_timer > 20:
            _signal_state = "YELLOW"
            _signal_timer = current_time

    # YELLOW phase
    elif _signal_state == "YELLOW":
        if current_time - _signal_timer > 3:
            _signal_state = "RED"
            _signal_timer = current_time

    # RED phase
    elif _signal_state == "RED":
        if vehicle_count > high:
            _signal_state = "GREEN"
            _signal_timer = current_time

    return _signal_state