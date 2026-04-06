def check_red_light(signal, center_y, stop_line):
    if signal == "RED" and center_y > stop_line:
        return "Red Light Jump"
    return None