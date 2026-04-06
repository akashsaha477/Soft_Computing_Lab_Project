def get_lane(x, width):
    return int(x // (width // 3))

def check_wrong_lane(lane):
    if lane == 0:
        return "Wrong Lane"
    return None