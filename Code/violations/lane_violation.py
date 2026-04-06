# Manually tuned lane boundaries (adjust based on your video frame)
LANE_BOUNDARIES = [400, 800]


def get_lane(x, width):
    """
    Determine lane index based on x-coordinate.
    """
    if x < LANE_BOUNDARIES[0]:
        return 0
    elif x < LANE_BOUNDARIES[1]:
        return 1
    return 2


def check_wrong_lane(lane):
    """
    Define wrong lane rule.
    You can customize this based on traffic rules.
    """
    # Example: leftmost lane is restricted
    if lane == 0:
        return "Wrong Lane"
    return None