from tracking.sort import Sort

class Tracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        return self.tracker.update(detections)