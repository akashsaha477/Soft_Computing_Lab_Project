class SystemState:
    def __init__(self):
        self.frame_id = 0
        self.prev_signal = "RED"
        self.plate_cache = {}
        self.output_buffer = []