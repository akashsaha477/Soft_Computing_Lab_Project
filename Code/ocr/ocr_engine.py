from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def read(self, img):
        result = self.ocr.ocr(img)
        if result:
            return result[0][0][1][0]
        return None

class OCRCache:
    def __init__(self, ttl):
        self.cache = {}
        self.ttl = ttl

    def get(self, track_id, frame_id):
        if track_id in self.cache:
            last_frame, text = self.cache[track_id]
            if frame_id - last_frame < self.ttl:
                return text
        return None

    def set(self, track_id, frame_id, text):
        self.cache[track_id] = (frame_id, text)