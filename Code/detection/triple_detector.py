from ultralytics import YOLO
import torch

class TripleRidingDetector:
    def __init__(self, model_path="best.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(model_path)

    def detect(self, crop):
        results = self.model(crop, imgsz=320, device=self.device)[0]

        triple = False

        for box in results.boxes:
            conf = float(box.conf)
            if conf > 0.5:
                triple = True

        return triple