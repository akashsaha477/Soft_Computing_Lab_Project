from ultralytics import YOLO
import torch

class HelmetDetector:
    def __init__(self, model_path="/Users/akashsaha/Downloads/Soft_Computing_Project/Data/yolov8s.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(model_path)

    def detect(self, crop):
        results = self.model(crop, imgsz=320, device=self.device)[0]

        has_helmet = False
        no_helmet = False

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if conf < 0.4:
                continue

            # YOU MUST VERIFY YOUR CLASS IDS HERE
            if cls == 0:
                has_helmet = True
            elif cls == 1:
                no_helmet = True

        return has_helmet, no_helmet