from ultralytics import YOLO
import torch

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(model_path)

    def detect(self, frame, imgsz):
        results = self.model(frame, imgsz=imgsz, device=self.device)[0]

        vehicles = []
        bikes = []
        persons = []

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if conf < 0.4:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # =========================
            # COCO CLASS MAPPING
            # =========================
            # 0: person
            # 1: bicycle
            # 2: car
            # 3: motorcycle
            # 5: bus
            # 7: truck

            if cls in [2, 3, 5, 7]:   # vehicles
                vehicles.append([x1, y1, x2, y2, conf])

            elif cls in [1, 3]:       # bikes (bicycle + motorcycle)
                bikes.append([x1, y1, x2, y2])

            elif cls == 0:            # person
                persons.append([x1, y1, x2, y2])

        return vehicles, bikes, persons