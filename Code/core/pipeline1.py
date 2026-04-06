import cv2
from time import time

# EasyOCR imports
import easyocr
import string

from core.config import *
from core.state import SystemState

from detection.vehicle_detector import VehicleDetector
from detection.helmet_detector import HelmetDetector
from detection.triple_detector import TripleRidingDetector
from tracking.tracker import Tracker
from ocr.ocr_engine import OCREngine, OCRCache
from traffic.signal_controller import update_signal
from violations.red_light import check_red_light
from violations.lane_violation import get_lane, check_wrong_lane
from database.csv_logger import CSVLogger
from utils.geometry import iou


dict_char_to_int = {'O': '0','I': '1','J': '3','A': '4','G': '6','S': '5'}
dict_int_to_char = {'0': 'O','1': 'I','3': 'J','4': 'A','6': 'G','5': 'S'}


def license_complies_format(text):
    if len(text) < 6:
        return False
    return text.isalnum()


def format_license(text):
    return text  # simplified but keeps structure


class TrafficPipeline:

    def __init__(self):
        self.state = SystemState()

        self.detector = VehicleDetector()
        self.tracker = Tracker()
        self.ocr = OCREngine()
        self.cache = OCRCache(OCR_TTL)
        self.logger = CSVLogger(CSV_FILE)

        # Additional detectors (ROI-based)
        self.helmet_detector = HelmetDetector("/Users/akashsaha/Downloads/Soft_Computing_Project/Data/yolov8s.pt")
        self.triple_detector = TripleRidingDetector("/Users/akashsaha/Downloads/Soft_Computing_Project/Data/best.pt")

        # Signal timing state
        self.signal_state = "RED"
        self.signal_timer = time()
        self.green_time = 15
        self.yellow_time = 3
        self.red_time = 15

        # Plate memory (avoid repetition)
        self.seen_plates = set()

        # EasyOCR works best on CPU (avoid MPS tensor mismatch issues)
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("OCR running on CPU (stable mode)")

    def read_plate(self, crop):
        try:
            # Ensure CPU numpy format (fix MPS error)
            crop = crop.copy()
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Contrast enhancement
            gray = cv2.equalizeHist(gray)

            # Adaptive threshold (better than fixed)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            results = self.reader.readtext(thresh)

            for _, text, score in results:
                text = text.upper().replace(' ', '')

                # stricter filtering
                if 6 <= len(text) <= 10 and text.isalnum():
                    return text, score

        except Exception as e:
            print("OCR error:", e)

        return None, None

    def process_frame(self, frame):

        self.state.frame_id += 1

        # Disable frame skipping for debugging (ensures drawing every frame)
        # if self.state.frame_id % FRAME_SKIP != 0:
        #     return frame
        run_helmet = (self.state.frame_id % 10 == 0)
        run_triple = (self.state.frame_id % 15 == 0)

        # Improve FPS by resizing earlier
        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]
        roi_y_offset = int(ROI_START*h)
        roi = frame[roi_y_offset:h, :]
        vehicles, bikes, persons = self.detector.detect(roi, IMG_SIZE)
        # DEBUG: ensure detection is working
        # print("Vehicles detected:", len(vehicles))

        if not bikes:
            run_helmet = False
            run_triple = False

        if len(persons) < 2:
            run_triple = False

        # shift boxes back to original frame
        def shift_boxes(boxes):
            shifted = []
            for b in boxes:
                x1, y1, x2, y2, *rest = b
                shifted.append([x1, y1 + roi_y_offset, x2, y2 + roi_y_offset, *rest])
            return shifted

        vehicles = shift_boxes(vehicles)
        bikes = shift_boxes(bikes)
        persons = shift_boxes(persons)

        import numpy as np

        # TEMP FIX: bypass tracker completely (tracker is unstable)
        tracks = []
        for i, v in enumerate(vehicles):
            x1, y1, x2, y2, *rest = v
            tracks.append([x1, y1, x2, y2, i])  # fake ID

        # DEBUG
        # print("Tracks:", tracks)

        vehicle_count = len(tracks)

        # =========================
        # Adaptive signal timing
        # =========================
        now = time()

        # Adjust GREEN duration based on congestion
        self.green_time = max(10, min(40, int(vehicle_count * 1.5)))

        elapsed = now - self.signal_timer

        if self.signal_state == "GREEN":
            if elapsed >= self.green_time:
                self.signal_state = "YELLOW"
                self.signal_timer = now
        elif self.signal_state == "YELLOW":
            if elapsed >= self.yellow_time:
                self.signal_state = "RED"
                self.signal_timer = now
        elif self.signal_state == "RED":
            if elapsed >= self.red_time:
                self.signal_state = "GREEN"
                self.signal_timer = now

        signal = self.signal_state

        # Remaining time
        if signal == "GREEN":
            remaining = int(self.green_time - elapsed)
        elif signal == "YELLOW":
            remaining = int(self.yellow_time - elapsed)
        else:
            remaining = int(self.red_time - elapsed)

        stop_line = int(0.75 * h)

        helmet_results = {}
        triple_results = {}

        for bike in bikes:
            x1, y1, x2, y2 = map(int, bike)

            # clamp crop
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = frame[y1:y2, x1:x2]

            # Helmet detection (scheduled)
            if run_helmet:
                has_helmet, no_helmet = self.helmet_detector.detect(crop)
                if no_helmet:
                    helmet_results[(x1, y1, x2, y2)] = "No Helmet"

            # Triple riding (scheduled)
            if run_triple:
                is_triple = self.triple_detector.detect(crop)
                if is_triple:
                    triple_results[(x1, y1, x2, y2)] = "Triple Riding"

        # Fallback: draw raw detections if tracking fails
        if not tracks and vehicles:
            for v in vehicles:
                x1, y1, x2, y2, *_ = map(int, v)
                cv2.rectangle(frame, (x1,y1),(x2,y2), (255,255,0), 1)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            lane = get_lane(cx, w)

            violations = []

            if signal == "RED":
                v1 = check_red_light(signal, cy, stop_line)
                if v1:
                    violations.append(v1)

            v2 = check_wrong_lane(lane)
            if v2: violations.append(v2)

            # Attach helmet violation
            for (bx1, by1, bx2, by2), v in helmet_results.items():
                if iou([x1,y1,x2,y2], [bx1,by1,bx2,by2]) > 0.3:
                    violations.append(v)

            # Attach triple riding violation
            for (bx1, by1, bx2, by2), v in triple_results.items():
                if iou([x1,y1,x2,y2], [bx1,by1,bx2,by2]) > 0.3:
                    violations.append(v)

            plate_text = None
            plate_box = None

            # Better plate detection (lower region + EasyOCR)
            if self.state.frame_id % 2 == 0:
                px1 = x1
                px2 = x2
                py1 = int(y1 + 0.3 * (y2 - y1))
                py2 = y2

                py1 = max(0, py1)
                py2 = min(h, py2)
                px1 = max(0, px1)
                px2 = min(w, px2)

                cv2.rectangle(frame, (px1, py1), (px2, py2), (255,0,255), 2)

                if (px2 - px1) > 50 and (py2 - py1) > 25:
                    crop = frame[py1:py2, px1:px2]
                    print("Trying OCR on region")

                    text, score = self.read_plate(crop)

                    if text and text not in self.seen_plates:
                        plate_text = text
                        plate_box = (px1, py1, px2, py2)
                        self.seen_plates.add(text)

            # Draw plate box + text if detected
            if plate_text and plate_box:
                px1, py1, px2, py2 = plate_box

                # Draw plate bounding box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,255), 2)

                # Draw plate text
                cv2.putText(frame, plate_text,
                            (px1, py2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,255), 2)

            self.logger.add({
                "time": time(),
                "track_id": track_id,
                "plate": plate_text,
                "violation": ", ".join(list(set(violations))) if violations else None,
                "signal": signal
            })

            # Draw
            color = (0,255,0) if not violations else (0,0,255)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)

            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}",
                        (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 2)

            # Draw violations label
            if violations:
                label = ",".join(list(set(violations)))
                cv2.putText(frame, label,
                            (x1, min(h-10, y2+20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 2)

        # =========================
        # Global overlays
        # =========================
        # Draw stop line
        cv2.line(frame, (0, stop_line), (w, stop_line), (255, 0, 0), 3)

        # Draw signal (with YELLOW support)
        if signal == "GREEN":
            signal_color = (0,255,0)
        elif signal == "YELLOW":
            signal_color = (0,255,255)
        else:
            signal_color = (0,0,255)

        cv2.putText(frame, f"Signal: {signal}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, signal_color, 3)

        # Draw timer display
        cv2.putText(frame, f"Time: {remaining}s",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

        # Draw vehicle count (move down to avoid overlap)
        cv2.putText(frame, f"Vehicles: {vehicle_count}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,0), 2)

        return frame

    def finalize(self):
        self.logger.flush()