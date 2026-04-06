import cv2
from time import time
import string
import csv
import os

import easyocr
import torch

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

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Character mapping tables (used by format_license)
# ---------------------------------------------------------------------------
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}


# ---------------------------------------------------------------------------
# License-plate helpers (ported from working code)
# ---------------------------------------------------------------------------
def license_complies_format(text):
    """
    Accepts plates of exactly 7 alphanumeric characters following the pattern
    AA-00-AAA (two letters, two digits, three letters).  Characters are
    corrected via the mapping tables before the check so that OCR confusions
    (O/0, I/1 …) do not cause false rejects.
    """
    if len(text) != 7:
        return False

    return (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char) and
        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char) and
        (text[2] in '0123456789' or text[2] in dict_char_to_int) and
        (text[3] in '0123456789' or text[3] in dict_char_to_int) and
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char) and
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char) and
        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char)
    )


def format_license(text):
    """
    Correct common OCR character confusions using position-aware mapping.
    Positions 0,1,4,5,6 → letter corrections; positions 2,3 → digit corrections.
    """
    mapping = {
        0: dict_int_to_char, 1: dict_int_to_char,
        2: dict_char_to_int, 3: dict_char_to_int,
        4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
    }
    result = ''
    for i in range(7):
        ch = text[i]
        result += mapping[i].get(ch, ch)
    return result


def get_car(license_plate, vehicle_track_ids):
    """
    Match a detected license-plate bounding box to the enclosing vehicle track.

    Parameters
    ----------
    license_plate : tuple  (x1, y1, x2, y2, score, class_id)
    vehicle_track_ids : list of [x1, y1, x2, y2, track_id]

    Returns
    -------
    (x1, y1, x2, y2, track_id) of matched vehicle, or (-1,-1,-1,-1,-1)
    """
    px1, py1, px2, py2 = license_plate[:4]
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if px1 > xcar1 and py1 > ycar1 and px2 < xcar2 and py2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1


# ---------------------------------------------------------------------------
# EasyOCR initializer with Apple MPS support + CPU fallback
# ---------------------------------------------------------------------------
def init_easyocr_reader(lang_list=('en',), verbose=True):
    import PIL

    # Pillow resampling compatibility shim
    if hasattr(PIL.Image, 'Resampling'):
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
        easyocr.utils.RESAMPLING_METHODS = {
            'NEAREST':   PIL.Image.Resampling.NEAREST,
            'BILINEAR':  PIL.Image.Resampling.BILINEAR,
            'BICUBIC':   PIL.Image.Resampling.BICUBIC,
            'LANCZOS':   PIL.Image.Resampling.LANCZOS,
            'ANTIALIAS': PIL.Image.Resampling.LANCZOS,
        }
    else:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
        easyocr.utils.RESAMPLING_METHODS = {
            'NEAREST':   PIL.Image.NEAREST,
            'BILINEAR':  PIL.Image.BILINEAR,
            'BICUBIC':   PIL.Image.BICUBIC,
            'LANCZOS':   PIL.Image.LANCZOS,
            'ANTIALIAS': PIL.Image.ANTIALIAS,
        }

    use_mps = torch.backends.mps.is_available()
    if verbose:
        print(f"[EasyOCR] Apple MPS available? {use_mps}")

    device = torch.device('mps') if use_mps else torch.device('cpu')

    # Try initialising with gpu=True first (works on newer EasyOCR + MPS)
    reader = None
    if use_mps:
        try:
            reader = easyocr.Reader(list(lang_list), gpu=True, verbose=verbose)
            # Manually move models to MPS (older EasyOCR may not do this automatically)
            if hasattr(reader, 'detector') and reader.detector is not None:
                reader.detector = reader.detector.to(device)
            if hasattr(reader, 'recognizer') and reader.recognizer is not None:
                reader.recognizer = reader.recognizer.to(device)
            reader.device = device
            _ = torch.zeros(1, device=device)  # smoke test
            if verbose:
                print("[EasyOCR] Running on MPS (GPU).")
        except Exception as e:
            if verbose:
                print(f"[EasyOCR] MPS init failed ({e}), falling back to CPU.")
            reader = None

    if reader is None:
        reader = easyocr.Reader(list(lang_list), gpu=False, verbose=verbose)
        device = torch.device('cpu')
        reader.device = device
        if verbose:
            print("[EasyOCR] Running on CPU.")

    return reader, device


# ---------------------------------------------------------------------------
# License-plate CSV helpers
# ---------------------------------------------------------------------------
def init_license_plate_csv(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'license_plate', 'confidence',
                             'vehicle_type', 'track_id', 'violations', 'signal'])
            print(f"[CSV] Created {file_path}")
        else:
            print(f"[CSV] Appending to {file_path}")
    return file_path


def save_license_plate_row(file_path, info: dict):
    violations_str = (
        ";".join(str(v) for v in info.get('violations', []))
        if info.get('violations') else "None"
    )
    try:
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                info.get('timestamp', ''),
                info.get('plate_number', ''),
                f"{float(info.get('confidence', 0)):.2f}",
                info.get('vehicle_type', 'vehicle'),
                info.get('track_id', -1),
                violations_str,
                info.get('signal', ''),
            ])
            f.flush()
        print(f"[CSV] Saved plate: {info.get('plate_number')}")
    except Exception as e:
        print(f"[CSV] Error saving plate: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
class TrafficPipeline:

    # Path to the dedicated license-plate detector weights.
    # Override via config if needed: LP_DETECTOR_PATH = "path/to/license_plate_detector.pt"
    LP_DETECTOR_PATH = getattr(
        __import__('core.config', fromlist=['LP_DETECTOR_PATH']),
        'LP_DETECTOR_PATH',
        'license_plate_detector.pt'
    )
    LP_CSV_PATH = getattr(
        __import__('core.config', fromlist=['LICENSE_PLATE_CSV']),
        'LICENSE_PLATE_CSV',
        'license_plates.csv'
    )

    def __init__(self):
        self.state = SystemState()

        self.detector = VehicleDetector()
        self.tracker = Tracker()
        self.ocr = OCREngine()
        self.cache = OCRCache(OCR_TTL)
        self.logger = CSVLogger(CSV_FILE)

        # Helmet / triple-riding detectors
        self.helmet_detector = HelmetDetector(
            "/Users/akashsaha/Downloads/Soft_Computing_Project/Data/yolov8s.pt"
        )
        self.triple_detector = TripleRidingDetector(
            "/Users/akashsaha/Downloads/Soft_Computing_Project/Data/best.pt"
        )

        # ── License-plate YOLO detector ──────────────────────────────────
        try:
            self.lp_detector = YOLO(self.LP_DETECTOR_PATH)
            print(f"[LP] License-plate detector loaded from {self.LP_DETECTOR_PATH}")
        except Exception as e:
            print(f"[LP] WARNING: Could not load LP detector ({e}). Falling back to EasyOCR-only mode.")
            self.lp_detector = None

        # ── EasyOCR reader (MPS-aware) ────────────────────────────────────
        try:
            self.reader, self.ocr_device = init_easyocr_reader(('en',), verbose=True)
            print(f"[OCR] Reader on device: {self.ocr_device}")
        except Exception as e:
            print(f"[OCR] Fatal: {e}")
            raise

        # ── License-plate CSV ─────────────────────────────────────────────
        self.lp_csv = init_license_plate_csv(self.LP_CSV_PATH)

        # ── Signal state ──────────────────────────────────────────────────
        self.signal_state = "RED"
        self.signal_timer = time()
        self.green_time  = 15
        self.yellow_time = 3
        self.red_time    = 15

        # Plates already logged this session (avoid duplicate rows)
        self.seen_plates = set()
        # Per-track frame counter for OCR cooldown
        self._plate_last_tried = {}

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------
    # Minimum confidence EasyOCR must report before we trust a reading
    OCR_MIN_CONFIDENCE = 0.20

    def _preprocess_plate_crop(self, crop):
        """Upscale to fixed width and apply CLAHE contrast enhancement."""
        crop = crop.copy()
        h, w = crop.shape[:2]
        if w == 0 or h == 0:
            return None
        scale = 300.0 / w
        new_h = max(1, int(h * scale))
        resized = cv2.resize(crop, (300, new_h), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)

    def _is_valid_indian_plate(self, text):
        """
        Validate against common Indian plate formats:
          - Standard:  AA-00-AA-0000   (e.g. WB-02-AB-1234)  → stripped: 10 chars
          - Old style: AA-0000          → stripped: 6 chars
          - New BH:    00-BH-0000-AA    → stripped: 10 chars
        We work with the stripped (no-space, no-hyphen) version.
        All chars must be alphanumeric.
        """
        # Strip everything except alphanumerics
        t = ''.join(c for c in text if c.isalnum())
        if not t:
            return False, t

        # Must be between 6 and 10 characters
        if not (6 <= len(t) <= 10):
            return False, t

        # Reject if all digits or all letters (real plates mix both)
        has_alpha = any(c.isalpha() for c in t)
        has_digit = any(c.isdigit() for c in t)
        if not (has_alpha and has_digit):
            return False, t

        # Reject suspiciously repetitive strings (e.g. "AAAAAAA", "1111111")
        if len(set(t)) <= 2:
            return False, t

        return True, t

    def read_license_plate(self, crop):
        """Run EasyOCR on preprocessed crop. Returns (text, score) or (None, None)."""
        try:
            img = self._preprocess_plate_crop(crop)
            if img is None:
                return None, None

            detections = self.reader.readtext(img, detail=1, paragraph=False)
            if not detections:
                return None, None

            # Individual fragment candidates
            candidates = []
            for _, frag, score in detections:
                cleaned = frag.upper().replace(" ", "").replace("-", "")
                candidates.append((cleaned, score))

            # Merged candidate (handles plates split across two OCR boxes)
            merged = "".join(f.upper().replace(" ", "").replace("-", "") for _, f, _ in detections)
            best_frag_score = max(s for _, _, s in detections)
            candidates.append((merged, best_frag_score))

            best_text, best_score = None, 0.0
            for text, score in candidates:
                if score < self.OCR_MIN_CONFIDENCE:
                    continue
                valid, cleaned = self._is_valid_indian_plate(text)
                if valid and score > best_score:
                    best_score, best_text = score, cleaned

            if best_text:
                print(f"[OCR] Plate: {best_text}  conf={best_score:.2f}")
            return (best_text, best_score) if best_text else (None, None)

        except Exception as e:
            print(f"[OCR] Error: {e}")
        return None, None

    def _detect_plates_yolo(self, frame, tracks):
        """
        Run the LP-YOLO model on the full frame and match detections to tracks.

        Returns
        -------
        dict : { track_id : (plate_text, score, (x1,y1,x2,y2)) }
        """
        results = {}
        if self.lp_detector is None:
            return results

        h, w = frame.shape[:2]
        lp_preds = self.lp_detector(frame)[0]

        for lp in lp_preds.boxes.data.tolist():
            lx1, ly1, lx2, ly2, score, class_id = lp

            # Find which vehicle track contains this plate
            _, _, _, _, track_id = get_car(
                (lx1, ly1, lx2, ly2, score, class_id),
                tracks
            )
            if track_id == -1:
                continue

            # Clamp crop coords
            cx1, cy1 = max(0, int(lx1)), max(0, int(ly1))
            cx2, cy2 = min(w, int(lx2)), min(h, int(ly2))
            if cx2 - cx1 < 20 or cy2 - cy1 < 20:
                continue

            crop = frame[cy1:cy2, cx1:cx2]
            text, conf = self.read_license_plate(crop)

            if text:
                results[track_id] = (text, conf, (cx1, cy1, cx2, cy2))
                self.seen_plates.add(text)

        return results

    # ------------------------------------------------------------------
    # Fallback: heuristic plate region from vehicle bounding box
    # ------------------------------------------------------------------
    def _detect_plates_heuristic(self, frame, tracks):
        """
        When no LP detector is available, crop the bottom 30% of each
        vehicle bounding box and try EasyOCR directly.

        Per-track cooldown: only attempt OCR on a given track once every
        20 frames to avoid hammering EasyOCR with bad/far-away crops.
        """
        results = {}
        h, w = frame.shape[:2]

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            # Per-track cooldown (20-frame gap)
            last = self._plate_last_tried.get(track_id, -999)
            if self.state.frame_id - last < 3:
                continue
            self._plate_last_tried[track_id] = self.state.frame_id

            # Vehicle must be large enough to contain a readable plate
            veh_w = x2 - x1
            veh_h = y2 - y1
            if veh_w < 50 or veh_h < 40:
                continue

            # Crop the bottom 30% of the vehicle box (where plates live)
            px1 = max(0, x1)
            px2 = min(w, x2)
            py1 = max(0, int(y1 + 0.60 * veh_h))
            py2 = min(h, y2)

            if px2 - px1 < 40 or py2 - py1 < 10:
                continue

            crop = frame[py1:py2, px1:px2]
            text, conf = self.read_license_plate(crop)

            if text:
                results[track_id] = (text, conf, (px1, py1, px2, py2))
                # Only add to seen_plates after confirmed log to avoid
                # blocking other vehicles with same plate reading
                self.seen_plates.add(text)

        return results

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        self.state.frame_id += 1

        run_helmet = (self.state.frame_id % 10 == 0)
        run_triple = (self.state.frame_id % 15 == 0)

        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]
        roi_y_offset = int(ROI_START * h)
        roi = frame[roi_y_offset:h, :]

        vehicles, bikes, persons = self.detector.detect(roi, IMG_SIZE)

        if not bikes:
            run_helmet = False
            run_triple = False
        if len(persons) < 2:
            run_triple = False

        def shift_boxes(boxes):
            shifted = []
            for b in boxes:
                x1, y1, x2, y2, *rest = b
                shifted.append([x1, y1 + roi_y_offset, x2, y2 + roi_y_offset, *rest])
            return shifted

        vehicles = shift_boxes(vehicles)
        bikes    = shift_boxes(bikes)
        persons  = shift_boxes(persons)

        # Build fake tracks (bypass unstable tracker)
        tracks = [
            [x1, y1, x2, y2, i]
            for i, (x1, y1, x2, y2, *_) in enumerate(vehicles)
        ]
        vehicle_count = len(tracks)

        # ── Adaptive signal timing ────────────────────────────────────────
        now = time()
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
        if signal == "GREEN":
            remaining = int(self.green_time  - elapsed)
        elif signal == "YELLOW":
            remaining = int(self.yellow_time - elapsed)
        else:
            remaining = int(self.red_time    - elapsed)

        stop_line = int(0.75 * h)

        # ── Helmet / triple-riding detection ─────────────────────────────
        helmet_results = {}
        triple_results = {}

        for bike in bikes:
            x1, y1, x2, y2 = map(int, bike[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = frame[y1:y2, x1:x2]

            if run_helmet:
                has_helmet, no_helmet = self.helmet_detector.detect(crop)
                if no_helmet:
                    helmet_results[(x1, y1, x2, y2)] = "No Helmet"

            if run_triple:
                if self.triple_detector.detect(crop):
                    triple_results[(x1, y1, x2, y2)] = "Triple Riding"

        # ── License-plate detection ───────────────────────────────────────
        if self.lp_detector is not None:
            plate_map = self._detect_plates_yolo(frame, tracks)
        else:
            plate_map = self._detect_plates_heuristic(frame, tracks)

        # Draw plate detections on frame and log new ones
        for track_id, (plate_text, plate_conf, (px1, py1, px2, py2)) in plate_map.items():
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            cv2.putText(frame, plate_text,
                        (px1, py2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        # Fallback drawing when there are no tracks
        if not tracks and vehicles:
            for v in vehicles:
                x1, y1, x2, y2 = map(int, v[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # ── Per-track processing ──────────────────────────────────────────
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
            if v2:
                violations.append(v2)

            for (bx1, by1, bx2, by2), v in helmet_results.items():
                if iou([x1, y1, x2, y2], [bx1, by1, bx2, by2]) > 0.3:
                    violations.append(v)

            for (bx1, by1, bx2, by2), v in triple_results.items():
                if iou([x1, y1, x2, y2], [bx1, by1, bx2, by2]) > 0.3:
                    violations.append(v)

            # Retrieve plate info for this track (if detected this frame)
            plate_info = plate_map.get(track_id)
            plate_text = plate_info[0] if plate_info else None
            plate_conf = plate_info[1] if plate_info else 0.0

            # Log to general CSV logger
            self.logger.add({
                "time":       now,
                "track_id":   track_id,
                "plate":      plate_text,
                "violation":  ", ".join(set(violations)) if violations else None,
                "signal":     signal,
            })

            # Log to dedicated license-plate CSV when a new plate is found
            if plate_text:
                save_license_plate_row(self.lp_csv, {
                    "timestamp":    now,
                    "plate_number": plate_text,
                    "confidence":   plate_conf,
                    "vehicle_type": "vehicle",
                    "track_id":     track_id,
                    "violations":   list(set(violations)),
                    "signal":       signal,
                })

            # Draw vehicle bounding box
            color = (0, 0, 255) if violations else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if violations:
                label = ",".join(set(violations))
                cv2.putText(frame, label,
                            (x1, min(h - 10, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ── Global overlays ───────────────────────────────────────────────
        cv2.line(frame, (0, stop_line), (w, stop_line), (255, 0, 0), 3)

        signal_color = {'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255), 'RED': (0, 0, 255)}[signal]
        cv2.putText(frame, f"Signal: {signal}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 3)
        cv2.putText(frame, f"Time: {remaining}s",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return frame

    def finalize(self):
        self.logger.flush()