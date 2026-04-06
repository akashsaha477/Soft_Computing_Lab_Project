

# Intelligent Traffic Monitoring System

## Overview

This project implements a real-time traffic monitoring system using computer vision techniques. It processes video feeds to detect vehicles, analyze traffic conditions, identify violations, and extract license plate information. The goal is to simulate a smart traffic management system that can adapt to congestion and assist in enforcement.

The system is modular and designed to run efficiently on standard hardware, including Apple Silicon machines.

---

## Features

### Vehicle Detection and Tracking
- Detects cars, bikes, trucks, and buses using YOLO models
- Assigns IDs to vehicles for basic tracking across frames

### Adaptive Traffic Signal Control
- Traffic signals adjust dynamically based on vehicle count
- Supports red, yellow, and green phases
- Helps simulate congestion-aware traffic flow

### Traffic Violation Detection
- Red light violation detection using a virtual stop line
- Wrong lane detection
- Helmet detection for two-wheelers
- Triple riding detection

### License Plate Recognition
- Hybrid approach combining detection and OCR
- Falls back to heuristic region-based extraction if model fails
- Uses EasyOCR for text recognition
- Filters and formats detected plate text

### Data Logging
- Saves detected plates and violations to a CSV file
- Includes timestamp, plate number, vehicle ID, and violation details

---

## System Pipeline

The processing flow is as follows:

Video Input → Frame Processing → Vehicle Detection → Tracking → Traffic Analysis → License Plate Detection → OCR → Logging and Visualization

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- EasyOCR
- NumPy

---

## Setup Instructions

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Place model files

Create a `Data/` directory inside the project root (if not already present) and place all model files there:

```
Data/
├── yolov8n.pt                  # Vehicle detection model
├── yolov8s.pt                  # Helmet detection model
├── best.pt                     # Custom model (triple riding, etc.)
├── license_plate_detector.pt   # Optional plate detector
```

If your code uses absolute paths, update them in the configuration or pipeline files accordingly.

---

### 3. Add video input

Place your input video inside the `Data/` folder or anywhere convenient. Example:

```
Data/
├── traffic.mp4
```

Then update the video path in `app.py` (or wherever the video source is defined):

```python
video_path = "Data/traffic.mp4"
```

You can also use a webcam by setting:

```python
video_path = 0
```

---

### 4. Run the project

```
python app.py
```

Make sure all paths (models and video) are correctly set before running.

---

---

## Project Structure

```
Code/
├── core/               # Main pipeline and configuration
├── detection/         # Detection modules (vehicle, helmet, etc.)
├── tracking/          # Tracking logic
├── violations/        # Violation detection modules
├── ocr/               # OCR handling
├── database/          # CSV logging
└── utils/             # Utility functions
```

---

## Performance Notes

- YOLO inference is relatively fast and can utilize available hardware acceleration
- EasyOCR runs on CPU for stability
- Performance is improved through frame skipping and selective OCR execution

---

## Limitations

- License plate detection depends heavily on image quality
- Small or blurred plates may not be detected reliably
- Accuracy depends on the quality of trained models

---

## Future Improvements

- Improve license plate detection with a better trained model
- Add temporal smoothing for OCR results
- Integrate a more robust tracking algorithm
- Build a dashboard for visualization and analytics

---

## Author

Akash Saha
122EC0952
Lavanya Tamgade
122EC0354
Electronics and Communication Engineering
NIT Rourkela

---

## Notes

This project is intended as an academic and experimental system. It demonstrates how computer vision can be applied to real-world traffic monitoring and analysis problems.