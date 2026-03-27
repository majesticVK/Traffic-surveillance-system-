"""
traffic_detector.py
YOLOv8-based detector specialised for traffic scenes.
Detects vehicles, traffic lights, and extracts plate regions.
"""
from ultralytics import YOLO
import cv2
import numpy as np
import config


class TrafficDetector:
    def __init__(self, conf_threshold=0.30):
        # Load fine-tuned model if available, else fall back to base
        try:
            self.model = YOLO(config.YOLO_TRAFFIC_MODEL)
            print(f"Loaded fine-tuned traffic model: {config.YOLO_TRAFFIC_MODEL}")
        except Exception:
            self.model = YOLO(config.YOLO_MODEL)
            print("Fine-tuned model not found — using base YOLOv8")

        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """Run inference and return structured detections."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls_name,
                    "class_id": cls_id,
                })
        return detections

    def count_vehicles(self, detections):
        return sum(1 for d in detections if d["class"] in config.VEHICLE_CLASSES)

    def get_traffic_lights(self, detections):
        return [d for d in detections if d["class"] == "traffic light"]

    def extract_plate_region(self, frame, det):
        """
        Rough license plate crop — bottom third of a vehicle bounding box.
        Replace with a dedicated plate detector for production use.
        """
        x1, y1, x2, y2 = map(int, det["bbox"])
        plate_y1 = y1 + int((y2 - y1) * 0.65)
        region = frame[plate_y1:y2, x1:x2]
        return region if region.size > 0 else None

    def classify_light_color(self, frame, light_det):
        """
        Simple HSV-based traffic light colour classifier.
        Works for clear dashcam views; replace with a CNN for noisy conditions.
        Returns: 'red', 'yellow', 'green', or 'unknown'
        """
        x1, y1, x2, y2 = map(int, light_det["bbox"])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

        masks = {
            "red":    cv2.inRange(hsv, (0, 120, 100),   (10, 255, 255)) +
                      cv2.inRange(hsv, (160, 120, 100),  (180, 255, 255)),
            "yellow": cv2.inRange(hsv, (15, 100, 100),   (35, 255, 255)),
            "green":  cv2.inRange(hsv, (40, 80, 80),     (90, 255, 255)),
        }

        scores = {k: cv2.countNonZero(v) for k, v in masks.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 50 else "unknown"

    def draw_detections(self, frame, detections, light_states=None):
        """Utility: draw boxes on a copy of the frame for display."""
        out = frame.copy()
        light_states = light_states or {}

        color_map = {
            "car": (100, 200, 255),
            "truck": (255, 180, 50),
            "bus": (255, 100, 100),
            "motorcycle": (180, 255, 100),
            "bicycle": (200, 200, 255),
            "traffic light": (255, 255, 100),
        }

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d["bbox"])
            color = color_map.get(d["class"], (200, 200, 200))
            label = d["class"]
            if d["class"] == "traffic light" and i in light_states:
                label = f"light:{light_states[i]}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out
