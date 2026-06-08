from ultralytics import YOLO
import torch
import numpy as np


class Detector:
    def __init__(self, model_path="models/yolo11s.pt", device=None):
        self.model = YOLO(model_path)

        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Detector using: {self.device}")

    def detect(self, frame, conf_threshold=0.5):
        """Return list of detections from single frame"""
        results = self.model(frame, verbose=False, device=self.device)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_id": cls,
                    "class_name": self.model.names[cls],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2]
                })

        return detections

    def detect_batch(self, frames, conf_threshold=0.5):
        """Process multiple frames at once"""
        results = self.model(frames, verbose=False, device=self.device)
        all_detections = []

        for result in results:
            frame_dets = []
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                frame_dets.append({
                    "class_id": cls,
                    "class_name": self.model.names[cls],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2]
                })
            all_detections.append(frame_dets)

        return all_detections