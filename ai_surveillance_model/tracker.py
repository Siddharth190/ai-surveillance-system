import numpy as np


class SimpleTracker:
    """Minimal tracker that matches detections across frames"""

    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.tracks = {}  # track_id -> {bbox, last_seen, class_id}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def update(self, detections):
        """Update tracks with new detections, return current tracks"""
        # Increment lost counter for all existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["lost"] += 1

        # Match detections to existing tracks
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        # Simple greedy matching by IoU
        for det_idx in unmatched_dets[:]:
            det = detections[det_idx]
            best_iou = 0
            best_track = None

            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                iou = self._compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track = track_id

            if best_track is not None:
                matched.append((det_idx, best_track))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(best_track)

        # Update matched tracks
        for det_idx, track_id in matched:
            det = detections[det_idx]
            self.tracks[track_id]["bbox"] = det["bbox"]
            self.tracks[track_id]["center"] = det["center"]
            self.tracks[track_id]["confidence"] = det["confidence"]
            self.tracks[track_id]["lost"] = 0
            self.tracks[track_id]["class_id"] = det["class_id"]
            self.tracks[track_id]["class_name"] = det["class_name"]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks[self.next_id] = {
                "track_id": self.next_id,
                "bbox": det["bbox"],
                "center": det["center"],
                "confidence": det["confidence"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "lost": 0,
                "first_seen": 0  # You can add frame counter
            }
            self.next_id += 1

        # Remove lost tracks
        to_remove = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in to_remove:
            del self.tracks[tid]

        return list(self.tracks.values())