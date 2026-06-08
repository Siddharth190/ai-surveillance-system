
import numpy as np
import time


class EventAnalyzer:
    def __init__(self):
        self.zone_memory = {}  # track_id -> {zone_id: entered_at, events_sent}
        self.crowd_memory = {}  # zone_id -> is_crowd_active

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        import cv2
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0

    def check_intrusion(self, track, zone):
        """Check if track has intruded into zone"""
        center = track["center"]
        polygon = np.array(zone["polygon"], dtype=np.int32)

        inside = self.point_in_polygon(center, polygon)

        if not inside:
            return None

        track_id = track["track_id"]
        zone_id = zone["id"]

        # Initialize memory for this track-zone pair
        if track_id not in self.zone_memory:
            self.zone_memory[track_id] = {}

        if zone_id not in self.zone_memory[track_id]:
            self.zone_memory[track_id][zone_id] = {
                "entered_at": time.time(),
                "intrusion_sent": False,
                "loitering_sent": False
            }

        memory = self.zone_memory[track_id][zone_id]

        if not memory["intrusion_sent"]:
            memory["intrusion_sent"] = True
            return {
                "event_type": "intrusion",
                "track_id": track_id,
                "zone_id": zone_id,
                "dwell_time": 0
            }

        return None

    def check_loitering(self, track, zone, loitering_seconds=30):
        """Check if track has been in zone too long"""
        center = track["center"]
        polygon = np.array(zone["polygon"], dtype=np.int32)

        inside = self.point_in_polygon(center, polygon)

        if not inside:
            return None

        track_id = track["track_id"]
        zone_id = zone["id"]

        if track_id not in self.zone_memory:
            return None
        if zone_id not in self.zone_memory[track_id]:
            return None

        memory = self.zone_memory[track_id][zone_id]
        dwell_time = time.time() - memory["entered_at"]

        if dwell_time >= loitering_seconds and not memory["loitering_sent"]:
            memory["loitering_sent"] = True
            return {
                "event_type": "loitering",
                "track_id": track_id,
                "zone_id": zone_id,
                "dwell_time": dwell_time
            }

        return None

    def check_crowd(self, tracks, zone, crowd_threshold=2):
        """Check if too many people are in zone"""
        polygon = np.array(zone["polygon"], dtype=np.int32)
        zone_id = zone["id"]

        # Count people (class_id=0) inside zone
        count = 0
        for track in tracks:
            if track["class_id"] == 0:  # person
                center = track["center"]
                if self.point_in_polygon(center, polygon):
                    count += 1

        is_crowd = count >= crowd_threshold

        # Only trigger once per crowd event
        if is_crowd and not self.crowd_memory.get(zone_id, False):
            self.crowd_memory[zone_id] = True
            return {
                "event_type": "crowd",
                "zone_id": zone_id,
                "count": count
            }
        elif not is_crowd:
            self.crowd_memory[zone_id] = False

        return None

    def reset_zone_memory(self, track_id, zone_id=None):
        """Reset memory for a track (useful when track leaves)"""
        if track_id in self.zone_memory:
            if zone_id is None:
                del self.zone_memory[track_id]
            elif zone_id in self.zone_memory[track_id]:
                del self.zone_memory[track_id][zone_id]