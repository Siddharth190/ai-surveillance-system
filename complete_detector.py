from ai_surveillance_model import Detector, SimpleTracker, EventAnalyzer
import cv2
import numpy as np
from collections import deque
import os
import pickle
import threading
import queue
import time


class CCTVStream:
    """Handle CCTV/RTSP stream connections"""

    def __init__(self, source, buffer_size=1, reconnect_delay=5, max_reconnect_attempts=5):
        self.source = source
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_thread = False
        self.thread = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0

    def connect(self):
        """Connect to CCTV stream"""
        try:
            if self.cap is not None:
                self.cap.release()

            if isinstance(self.source, str) and (self.source.startswith("rtsp://") or
                                                 self.source.startswith("rtmp://") or
                                                 self.source.startswith("http://")):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 30

                self.is_connected = True
                self.reconnect_attempts = 0
                print(f"✓ Connected to CCTV: {self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps")
                return True
        except Exception as e:
            print(f"Connection error: {e}")

        self.is_connected = False
        return False

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def start_streaming(self):
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_thread = False
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print("✓ CCTV stream started in background")

    def _stream_loop(self):
        while not self.stop_thread:
            if not self.is_connected:
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    time.sleep(self.reconnect_delay)
                    self.reconnect_attempts += 1
                    print(f"🔄 Reconnecting to CCTV... Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    if self.connect():
                        print("✓ Reconnected to CCTV")
                    else:
                        continue
                else:
                    print("❌ Max reconnection attempts reached")
                    break

            ret, frame = self.read_frame()

            if not ret:
                self.is_connected = False
                print("⚠️ Lost connection to CCTV, attempting to reconnect...")
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        self.release()

    def get_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        self.stop_thread = True
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        self.is_connected = False

    def get_fps(self):
        return self.fps

    def get_resolution(self):
        return self.frame_width, self.frame_height


class CompleteDetector:
    def __init__(self, allowed_direction="right", loitering_seconds=5, crowd_threshold=2,
                 track_vehicles=True, track_people=True):
        self.event_analyzer = EventAnalyzer()
        self.allowed_direction = allowed_direction
        self.loitering_seconds = loitering_seconds
        self.crowd_threshold = crowd_threshold
        self.track_vehicles = track_vehicles
        self.track_people = track_people

        self.track_history = {}
        self.track_history_length = 30
        self.violations = set()
        self.zones = []

        self.VEHICLE_CLASSES = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        self.PERSON_CLASS_ID = 0

    def should_track_class(self, class_id):
        if class_id == self.PERSON_CLASS_ID:
            return self.track_people
        elif class_id in self.VEHICLE_CLASSES:
            return self.track_vehicles
        return False

    def get_class_name(self, class_id):
        if class_id == self.PERSON_CLASS_ID:
            return "person"
        elif class_id in self.VEHICLE_CLASSES:
            return self.VEHICLE_CLASSES[class_id]
        return "object"

    def add_zone(self, polygon_points, zone_id=None, name=None):
        zone = {
            "id": zone_id if zone_id else len(self.zones) + 1,
            "name": name if name else f"Zone {len(self.zones) + 1}",
            "polygon": polygon_points
        }
        self.zones.append(zone)
        return zone

    def load_zones(self, filename="zones.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.zones = pickle.load(f)
            print(f"✓ Loaded {len(self.zones)} zone(s) from {filename}")
            return True
        return False

    def save_zones(self, filename="zones.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.zones, f)
        print(f"✓ Saved {len(self.zones)} zone(s) to {filename}")

    def draw_zones_interactive(self, frame, original_resolution=None):
        """Interactive zone drawing tool with mouse clicks"""
        window_name = "Draw Zones - Interactive"

        # Store original frame for zone coordinates
        original_frame = frame.copy()
        original_height, original_width = original_frame.shape[:2]

        # Create display frame (resized for screen, but coordinates will be scaled back)
        display_frame = frame.copy()
        scale_x = 1.0
        scale_y = 1.0

        # Only resize for display if too large (but keep original coordinates)
        if display_frame.shape[1] > 1280:
            new_width = 1280
            new_height = int(display_frame.shape[0] * (1280 / display_frame.shape[1]))
            display_frame = cv2.resize(display_frame, (new_width, new_height))

        clone = display_frame.copy()
        points = []
        temp_zones = self.zones.copy()

        def scale_to_original(x, y):
            """Convert display coordinates back to original resolution"""
            orig_x = int(x * (original_width / display_frame.shape[1]))
            orig_y = int(y * (original_height / display_frame.shape[0]))
            return orig_x, orig_y

        def scale_to_display(x, y):
            """Convert original coordinates to display resolution"""
            disp_x = int(x * (display_frame.shape[1] / original_width))
            disp_y = int(y * (display_frame.shape[0] / original_height))
            return disp_x, disp_y

        def mouse_callback(event, x, y, flags, param):
            nonlocal points

            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                orig_x, orig_y = scale_to_original(x, y)
                print(f"  Point added - Display: ({x}, {y}) | Original: ({orig_x}, {orig_y})")

            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(points) >= 3:
                    zone_name = input(f"\nEnter name for zone {len(temp_zones) + 1}: ").strip()
                    if not zone_name:
                        zone_name = f"Zone {len(temp_zones) + 1}"

                    original_points = [scale_to_original(x, y) for x, y in points]

                    zone = {
                        "id": len(temp_zones) + 1,
                        "name": zone_name,
                        "polygon": original_points
                    }
                    temp_zones.append(zone)
                    print(f"✓ Zone '{zone_name}' saved with {len(points)} points (original resolution)")
                    points = []
                else:
                    print("  Need at least 3 points for a zone")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_frame.shape[1], display_frame.shape[0])
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "=" * 60)
        print("INTERACTIVE ZONE DRAWING")
        print("=" * 60)
        print(f"Original Video Resolution: {original_width} x {original_height}")
        print(f"Display Resolution: {display_frame.shape[1]} x {display_frame.shape[0]}")
        print("\nInstructions:")
        print("  • LEFT CLICK - Add points around the area you want to monitor")
        print("  • RIGHT CLICK - Finish and save the current zone")
        print("  • Press 'c' - Clear current points")
        print("  • Press 'r' - Reset all zones")
        print("  • Press 's' - Save zones to file")
        print("  • Press 'l' - Load zones from file")
        print("  • Press 'q' - Finish drawing and continue")
        print("=" * 60 + "\n")

        while True:
            display = clone.copy()

            # Draw existing zones
            for zone in temp_zones:
                display_polygon = [scale_to_display(x, y) for x, y in zone["polygon"]]
                pts = np.array(display_polygon, np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                if len(display_polygon) > 0:
                    center_x = sum(p[0] for p in display_polygon) // len(display_polygon)
                    center_y = sum(p[1] for p in display_polygon) // len(display_polygon)
                    cv2.putText(display, zone["name"], (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw current drawing points
            if len(points) > 0:
                for i, pt in enumerate(points):
                    cv2.circle(display, pt, 5, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(display, points[i - 1], pt, (0, 0, 255), 2)

            # Draw instructions
            cv2.putText(display, f"Points: {len(points)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Zones: {len(temp_zones)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Original Res: {original_width}x{original_height}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display, "Left:add point | Right:save zone | C:clear | R:reset | S:save | L:load | Q:quit",
                        (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                points = []
                print("Cleared current points")
            elif key == ord('r'):
                temp_zones = []
                points = []
                print("Reset all zones")
            elif key == ord('s'):
                self.zones = temp_zones
                self.save_zones()
                print("Zones saved!")
            elif key == ord('l'):
                if self.load_zones():
                    temp_zones = self.zones
                    print("Zones loaded!")

        cv2.destroyWindow(window_name)
        self.zones = temp_zones
        return self.zones

    def check_wrong_direction(self, track):
        track_id = track["track_id"]
        center = tuple(track["center"])

        if self.allowed_direction is None:
            return None

        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.track_history_length)

        self.track_history[track_id].append(center)

        if len(self.track_history[track_id]) < 5:
            return None

        positions = list(self.track_history[track_id])
        start = positions[0]
        end = positions[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        movement = abs(dx) + abs(dy)

        if movement < 20:
            return None

        is_wrong = False
        if self.allowed_direction == "right":
            is_wrong = dx < 0
        elif self.allowed_direction == "left":
            is_wrong = dx > 0
        elif self.allowed_direction == "up":
            is_wrong = dy > 0
        elif self.allowed_direction == "down":
            is_wrong = dy < 0

        if is_wrong and track_id not in self.violations:
            self.violations.add(track_id)
            return {
                "type": "wrong_direction",
                "track_id": track_id,
                "class_name": self.get_class_name(track["class_id"]),
                "direction": (dx, dy),
                "movement": movement
            }

        return None

    def check_all_events(self, tracks, frame_count):
        all_events = []

        for track in tracks:
            if not self.should_track_class(track["class_id"]):
                continue

            wrong_dir = self.check_wrong_direction(track)
            if wrong_dir:
                all_events.append(wrong_dir)

            for zone in self.zones:
                intrusion = self.event_analyzer.check_intrusion(track, zone)
                if intrusion:
                    intrusion["type"] = "intrusion"
                    intrusion["frame"] = frame_count
                    intrusion["class_name"] = self.get_class_name(track["class_id"])
                    all_events.append(intrusion)

                loitering = self.event_analyzer.check_loitering(
                    track, zone, self.loitering_seconds
                )
                if loitering:
                    loitering["type"] = "loitering"
                    loitering["frame"] = frame_count
                    loitering["class_name"] = self.get_class_name(track["class_id"])
                    all_events.append(loitering)

        if self.track_people:
            people_tracks = [t for t in tracks if t["class_id"] == self.PERSON_CLASS_ID]
            for zone in self.zones:
                crowd = self.event_analyzer.check_crowd(people_tracks, zone, self.crowd_threshold)
                if crowd:
                    crowd["type"] = "crowd"
                    crowd["frame"] = frame_count
                    all_events.append(crowd)

        return all_events

    def cleanup(self, active_tracks):
        active_ids = {t["track_id"] for t in active_tracks}
        for tid in list(self.track_history.keys()):
            if tid not in active_ids:
                del self.track_history[tid]
                if tid in self.violations:
                    self.violations.remove(tid)

    def reset(self):
        self.track_history.clear()
        self.violations.clear()
        self.event_analyzer.zone_memory.clear()
        self.event_analyzer.crowd_memory.clear()


def list_available_cameras(max_cameras=5):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def main():
    print("=" * 60)
    print("COMPLETE SURVEILLANCE DETECTOR with Vehicle Support")
    print("Detects: People | Cars | Motorcycles | Buses | Trucks")
    print("=" * 60)

    # Configuration
    print("\nConfigure detection parameters:")

    # What to track
    print("\nWhat would you like to track?")
    print("1. People only")
    print("2. Vehicles only (cars, trucks, buses, motorcycles)")
    print("3. Both people and vehicles (recommended)")
    track_choice = input("Choice (1-3) [default 3]: ").strip()

    track_people = True
    track_vehicles = True

    if track_choice == "1":
        track_vehicles = False
        print("✓ Tracking: People only")
    elif track_choice == "2":
        track_people = False
        print("✓ Tracking: Vehicles only")
    else:
        print("✓ Tracking: People and Vehicles")

    # Direction
    print("\nAllowed direction for wrong-direction detection:")
    print("1. Right  2. Left  3. Up  4. Down  5. Disabled")
    dir_choice = input("Choice (1-5): ").strip()
    dir_map = {"1": "right", "2": "left", "3": "up", "4": "down", "5": None}
    allowed_dir = dir_map.get(dir_choice, "right")

    # Loitering seconds
    loiter_sec = input("\nLoitering threshold (seconds, default 5): ").strip()
    loiter_sec = int(loiter_sec) if loiter_sec.isdigit() else 5

    # Crowd threshold
    crowd_thresh = input("\nCrowd threshold (people count, default 2): ").strip()
    crowd_thresh = int(crowd_thresh) if crowd_thresh.isdigit() else 2

    # Confidence threshold
    conf_thresh = input("\nConfidence threshold (0.1-0.9, default 0.3 for better detection): ").strip()
    conf_thresh = float(conf_thresh) if conf_thresh else 0.3
    print(f"✓ Confidence threshold: {conf_thresh}")

    # Model selection
    print("\nSelect YOLO model:")
    print("1. YOLO11n (Nano) - Fastest")
    print("2. YOLO11s (Small) - Fast")
    print("3. YOLO11m (Medium) - Balanced (recommended)")
    print("4. YOLO11l (Large) - Slower")
    print("5. YOLO11x (X-Large) - Best accuracy")

    model_choice = input("Choice (1-5) [default 3]: ").strip()
    model_map = {
        "1": "yolo11n.pt",
        "2": "yolo11s.pt",
        "3": "yolo11m.pt",
        "4": "yolo11l.pt",
        "5": "yolo11x.pt"
    }
    model_path = model_map.get(model_choice, "yolo11m.pt")

    # Video source
    print("\n" + "=" * 60)
    print("VIDEO SOURCE SELECTION")
    print("=" * 60)
    print("1. Video file")
    print("2. Webcam")
    print("3. RTSP Camera (CCTV)")

    source_choice = input("\nChoice (1-3): ").strip()

    source = None
    is_cctv = False
    cctv_stream = None

    if source_choice == "1":
        video_path = input("Enter video path: ").strip('"')
        if not os.path.exists(video_path):
            print(f"Error: File not found")
            return
        source = video_path
        print(f"✓ Selected video file: {video_path}")

    elif source_choice == "2":
        available_cams = list_available_cameras()
        if available_cams:
            print(f"\n✓ Available webcams: {available_cams}")
            cam_index = input(f"Select camera index [default 0]: ").strip()
            source = int(cam_index) if cam_index.isdigit() else 0
        else:
            source = 0
        print(f"✓ Selected webcam {source}")

    elif source_choice == "3":
        print("\nRTSP Camera Setup")
        print("Example formats:")
        print("  • rtsp://username:password@ip:554/stream")
        print("  • rtsp://ip:554/Streaming/Channels/101")

        rtsp_url = input("\nEnter RTSP URL: ").strip()
        source = rtsp_url
        is_cctv = True
        print("✓ RTSP stream configured")

    else:
        print("Invalid choice")
        return

    # Initialize
    print("\nLoading AI models...")
    detector = Detector(model_path=model_path)
    tracker = SimpleTracker()
    complete_detector = CompleteDetector(
        allowed_direction=allowed_dir,
        loitering_seconds=loiter_sec,
        crowd_threshold=crowd_thresh,
        track_vehicles=track_vehicles,
        track_people=track_people
    )

    # Open video source
    if is_cctv:
        print("\nConnecting to CCTV stream...")
        cctv_stream = CCTVStream(source)
        if cctv_stream.connect():
            cctv_stream.start_streaming()
            time.sleep(2)
            ret, first_frame = cctv_stream.get_frame()
            if not ret:
                for _ in range(30):
                    time.sleep(0.1)
                    ret, first_frame = cctv_stream.get_frame()
                    if ret:
                        break
            cap = None
        else:
            print("Error connecting to CCTV stream")
            return
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error opening video source")
            return

    # Get video properties
    if is_cctv and cctv_stream:
        original_width, original_height = cctv_stream.get_resolution()
        fps = cctv_stream.get_fps()
        if original_width == 0 and 'first_frame' in locals() and first_frame is not None:
            original_height, original_width = first_frame.shape[:2]
        total_frames = 0
    else:
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get first frame
    if is_cctv and cctv_stream and 'first_frame' in locals() and first_frame is not None:
        pass
    elif is_cctv and cctv_stream:
        ret, first_frame = cctv_stream.get_frame()
        if not ret:
            for _ in range(30):
                time.sleep(0.1)
                ret, first_frame = cctv_stream.get_frame()
                if ret:
                    break
    else:
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            return

    print(f"\n📹 Video Info:")
    print(f"   Original Resolution: {original_width} x {original_height}")
    print(f"   Display will be scaled for screen, but processing at original resolution")
    print(f"   FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"   Total frames: {total_frames}")

    # Zone configuration
    print("\n" + "=" * 60)
    print("ZONE CONFIGURATION")
    print("=" * 60)

    if os.path.exists("zones.pkl"):
        load_existing = input("\nFound saved zones. Load them? (y/n): ").lower()
        if load_existing == 'y':
            complete_detector.load_zones()

    if len(complete_detector.zones) == 0:
        draw_zones = input("\nDo you want to draw zones interactively? (y/n): ").lower()
        if draw_zones == 'y':
            print("\nDrawing zones on first frame...")
            complete_detector.draw_zones_interactive(first_frame, (original_width, original_height))
        else:
            use_default = input("\nUse default center zone? (y/n): ").lower()
            if use_default == 'y':
                default_zone = [
                    (original_width // 4, original_height // 4),
                    (3 * original_width // 4, original_height // 4),
                    (3 * original_width // 4, 3 * original_height // 4),
                    (original_width // 4, 3 * original_height // 4)
                ]
                complete_detector.add_zone(default_zone, name="Default Zone")
                print(f"✓ Added default zone")

    if not is_cctv and cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\n✓ Ready! FPS: {fps:.1f}")
    print(f"✓ Tracking: {'People ' if track_people else ''}{'Vehicles' if track_vehicles else ''}")
    print(f"✓ Monitoring: Intrusion, Loitering ({loiter_sec}s), Wrong Direction")
    print(f"✓ Active Zones: {len(complete_detector.zones)}")
    print(f"✓ Processing at: {original_width}x{original_height}")
    print("\nControls: Q=Quit, S=Screenshot, R=Reset, Space=Pause, Z=Save Zones\n")

    frame_count = 0
    event_count = 0
    paused = False
    all_events = []
    start_time = time.time()

    # Create window with original aspect ratio
    window_name = "Surveillance - Vehicle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Calculate display size while maintaining aspect ratio (max 1280 width)
    display_width = min(original_width, 1280)
    display_height = int(original_height * (display_width / original_width))
    cv2.resizeWindow(window_name, display_width, display_height)

    try:
        while True:
            if not paused:
                # Read frame
                if is_cctv and cctv_stream:
                    ret, frame = cctv_stream.get_frame()
                    if not ret:
                        time.sleep(0.01)
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n🏁 End of video reached")
                        break

                frame_count += 1

                # Process frame at original resolution
                detections = detector.detect(frame, conf_threshold=conf_thresh)
                tracks = tracker.update(detections)

                # Check events
                events = complete_detector.check_all_events(tracks, frame_count)

                # Process events
                for event in events:
                    event_count += 1
                    all_events.append(event)

                    event_type = event.get("type", "unknown")
                    class_name = event.get("class_name", "object")

                    if event_type == "intrusion":
                        print(
                            f"⚠️ [INTRUSION] Frame {frame_count} | {class_name} {event['track_id']} | Zone {event['zone_id']}")
                    elif event_type == "loitering":
                        print(
                            f"⚠️ [LOITERING] Frame {frame_count} | {class_name} {event['track_id']} | {event['dwell_time']:.1f}s")
                    elif event_type == "crowd":
                        print(f"⚠️ [CROWD] Frame {frame_count} | Zone {event['zone_id']} | {event['count']} people")
                    elif event_type == "wrong_direction":
                        print(f"⚠️ [WRONG DIRECTION] Frame {frame_count} | {class_name} {event['track_id']}")

                # Draw zones (at original resolution)
                for zone in complete_detector.zones:
                    pts = np.array(zone["polygon"], np.int32)
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                    if len(zone["polygon"]) > 0:
                        center_x = sum(p[0] for p in zone["polygon"]) // len(zone["polygon"])
                        center_y = sum(p[1] for p in zone["polygon"]) // len(zone["polygon"])
                        cv2.putText(frame, zone["name"], (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw tracks (at original resolution)
                vehicle_count = 0
                person_count = 0

                for track in tracks:
                    if not complete_detector.should_track_class(track["class_id"]):
                        continue

                    x1, y1, x2, y2 = track["bbox"]
                    track_id = track["track_id"]
                    class_id = track["class_id"]

                    if class_id == 0:
                        color = (0, 255, 0)
                        person_count += 1
                    elif class_id in [2, 3, 5, 7]:
                        color = (255, 165, 0)
                        vehicle_count += 1
                    else:
                        color = (255, 255, 0)

                    is_wrong = track_id in complete_detector.violations
                    if is_wrong:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add label with background
                    label = f"{complete_detector.get_class_name(class_id)} {track_id}"
                    if is_wrong:
                        label += " ⚠️"

                    # Get text size for background
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1 - 5), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw stats (at original resolution)
                cv2.putText(frame, f"Events: {event_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"People: {person_count} | Vehicles: {vehicle_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if is_cctv:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if allowed_dir:
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"Allowed: {allowed_dir.upper()}", (w - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                complete_detector.cleanup(tracks)

                # Create a resized version for display only (preserves original for processing)
                if frame.shape[1] > 1280:
                    scale = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale)
                    display_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    display_frame = frame.copy()

                # Show the resized frame
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved {filename} at original resolution {original_width}x{original_height}")
                elif key == ord('r') or key == ord('R'):
                    event_count = 0
                    all_events.clear()
                    complete_detector.reset()
                    print("🔄 Reset statistics")
                elif key == ord('z') or key == ord('Z'):
                    complete_detector.save_zones()
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")

                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"📊 Processed {frame_count} frames, {event_count} events")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Total events: {event_count}")

    if all_events:
        print("\nEvent breakdown:")
        event_types = {}
        for event in all_events:
            etype = event.get("type", "unknown")
            event_types[etype] = event_types.get(etype, 0) + 1

        for etype, count in event_types.items():
            print(f"  {etype}: {count}")

    # Cleanup
    if is_cctv and cctv_stream:
        cctv_stream.release()
    else:
        cap.release()

    cv2.destroyAllWindows()
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()