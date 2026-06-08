import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pickle
from collections import deque
from PIL import Image
import time
import threading
import queue
import urllib.request
from datetime import datetime
import shutil

# Import your surveillance model
from ai_surveillance_model import Detector, SimpleTracker, EventAnalyzer

# Set page config
st.set_page_config(
    page_title="AI Surveillance Dashboard - Large File Support",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        font-size: 16px;
    }
    .event-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .intrusion { background-color: #ff4444; color: white; }
    .loitering { background-color: #ff8800; color: white; }
    .crowd { background-color: #ff00ff; color: white; }
    .wrong-direction { background-color: #ff0000; color: white; }
    .vehicle-detected { background-color: #ff8800; color: white; }
    .person-detected { background-color: #00ff00; color: black; }
    .cctv-status-connected { color: #00ff00; }
    .cctv-status-disconnected { color: #ff0000; }
    .upload-warning { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


class LargeFileHandler:
    """Handle large video files efficiently"""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="surveillance_video_")

    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to disk without loading into memory"""
        if uploaded_file is None:
            return None

        # Get file extension
        file_ext = os.path.splitext(uploaded_file.name)[1]
        if not file_ext:
            file_ext = '.mp4'

        # Create temporary file path
        temp_path = os.path.join(self.temp_dir, f"video_{int(time.time())}{file_ext}")

        # Save file in chunks
        with open(temp_path, 'wb') as f:
            # Get file size for progress bar
            file_size = uploaded_file.size
            bytes_written = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            progress_bar = st.progress(0, text="Uploading large video file...")
            status_text = st.empty()

            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
                progress = bytes_written / file_size
                progress_bar.progress(progress)
                status_text.text(
                    f"Uploading: {bytes_written / (1024 * 1024):.1f} MB / {file_size / (1024 * 1024):.1f} MB")

            progress_bar.empty()
            status_text.empty()

        return temp_path

    def cleanup(self):
        """Remove temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")


class CCTVStreamHandler:
    """Handle CCTV/RTSP streams in background thread for Streamlit"""

    def __init__(self, source, max_reconnect_attempts=5):
        self.source = source
        self.max_reconnect_attempts = max_reconnect_attempts
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_thread = False
        self.thread = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0
        self.last_frame_time = 0
        self.connection_status = "Disconnected"

    def connect(self):
        """Connect to stream"""
        try:
            if self.cap is not None:
                self.cap.release()

            # Use FFMPEG for RTSP streams
            if isinstance(self.source, str) and self.source.startswith("rtsp://"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                self.connection_status = "Connected"
                return True
        except Exception as e:
            st.warning(f"Connection error: {e}")

        self.is_connected = False
        self.connection_status = "Disconnected"
        return False

    def start_streaming(self):
        """Start background streaming thread"""
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_thread = False
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def _stream_loop(self):
        """Background thread for continuous streaming"""
        while not self.stop_thread:
            if not self.is_connected:
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    time.sleep(3)
                    self.reconnect_attempts += 1
                    if self.connect():
                        continue
                else:
                    time.sleep(1)
                    continue

            ret, frame = self.cap.read()

            if not ret:
                self.is_connected = False
                self.connection_status = "Reconnecting..."
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
                self.last_frame_time = time.time()
            except queue.Full:
                pass

        if self.cap:
            self.cap.release()

    def get_frame(self):
        """Get latest frame"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        """Release stream"""
        self.stop_thread = True
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def get_resolution(self):
        return self.frame_width, self.frame_height

    def get_fps(self):
        return self.fps


class SurveillanceUI:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.detector_initialized = False
        self.events_list = []
        self.frame_count = 0
        self.violation_count = 0
        self.cctv_stream = None
        self.is_live = False

        # Vehicle tracking
        self.track_history = {}
        self.track_history_length = 30
        self.violations = set()

        # COCO class IDs for vehicles
        self.VEHICLE_CLASSES = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        self.PERSON_CLASS_ID = 0

    def should_track_class(self, class_id, track_vehicles, track_people):
        """Check if we should track this class"""
        if class_id == self.PERSON_CLASS_ID:
            return track_people
        elif class_id in self.VEHICLE_CLASSES:
            return track_vehicles
        return False

    def get_class_name(self, class_id):
        """Get class name for display"""
        if class_id == self.PERSON_CLASS_ID:
            return "person"
        elif class_id in self.VEHICLE_CLASSES:
            return self.VEHICLE_CLASSES[class_id]
        return "object"

    def initialize_detector(self, model_name="yolo11m.pt"):
        """Initialize YOLO detector"""
        with st.spinner(f"Loading {model_name}..."):
            self.detector = Detector(model_path=model_name)
            self.tracker = SimpleTracker()
            self.detector_initialized = True
        st.success(f"✅ Model loaded: {model_name}")

    def check_wrong_direction(self, track, allowed_direction):
        """Check if track is moving wrong direction"""
        if allowed_direction == "disabled":
            return None

        track_id = track["track_id"]
        center = tuple(track["center"])

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
        if allowed_direction == "right":
            is_wrong = dx < 0
        elif allowed_direction == "left":
            is_wrong = dx > 0
        elif allowed_direction == "up":
            is_wrong = dy > 0
        elif allowed_direction == "down":
            is_wrong = dy < 0

        if is_wrong and track_id not in self.violations:
            self.violations.add(track_id)
            return {
                "type": "wrong_direction",
                "track_id": track_id,
                "class_name": self.get_class_name(track["class_id"]),
                "movement": movement
            }
        return None

    def process_frame(self, frame, zones, allowed_direction, loitering_seconds,
                      crowd_threshold, track_vehicles=True, track_people=True, conf_threshold=0.3):
        """Process a single frame with vehicle and people detection"""
        if not self.detector_initialized:
            return frame, [], 0, 0, 0

        # Detect objects with lower confidence for better detection
        detections = self.detector.detect(frame, conf_threshold=conf_threshold)

        # Update tracker
        tracks = self.tracker.update(detections)

        # Filter tracks we want to track
        relevant_tracks = [t for t in tracks if self.should_track_class(t["class_id"], track_vehicles, track_people)]

        # Check events
        events = []
        wrong_direction_violations = set()
        person_count = 0
        vehicle_count = 0

        # Check wrong direction for all relevant tracks
        for track in relevant_tracks:
            # Count people and vehicles
            if track["class_id"] == self.PERSON_CLASS_ID:
                person_count += 1
            elif track["class_id"] in self.VEHICLE_CLASSES:
                vehicle_count += 1

            # Check wrong direction
            wrong_dir = self.check_wrong_direction(track, allowed_direction)
            if wrong_dir:
                wrong_direction_violations.add(track["track_id"])
                events.append(wrong_dir)

        # Check zone-based events
        event_analyzer = EventAnalyzer()
        for zone in zones:
            for track in relevant_tracks:
                # Intrusion detection
                intrusion = event_analyzer.check_intrusion(track, zone)
                if intrusion:
                    events.append({
                        "type": "intrusion",
                        "track_id": track["track_id"],
                        "zone_id": zone["id"],
                        "class_name": self.get_class_name(track["class_id"]),
                        "frame": self.frame_count
                    })

                # Loitering detection (only for people)
                if track["class_id"] == self.PERSON_CLASS_ID:
                    loitering = event_analyzer.check_loitering(track, zone, loitering_seconds)
                    if loitering:
                        events.append({
                            "type": "loitering",
                            "track_id": track["track_id"],
                            "zone_id": zone["id"],
                            "class_name": "person",
                            "dwell_time": loitering["dwell_time"],
                            "frame": self.frame_count
                        })

            # Crowd detection (only for people)
            people_tracks = [t for t in relevant_tracks if t["class_id"] == self.PERSON_CLASS_ID]
            crowd = event_analyzer.check_crowd(people_tracks, zone, crowd_threshold)
            if crowd:
                events.append({
                    "type": "crowd",
                    "zone_id": zone["id"],
                    "count": crowd["count"],
                    "frame": self.frame_count
                })

        # Draw zones
        for zone in zones:
            pts = np.array(zone["polygon"], np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            if len(zone["polygon"]) > 0:
                center_x = sum(p[0] for p in zone["polygon"]) // len(zone["polygon"])
                center_y = sum(p[1] for p in zone["polygon"]) // len(zone["polygon"])
                cv2.putText(frame, zone["name"], (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw tracks with different colors for different classes
        for track in relevant_tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["track_id"]
            class_id = track["class_id"]

            # Different colors for different object types
            if class_id == self.PERSON_CLASS_ID:
                color = (0, 255, 0)  # Green for people
            elif class_id in self.VEHICLE_CLASSES:
                color = (255, 165, 0)  # Orange for vehicles
            else:
                color = (255, 255, 0)  # Yellow for others

            is_wrong = track_id in wrong_direction_violations
            if is_wrong:
                color = (0, 0, 255)  # Red for violations

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label with class name and ID
            label = f"{self.get_class_name(class_id)} {track_id}"
            if is_wrong:
                label += " ⚠️"

            # Add background for text
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw stats
        cv2.putText(frame, f"People: {person_count} | Vehicles: {vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Events: {len(events)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if allowed_direction != "disabled":
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Allowed: {allowed_direction.upper()}", (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Cleanup old track history
        active_ids = {t["track_id"] for t in relevant_tracks}
        for tid in list(self.track_history.keys()):
            if tid not in active_ids:
                del self.track_history[tid]
                if tid in self.violations:
                    self.violations.remove(tid)

        return frame, events, len(relevant_tracks), person_count, vehicle_count

    def start_cctv_stream(self, source):
        """Start CCTV stream"""
        self.cctv_stream = CCTVStreamHandler(source)
        if self.cctv_stream.connect():
            self.cctv_stream.start_streaming()
            self.is_live = True
            return True
        return False

    def stop_cctv_stream(self):
        """Stop CCTV stream"""
        if self.cctv_stream:
            self.cctv_stream.release()
            self.cctv_stream = None
        self.is_live = False

    def reset(self):
        """Reset tracking data"""
        self.track_history.clear()
        self.violations.clear()
        self.frame_count = 0
        self.events_list = []


def main():
    st.title("🎥 AI Surveillance Dashboard - Large File Support")
    st.markdown("Real-time monitoring: People | Cars | Motorcycles | Buses | Trucks")
    st.markdown("Detects: Intrusion | Loitering | Crowd | Wrong Direction")
    st.markdown("---")

    # Show upload limit info
    st.info("💡 **Large File Support**: Videos up to 10GB+ are supported using chunked upload and streaming processing")

    # Initialize session state
    if "ui" not in st.session_state:
        st.session_state.ui = SurveillanceUI()
    if "zones" not in st.session_state:
        st.session_state.zones = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "events" not in st.session_state:
        st.session_state.events = []
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "file_handler" not in st.session_state:
        st.session_state.file_handler = LargeFileHandler()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model_option = st.selectbox(
            "YOLO Model",
            ["yolo11n.pt (Fastest)", "yolo11s.pt (Fast)", "yolo11m.pt (Balanced)",
             "yolo11l.pt (Accurate)", "yolo11x.pt (Most Accurate)"]
        )
        model_map = {
            "yolo11n.pt (Fastest)": "yolo11n.pt",
            "yolo11s.pt (Fast)": "yolo11s.pt",
            "yolo11m.pt (Balanced)": "yolo11m.pt",
            "yolo11l.pt (Accurate)": "yolo11l.pt",
            "yolo11x.pt (Most Accurate)": "yolo11x.pt"
        }
        model_name = model_map[model_option]

        if st.button("🚀 Load Model", type="primary"):
            st.session_state.ui.initialize_detector(model_name)

        st.markdown("---")

        # What to track
        st.subheader("🎯 Track Objects")
        track_people = st.checkbox("Track People", value=True)
        track_vehicles = st.checkbox("Track Vehicles (cars, trucks, buses, motorcycles)", value=True)

        st.markdown("---")

        # Detection settings
        st.subheader("⚡ Detection Settings")

        allowed_direction = st.selectbox(
            "Allowed Direction (Wrong Direction Detection)",
            ["disabled", "right", "left", "up", "down"]
        )

        loitering_seconds = st.slider(
            "Loitering Threshold (seconds)",
            min_value=1, max_value=60, value=5
        )

        crowd_threshold = st.slider(
            "Crowd Threshold (people)",
            min_value=1, max_value=10, value=2
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9, value=0.3, step=0.05,
            help="Lower values detect more objects but may increase false positives"
        )

        # Frame skipping for large files
        frame_skip = st.slider(
            "Frame Processing Skip",
            min_value=1, max_value=10, value=2,
            help="Process every Nth frame. Higher values = faster processing for large files"
        )

        st.markdown("---")

        # Zone configuration
        st.subheader("🗺️ Zone Configuration")

        zone_name = st.text_input("Zone Name", "Restricted Area")
        zone_points = st.text_area(
            "Polygon Points (x1,y1 x2,y2 x3,y3 ...)",
            placeholder="100,100 500,100 500,400 100,400",
            help="Enter coordinates in original video resolution"
        )

        if st.button("➕ Add Zone"):
            if zone_points:
                try:
                    points = []
                    for point in zone_points.split():
                        x, y = map(int, point.split(','))
                        points.append((x, y))
                    if len(points) >= 3:
                        new_zone = {
                            "id": len(st.session_state.zones) + 1,
                            "name": zone_name,
                            "polygon": points
                        }
                        st.session_state.zones.append(new_zone)
                        st.success(f"✓ Added zone: {zone_name}")
                    else:
                        st.error("Need at least 3 points")
                except Exception as e:
                    st.error(f"Invalid format: {e}")

        # Display existing zones
        if st.session_state.zones:
            st.subheader("📋 Current Zones")
            for zone in st.session_state.zones:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{zone['name']}** ({len(zone['polygon'])} points)")
                with col2:
                    if st.button(f"🗑️", key=f"del_{zone['id']}"):
                        st.session_state.zones.remove(zone)
                        st.rerun()

        # Save/Load zones
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Zones"):
                with open("zones_ui.pkl", "wb") as f:
                    pickle.dump(st.session_state.zones, f)
                st.success("Zones saved!")
        with col2:
            if st.button("📂 Load Zones"):
                if os.path.exists("zones_ui.pkl"):
                    with open("zones_ui.pkl", "rb") as f:
                        st.session_state.zones = pickle.load(f)
                    st.success(f"Loaded {len(st.session_state.zones)} zones")

        st.markdown("---")

        # Video source
        st.subheader("📹 Video Source")

        source_type = st.radio(
            "Source Type",
            ["Video File", "Webcam", "RTSP Camera", "HTTP/MJPEG Stream", "RTMP Stream"]
        )

        video_file = None
        rtsp_url = None
        http_url = None
        rtmp_url = None
        video_path = None

        if source_type == "Video File":
            st.markdown('<div class="upload-warning">📁 Supports large files (10GB+) - Upload will be chunked</div>',
                        unsafe_allow_html=True)
            video_file = st.file_uploader(
                "Choose video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'm4v', 'wmv', 'flv', 'webm'],
                help="Large files are supported via chunked upload"
            )

            # Option for local file path
            use_local_path = st.checkbox("Use local file path instead of upload",
                                         help="For very large files, place them in a known directory and provide the path")
            if use_local_path:
                video_path = st.text_input("Local video file path",
                                           placeholder="C:/Videos/surveillance.mp4 or /home/user/video.mp4")
                if video_path and os.path.exists(video_path):
                    file_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)  # GB
                    st.success(f"✅ Found video file ({file_size:.2f} GB)")
                elif video_path:
                    st.error("File not found. Please check the path.")

        elif source_type == "RTSP Camera":
            st.info(
                "RTSP URL formats:\n- Hikvision: rtsp://username:password@ip:554/Streaming/Channels/101\n- Dahua: rtsp://username:password@ip:554/cam/realmonitor?channel=1&subtype=0")
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:554/stream")

            if rtsp_url and st.button("🔍 Test Connection"):
                with st.spinner("Testing connection..."):
                    test_cap = cv2.VideoCapture(rtsp_url)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        test_cap.release()
                        if ret:
                            st.success("✅ Connection successful!")
                        else:
                            st.error("❌ Connection failed - Cannot read frame")
                    else:
                        st.error("❌ Connection failed")
        elif source_type == "HTTP/MJPEG Stream":
            http_url = st.text_input("MJPEG URL", placeholder="http://192.168.1.100:8080/video")
        elif source_type == "RTMP Stream":
            rtmp_url = st.text_input("RTMP URL", placeholder="rtmp://server/live/stream")

        # Reset button
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.ui.reset()
            st.session_state.events = []
            st.session_state.frame_count = 0
            st.success("Statistics reset!")

        start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)
        stop_button = st.button("⏹️ Stop Processing", use_container_width=True)

        if st.session_state.ui.is_live and st.session_state.ui.cctv_stream:
            status_color = "🟢" if st.session_state.ui.cctv_stream.is_connected else "🔴"
            st.markdown(f"{status_color} **CCTV Status:** {st.session_state.ui.cctv_stream.connection_status}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📸 Live Feed")
        video_placeholder = st.empty()

        # Performance metrics
        metrics_placeholder = st.empty()

        if start_button and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.events = []
            st.session_state.frame_count = 0

            # Initialize video capture based on source type
            cap = None
            is_cctv = False
            temp_file_path = None

            try:
                if source_type == "Video File":
                    if video_path and os.path.exists(video_path):
                        # Use local file directly
                        cap = cv2.VideoCapture(video_path)
                        file_size_gb = os.path.getsize(video_path) / (1024 * 1024 * 1024)
                        st.success(f"✓ Loading local video file: {video_path} ({file_size_gb:.2f} GB)")
                    elif video_file:
                        # Save uploaded file to disk using chunked handler
                        with st.spinner("Saving large video file to disk..."):
                            temp_file_path = st.session_state.file_handler.save_uploaded_file(video_file)
                            cap = cv2.VideoCapture(temp_file_path)
                        st.success(f"✓ Loaded video: {video_file.name}")
                    else:
                        st.error("Please select a video file or provide a local path")
                        st.session_state.processing = False

                elif source_type == "Webcam":
                    cap = cv2.VideoCapture(0)
                    st.success("✓ Webcam initialized")
                elif source_type == "RTSP Camera" and rtsp_url:
                    if st.session_state.ui.start_cctv_stream(rtsp_url):
                        is_cctv = True
                        st.success("✓ RTSP Camera connected")
                    else:
                        st.error("Failed to connect to RTSP camera")
                elif source_type == "HTTP/MJPEG Stream" and http_url:
                    if st.session_state.ui.start_cctv_stream(http_url):
                        is_cctv = True
                        st.success("✓ HTTP Stream connected")
                    else:
                        st.error("Failed to connect to HTTP stream")
                elif source_type == "RTMP Stream" and rtmp_url:
                    if st.session_state.ui.start_cctv_stream(rtmp_url):
                        is_cctv = True
                        st.success("✓ RTMP Stream connected")
                    else:
                        st.error("Failed to connect to RTMP stream")
                else:
                    st.error("Please select a valid video source")
                    st.session_state.processing = False

                if cap or is_cctv:
                    st.success("Processing started...")

                    frame_count = 0
                    processed_count = 0
                    start_time = time.time()

                    # Get original resolution info
                    if is_cctv and st.session_state.ui.cctv_stream:
                        orig_width, orig_height = st.session_state.ui.cctv_stream.get_resolution()
                        st.info(f"📹 Original Resolution: {orig_width}x{orig_height}")
                    elif cap:
                        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_fps = cap.get(cv2.CAP_PROP_FPS)
                        st.info(
                            f"📹 Original Resolution: {orig_width}x{orig_height} | Total Frames: {total_frames} | FPS: {video_fps:.1f}")

                    # Progress bar for video files
                    progress_bar = st.progress(0) if not is_cctv else None
                    status_text = st.empty() if not is_cctv else None

                    while st.session_state.processing:
                        # Get frame
                        if is_cctv and st.session_state.ui.cctv_stream:
                            ret, frame = st.session_state.ui.cctv_stream.get_frame()
                            if not ret:
                                time.sleep(0.01)
                                continue
                        elif cap:
                            ret, frame = cap.read()
                            if not ret:
                                st.info("Video ended")
                                break
                        else:
                            break

                        frame_count += 1

                        # Update progress for video files
                        if not is_cctv and total_frames > 0:
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress * 100:.1f}%)")

                        # Skip frames for faster processing
                        if frame_count % frame_skip == 0:
                            st.session_state.frame_count = frame_count
                            st.session_state.ui.frame_count = frame_count

                            # Process frame
                            processed_frame, new_events, track_count, person_count, vehicle_count = st.session_state.ui.process_frame(
                                frame, st.session_state.zones, allowed_direction,
                                loitering_seconds, crowd_threshold, track_vehicles,
                                track_people, confidence_threshold
                            )

                            processed_count += 1

                            # Update events
                            for event in new_events:
                                event["timestamp"] = datetime.now().strftime("%H:%M:%S")
                                st.session_state.events.append(event)

                            # Convert to RGB for display
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                            # Resize for display
                            if processed_frame_rgb.shape[1] > 800:
                                scale = 800 / processed_frame_rgb.shape[1]
                                new_width = 800
                                new_height = int(processed_frame_rgb.shape[0] * scale)
                                processed_frame_rgb = cv2.resize(processed_frame_rgb, (new_width, new_height))

                            # Calculate FPS
                            elapsed = time.time() - start_time
                            current_fps = processed_count / elapsed if elapsed > 0 else 0

                            # Update display
                            video_placeholder.image(processed_frame_rgb, channels="RGB",
                                                    caption=f"Frame: {frame_count}/{total_frames if not is_cctv else 'live'} | People: {person_count} | Vehicles: {vehicle_count} | FPS: {current_fps:.1f}")

                            # Update metrics
                            with metrics_placeholder.container():
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("📊 FPS", f"{current_fps:.1f}")
                                with col_b:
                                    st.metric("👥 People", person_count)
                                with col_c:
                                    st.metric("🚗 Vehicles", vehicle_count)
                                with col_d:
                                    st.metric("🎯 Total Tracks", track_count)

                        # Maintain real-time for live streams
                        if is_cctv and current_fps > st.session_state.ui.cctv_stream.get_fps():
                            time.sleep(0.001)

                    if progress_bar:
                        progress_bar.empty()
                    if status_text:
                        status_text.empty()

                    if cap:
                        cap.release()
                    if is_cctv:
                        st.session_state.ui.stop_cctv_stream()

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.processing = False
                # Clean up temp file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass

        if stop_button:
            st.session_state.processing = False
            st.session_state.ui.stop_cctv_stream()
            st.warning("Processing stopped")

    with col2:
        st.subheader("📊 Statistics")

        # Event counters
        event_counts = {
            "intrusion": 0,
            "loitering": 0,
            "crowd": 0,
            "wrong_direction": 0
        }

        # Class counters
        class_counts = {}

        for event in st.session_state.events:
            event_type = event.get("type", "unknown")
            if event_type in event_counts:
                event_counts[event_type] += 1

            # Track class counts for events
            class_name = event.get("class_name", "unknown")
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

        # Display metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🚨 Intrusions", event_counts["intrusion"])
            st.metric("⏱️ Loitering", event_counts["loitering"])
        with col_b:
            st.metric("👥 Crowd", event_counts["crowd"])
            st.metric("⬅️ Wrong Dir", event_counts["wrong_direction"])

        st.metric("📊 Total Frames", st.session_state.frame_count)
        st.metric("📋 Total Events", len(st.session_state.events))

        if class_counts:
            st.markdown("---")
            st.subheader("🎯 Events by Type")
            for class_name, count in class_counts.items():
                st.metric(class_name.capitalize(), count)

        st.markdown("---")
        st.subheader("📋 Recent Events")

        # Display recent events
        recent_events = st.session_state.events[-10:]
        if recent_events:
            for event in reversed(recent_events):
                event_type = event.get("type", "unknown")
                timestamp = event.get("timestamp", "")
                class_name = event.get("class_name", "")

                if event_type == "intrusion":
                    st.warning(
                        f"🚨 INTRUSION - {class_name} {event.get('track_id', '?')} in Zone {event.get('zone_id', '?')} [{timestamp}]")
                elif event_type == "loitering":
                    st.warning(
                        f"⏱️ LOITERING - {class_name} {event.get('track_id', '?')} ({event.get('dwell_time', 0):.1f}s) [{timestamp}]")
                elif event_type == "crowd":
                    st.error(
                        f"👥 CROWD - {event.get('count', 0)} people in Zone {event.get('zone_id', '?')} [{timestamp}]")
                elif event_type == "wrong_direction":
                    st.error(f"⬅️ WRONG DIRECTION - {class_name} {event.get('track_id', '?')} [{timestamp}]")
        else:
            st.info("No events detected yet")

        # Export events button
        if st.button("📥 Export Events", use_container_width=True):
            import json
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_events": len(st.session_state.events),
                "event_breakdown": event_counts,
                "events": st.session_state.events
            }
            with open("events_export.json", "w") as f:
                json.dump(export_data, f, indent=2)
            st.success(f"Exported {len(st.session_state.events)} events to events_export.json")

    # Cleanup on session end
    def on_session_end():
        if hasattr(st.session_state, 'file_handler'):
            st.session_state.file_handler.cleanup()

    # Register cleanup
    import atexit
    atexit.register(on_session_end)


if __name__ == "__main__":
    main()