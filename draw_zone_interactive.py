import cv2
import pickle
import os
import numpy as np

# Global variables
drawing = False
points = []
zones = []

def draw_zone(event, x, y, flags, param):
    global drawing, points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
        print(f"Point added: ({x}, {y})")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 3:
            # Save the zone
            zone = {
                "id": len(zones) + 1,
                "name": f"Zone {len(zones) + 1}",
                "polygon": points.copy()
            }
            zones.append(zone)
            print(f"\n✓ Zone {zone['id']} saved with {len(points)} points")
            points = []
        else:
            print("Need at least 3 points for a zone")

# Get video path
print("="*50)
print("INTERACTIVE ZONE DRAWING TOOL")
print("="*50)
print("\nThis tool lets you draw zones on your video frame")
print("Zones will be saved and used for event detection\n")

video_path = input("Enter video path: ").strip('"')

if not os.path.exists(video_path):
    print(f"Error: File not found - {video_path}")
    exit(1)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit(1)

# Get first frame
ret, frame = cap.read()
if not ret:
    print("Error reading frame")
    exit(1)

# Resize for display if too large
original_height, original_width = frame.shape[:2]
max_display_width = 1280
if frame.shape[1] > max_display_width:
    scale = max_display_width / frame.shape[1]
    new_width = int(frame.shape[1] * scale)
    new_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (new_width, new_height))
    print(f"\nFrame resized from {original_width}x{original_height} to {new_width}x{new_height} for display")

clone = frame.copy()
cv2.namedWindow("Draw Zone")
cv2.setMouseCallback("Draw Zone", draw_zone)

print("\n" + "="*50)
print("INSTRUCTIONS")
print("="*50)
print("1. LEFT CLICK to add points around the area you want to monitor")
print("2. RIGHT CLICK to finish and save the zone")
print("3. Press 'c' to clear current points")
print("4. Press 'r' to reset all zones")
print("5. Press 'q' when done\n")

while True:
    # Draw current points
    temp = frame.copy()
    
    # Draw existing zones
    for zone in zones:
        pts = zone["polygon"]
        pts_array = np.array(pts, dtype=np.int32)
        cv2.polylines(temp, [pts_array], True, (0, 255, 0), 2)
        # Add zone label
        center_x = sum(p[0] for p in pts) // len(pts)
        center_y = sum(p[1] for p in pts) // len(pts)
        cv2.putText(temp, zone["name"], (center_x, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw current drawing points
    if len(points) > 0:
        for i, pt in enumerate(points):
            cv2.circle(temp, pt, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(temp, points[i-1], pt, (0, 0, 255), 2)
    
    # Show instructions on frame
    cv2.putText(temp, f"Points: {len(points)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(temp, f"Zones: {len(zones)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(temp, "Left click: add point | Right click: save zone | C: clear | R: reset | Q: quit", 
               (10, temp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Draw Zone", temp)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        points = []
        print("Cleared current points")
    elif key == ord('r'):
        zones = []
        points = []
        print("Reset all zones")

cv2.destroyAllWindows()
cap.release()

# Save zones to file
if zones:
    # Adjust zone coordinates back to original size if we resized
    if original_width > max_display_width:
        scale = original_width / max_display_width
        for zone in zones:
            zone["polygon"] = [(int(x * scale), int(y * scale)) for x, y in zone["polygon"]]
    
    with open("zones.pkl", "wb") as f:
        pickle.dump(zones, f)
    
    print(f"\n✓ Saved {len(zones)} zones to 'zones.pkl'")
    print("\nZones created:")
    for zone in zones:
        print(f"\nZone {zone['id']}: {zone['name']}")
        print(f"  Polygon points: {zone['polygon']}")
    
    # Option to run analysis
    run_analysis = input("\nRun video analysis with these zones? (y/n): ").lower()
    if run_analysis == 'y':
        print("\nNow run test_with_video.py and it will use these zones")
        print("Or press any key to exit")
else:
    print("\nNo zones created")

print("\nDone!")