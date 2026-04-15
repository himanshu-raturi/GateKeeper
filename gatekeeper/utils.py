import json
import os
from datetime import datetime
import cv2

def draw_trail(frame, track_history, track_id, color=(0, 255, 0)):
    """Draws a 'tail' behind the person to show their recent movement."""
    if track_id not in track_history or len(track_history[track_id]) < 2:
        return
    
    # Get the last 10 points
    points = track_history[track_id][-10:]
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], color, 2)

def draw_hud(frame, counts, line_y, alert=False):
    """Draws a professional HUD and a flashing tripwire."""
    line_color = (0, 255, 0) if alert else (0, 0, 255)
    thickness = 4 if alert else 2
    
    # Draw Tripwire
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), line_color, thickness)
    
    # Draw Background for Text
    cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"IN: {counts['In']}", (30, 40), 0, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {counts['Out']}", (30, 70), 0, 0.8, (0, 0, 255), 2)
    
def log_crossing(track_id, direction, log_path="output/gatekeeper_logs.json"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "track_id": track_id,
        "direction": direction
    }

    # Append to the list in the JSON file
    logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(data)

    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=4)