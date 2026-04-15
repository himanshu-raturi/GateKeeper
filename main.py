import cv2
from ultralytics import YOLO
from gatekeeper.logic import GateKeeperLogic

# 1. Setup Model & Logic
model = YOLO('yolov8n.pt')  # Nano version for high FPS
gate = GateKeeperLogic(line_y=400, buffer_threshold=5)

# 2. Video Source (0 for Webcam, or path to file)
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Tracking (Person class only: 0)
    results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            # Use bottom-center (feet) for crossing accuracy
            cx, cy = int((x1 + x2) / 2), int(y2)

            # 4. Process Crossing
            event = gate.check_crossing(track_id, cy)
            
            # 5. Visuals
            color = (0, 255, 0) if event else (0, 165, 255) # Green if crossed
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10), 0, 0.6, color, 2)

    # Draw the Tripwire and HUD
    cv2.line(frame, (0, gate.line_y), (frame.shape[1], gate.line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"IN: {gate.counts['In']} | OUT: {gate.counts['Out']}", (20, 50), 0, 1, (255, 255, 255), 3)

    cv2.imshow("GateKeeper v1.0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()