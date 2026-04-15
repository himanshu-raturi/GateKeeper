import cv2
import yaml
from ultralytics import YOLO
from gatekeeper.logic import GateKeeperLogic
from gatekeeper.utils import draw_trail, draw_hud, log_crossing

# Load Config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize with YOUR research weights
model = YOLO(config['model']['path'])
gate = GateKeeperLogic(line_y=config['tripwire']['line_y'])
track_history = {} # {id: [(x,y), (x,y)...]}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
    alert_this_frame = False

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            cx, cy = int((box[0] + box[2]) / 2), int(box[3])
            
            # Store history for trails
            if track_id not in track_history: track_history[track_id] = []
            track_history[track_id].append((cx, cy))

            # Logic & Logging
            event = gate.check_crossing(track_id, cy)
            if event:
                log_crossing(track_id, event)
                alert_this_frame = True

            # Visuals
            draw_trail(frame, track_history, track_id)
    
    draw_hud(frame, gate.counts, gate.line_y, alert=alert_this_frame)
    cv2.imshow("GateKeeper - Research Edition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()