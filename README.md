# GateKeeper 🚪

> *"Developed a real-time temporal analysis pipeline using YOLOv8 and ByteTrack to solve the identity-persistence problem in dynamic environments. Implemented a robust coordinate-geometry algorithm to handle non-linear crossing trajectories."*

GateKeeper is a research-grade people-counting system that uses a **virtual tripwire** to count individuals entering and exiting a region of interest in a video stream. It is designed to be robust, configurable, and extensible — suitable for both production CCTV deployments and Computer Vision research.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Detection** | YOLOv8 (n / s / m / l / x) via Ultralytics |
| **Tracking** | ByteTrack or BoT-SORT via `supervision` |
| **Crossing logic** | Vector **cross-product** method — works at any angle, not just horizontal lines |
| **Base point** | Bottom-centre `(x_mid, y_max)` — the feet — for accurate floor-plane crossing |
| **Temporal smoothing** | Configurable directional buffer prevents flickering counts |
| **Velocity estimation** | Pixels/second (or m/s via optional homography) |
| **HUD overlay** | Live IN / OUT / NET counters and per-track speed labels |

---

## 🛠 Technical Deep Dive

### 1. The Vector Cross-Product Method

Instead of comparing raw *y*-coordinates (only valid for horizontal lines), GateKeeper uses the **signed 2-D cross product** to determine which side of the tripwire a point lies on:

```
side = (B.x - A.x)(P.y - A.y) − (B.y - A.y)(P.x - A.x)
```

where `A → B` is the tripwire vector and `P` is the object's ground-contact point.  
A crossing is detected when consecutive observations yield **opposite signs**.

### 2. Centroid vs. Base Point

The **bottom-centre** of the bounding box is used as the ground-contact proxy:

```
base_point = ( (x1 + x2) / 2,  y2 )
```

In perspective views (typical CCTV geometry) the feet are anchored to the floor plane, making them a more physically accurate crossing indicator than the centroid.

### 3. Temporal Smoothing (Directional Buffer)

A naive frame-by-frame test flickers when a bounding box wavers on the line.  
GateKeeper requires a track to remain on the **same side for N consecutive frames** before committing a side assignment.  A crossing is only counted when the *confirmed* side changes.

```
buffer_frames_in: 3   # frames to confirm "entry" side
buffer_frames_out: 3  # frames to confirm "exit" side
```

### 4. Velocity Estimation

Given the track ID and per-frame timestamps, speed is estimated as:

```
v = Δd / Δt   (pixels/second)
```

A rolling window of recent positions is averaged to reduce single-frame noise.  
If a **homography matrix** is provided in the config, positions are projected to a top-down plane and the unit becomes **m/s**.

---

## 📁 Project Structure

```
├── assets/              # Demo GIFs and before/after images
├── configs/
│   └── default.yaml     # Line coordinates, model thresholds, tracker params
├── src/
│   ├── __init__.py
│   ├── detection.py     # YOLOv8 wrapper  (Detector, Detection)
│   ├── tracking.py      # ByteTrack / BoT-SORT wrapper  (Tracker, TrackedObject)
│   ├── logic.py         # Cross-product · base-point · buffer · velocity · TripwireCounter
│   └── utils.py         # HUD, polyline, and bounding-box visualisation
├── main.py              # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on a video file

```bash
python main.py --source path/to/video.mp4
```

### 3. Run on a webcam

```bash
python main.py --source 0
```

### 4. Save annotated output

```bash
python main.py --source path/to/video.mp4 --save-video --output annotated.mp4
```

### 5. Use a custom config

```bash
python main.py --source path/to/video.mp4 --config configs/default.yaml
```

---

## ⚙️ Configuration

Edit `configs/default.yaml` to tune the pipeline without changing any code.

```yaml
model:
  weights: yolov8n.pt      # swap for yolov8s/m/l/x for higher accuracy
  confidence: 0.40
  classes: [0]             # 0 = person

tripwire:
  start: [320, 360]        # pixel coordinates of tripwire endpoint A
  end:   [960, 360]        # pixel coordinates of tripwire endpoint B
  buffer_frames_in: 3      # temporal smoothing — entry side confirmation
  buffer_frames_out: 3     # temporal smoothing — exit side confirmation

velocity:
  enabled: true
  homography_matrix: null  # set to a 3×3 matrix for real-world m/s output
```

---

## 📊 Performance

| Model | Resolution | Device | FPS (approx.) | mAP |
|---|---|---|---|---|
| YOLOv8n | 640×640 | RTX 3080 | ~85 | 37.3 |
| YOLOv8s | 640×640 | RTX 3080 | ~65 | 44.9 |
| YOLOv8m | 640×640 | RTX 3080 | ~40 | 50.2 |
| YOLOv8l | 640×640 | RTX 3080 | ~25 | 52.9 |
| YOLOv8n | 640×640 | CPU (i7) | ~12 | 37.3 |

> Actual throughput depends on scene complexity and hardware. FPS values above are single-stream estimates.

---

## 🔬 Research Context

This project directly complements work on **multi-object tracking (MOT)** and builds on research in asymmetric cross-modal attention and social interaction modelling.  Key algorithmic choices — the cross-product side test, base-point grounding, and temporal buffer — mirror the requirements for robust identity-persistence in dynamic environments.

Future extensions:
- **Re-ID gallery**: compare feature embeddings (lightweight ViT / OSNet) of new IDs appearing near the line against recently lost tracks to handle occlusion at crossing
- **Homography calibration tool**: GUI to map four ground-plane points to metric coordinates
- **Multi-line support**: define multiple tripwires per scene with independent counters

