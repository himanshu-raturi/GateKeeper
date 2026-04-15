"""main.py — GateKeeper entry point.

Usage
-----
  # Use default config + webcam (device index 0)
  python main.py

  # Specify a video file and a custom config
  python main.py --source path/to/video.mp4 --config configs/default.yaml

  # Save the annotated output
  python main.py --source path/to/video.mp4 --save-video --output annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.detection import Detector
from src.logic import TripwireCounter
from src.tracking import Tracker
from src.utils import draw_hud, draw_tracks, draw_tripwire


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="GateKeeper",
        description=(
            "Real-time people counter using YOLOv8 + ByteTrack and a "
            "virtual tripwire with cross-product crossing detection."
        ),
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: path to a video file, RTSP URL, or webcam index (default: 0).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write annotated output to disk (path set by --output or config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (overrides the config value).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the live OpenCV display window.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return the YAML configuration dictionary."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Open video source ────────────────────────────────────────────────
    source = args.source
    try:
        source = int(source)  # webcam index
    except ValueError:
        pass  # file path or URL

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open source: {source!r}")

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Optional video writer ────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    out_cfg = cfg.get("output", {})
    should_save = args.save_video or out_cfg.get("save_video", False)
    if should_save:
        out_path = args.output or out_cfg.get("output_path", "output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps_src, (frame_w, frame_h))
        print(f"[INFO] Saving annotated video to: {out_path}")

    show_window = (not args.no_window) and out_cfg.get("show_window", True)

    # ── Build pipeline components ────────────────────────────────────────
    print("[INFO] Loading YOLOv8 model …")
    detector = Detector.from_config(cfg.get("model", {}))

    print("[INFO] Initialising tracker …")
    tracker = Tracker.from_config(cfg.get("tracker", {}))

    wire_cfg = cfg.get("tripwire", {})
    line_start = tuple(wire_cfg.get("start", [frame_w // 4, frame_h // 2]))
    line_end   = tuple(wire_cfg.get("end",   [frame_w * 3 // 4, frame_h // 2]))

    vel_cfg  = cfg.get("velocity", {})
    homog    = None
    if vel_cfg.get("homography_matrix") is not None:
        homog = np.array(vel_cfg["homography_matrix"], dtype=np.float64)

    counter = TripwireCounter(
        line_start=line_start,
        line_end=line_end,
        buffer_in=wire_cfg.get("buffer_frames_in", 3),
        buffer_out=wire_cfg.get("buffer_frames_out", 3),
        velocity_enabled=vel_cfg.get("enabled", True),
        homography=homog,
    )

    speed_unit = "m/s" if homog is not None else "px/s"

    # ── Tracking of active IDs (for cleanup) ────────────────────────────
    active_ids: set = set()

    # ── Main loop ────────────────────────────────────────────────────────
    frame_count = 0
    t_start     = time.monotonic()
    print("[INFO] Pipeline running — press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame_count += 1

        # -- Detect
        detections = detector.detect(frame)

        # -- Track
        tracks = tracker.update(detections, frame)

        # -- Tripwire logic
        current_ids: set = set()
        for obj in tracks:
            current_ids.add(obj.track_id)
            event = counter.update(obj.track_id, obj.bbox)
            if event:
                direction = "→ IN" if event == "in" else "← OUT"
                print(
                    f"  [Frame {frame_count:05d}] Track #{obj.track_id:3d} crossed {direction} "
                    f"(IN={counter.count_in}, OUT={counter.count_out}, NET={counter.net_count:+d})"
                )

        # Clean up lost tracks
        lost_ids = active_ids - current_ids
        for tid in lost_ids:
            counter.remove_track(tid)
        active_ids = current_ids

        # -- Annotate frame
        draw_tripwire(frame, line_start, line_end)
        draw_tracks(frame, tracks, counter, speed_unit=speed_unit)
        draw_hud(frame, counter)

        # FPS overlay (bottom-right)
        elapsed = time.monotonic() - t_start
        fps_live = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            frame,
            f"FPS: {fps_live:.1f}",
            (frame_w - 150, frame_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(frame)

        if show_window:
            cv2.imshow("GateKeeper", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User quit.")
                break

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(
        f"\n[DONE] Processed {frame_count} frames in {elapsed:.1f}s "
        f"({avg_fps:.1f} FPS avg)\n"
        f"       IN={counter.count_in}  OUT={counter.count_out}  NET={counter.net_count:+d}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(parse_args())
