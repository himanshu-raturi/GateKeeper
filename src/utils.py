"""utils.py — OpenCV visualisation helpers for GateKeeper.

Provides:
  • draw_tripwire  – render the virtual line with directional arrows
  • draw_tracks    – render bounding boxes, track IDs, and speed labels
  • draw_hud       – overlay the in / out / net counters on the frame
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .logic import TripwireCounter
from .tracking import TrackedObject


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------

COLOUR_LINE     = (0,   255, 255)   # cyan
COLOUR_IN       = (0,   200,   0)   # green
COLOUR_OUT      = (0,    50, 220)   # red-orange
COLOUR_BOX      = (255, 165,   0)   # blue-ish orange
COLOUR_BOX_FAST = (0,    50, 220)   # red for fast movers
COLOUR_TEXT_BG  = (30,   30,  30)
COLOUR_WHITE    = (255, 255, 255)

FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Individual drawing helpers
# ---------------------------------------------------------------------------

def draw_tripwire(
    frame: np.ndarray,
    line_start: Tuple[float, float],
    line_end:   Tuple[float, float],
    thickness: int = 2,
) -> np.ndarray:
    """Draw the virtual tripwire on *frame* (in-place).

    An arrowhead at the midpoint indicates the "in" direction (from +1 side
    towards -1 side of the line, i.e. left-of-line → right-of-line).

    Args:
        frame:      BGR NumPy array to annotate.
        line_start: ``(x, y)`` pixel coordinate of one endpoint.
        line_end:   ``(x, y)`` pixel coordinate of the other endpoint.
        thickness:  Line thickness in pixels.

    Returns:
        Annotated frame (same array, modified in-place).
    """
    p1 = (int(line_start[0]), int(line_start[1]))
    p2 = (int(line_end[0]),   int(line_end[1]))
    cv2.line(frame, p1, p2, COLOUR_LINE, thickness, cv2.LINE_AA)

    # Small circles at the endpoints
    cv2.circle(frame, p1, 5, COLOUR_LINE, -1, cv2.LINE_AA)
    cv2.circle(frame, p2, 5, COLOUR_LINE, -1, cv2.LINE_AA)

    # Arrow at the midpoint pointing in the "in" direction (perpendicular,
    # towards the left side of the line vector).
    mx = int((p1[0] + p2[0]) / 2)
    my = int((p1[1] + p2[1]) / 2)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = max(1, int((dx ** 2 + dy ** 2) ** 0.5))
    arrow_len = 20
    # Perpendicular direction (rotate 90° counter-clockwise) = (-dy, dx)
    px = int(mx + arrow_len * (-dy) / length)
    py = int(my + arrow_len *   dx  / length)
    cv2.arrowedLine(frame, (mx, my), (px, py), COLOUR_IN, thickness, cv2.LINE_AA, tipLength=0.4)

    return frame


def draw_tracks(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    counter: TripwireCounter,
    speed_unit: str = "px/s",
) -> np.ndarray:
    """Draw bounding boxes, track IDs, and speed labels for all active tracks.

    Args:
        frame:      BGR NumPy array to annotate.
        tracks:     Active :class:`~tracking.TrackedObject` instances.
        counter:    :class:`~logic.TripwireCounter` used to query speed.
        speed_unit: Label appended to the speed value (``"px/s"`` or ``"m/s"``).

    Returns:
        Annotated frame (same array, modified in-place).
    """
    for obj in tracks:
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        speed = counter.get_speed(obj.track_id)
        colour = COLOUR_BOX_FAST if (speed is not None and speed > 200) else COLOUR_BOX

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        # Base point (feet)
        bx = (x1 + x2) // 2
        cv2.circle(frame, (bx, y2), 4, colour, -1, cv2.LINE_AA)

        # Label: ID + speed
        label = f"#{obj.track_id}"
        if speed is not None:
            label += f"  {speed:.0f} {speed_unit}"

        _put_label(frame, label, (x1, y1 - 6))

    return frame


def draw_hud(
    frame: np.ndarray,
    counter: TripwireCounter,
) -> np.ndarray:
    """Overlay the IN / OUT / NET counters in the top-left corner.

    Args:
        frame:   BGR NumPy array to annotate.
        counter: :class:`~logic.TripwireCounter` holding the running counts.

    Returns:
        Annotated frame (same array, modified in-place).
    """
    lines = [
        (f"IN  : {counter.count_in:4d}", COLOUR_IN),
        (f"OUT : {counter.count_out:4d}", COLOUR_OUT),
        (f"NET : {counter.net_count:+4d}", COLOUR_WHITE),
    ]

    x_origin, y_origin = 20, 40
    line_height = 34
    padding     = 8

    # Background rectangle
    box_w = 210
    box_h = line_height * len(lines) + padding * 2
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_origin - padding, y_origin - 26),
        (x_origin + box_w, y_origin + box_h - 10),
        COLOUR_TEXT_BG,
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for idx, (text, colour) in enumerate(lines):
        y = y_origin + idx * line_height
        cv2.putText(
            frame, text, (x_origin, y),
            FONT, 0.85, colour, 2, cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw a small semi-transparent label with a dark background."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    # Clamp to frame bounds
    x = max(x, 0)
    y = max(y, th + baseline)
    pad = 3
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        COLOUR_TEXT_BG,
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, scale, COLOUR_WHITE, thickness, cv2.LINE_AA)
