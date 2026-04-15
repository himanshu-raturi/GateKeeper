"""logic.py — coordinate-geometry core for GateKeeper.

Implements:
  • get_base_point      – extract the ground-contact point from a bounding box
  • cross_product       – signed 2-D cross product used for side-of-line tests
  • segments_intersect  – robust segment-crossing test via the cross-product method
  • VelocityEstimator   – per-track speed estimation (px/s, optional homography)
  • DirectionalBuffer   – temporal smoothing to prevent flickering counts
  • TripwireCounter     – high-level state machine that ties everything together
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point = Tuple[float, float]   # (x, y)
BBox  = Tuple[float, float, float, float]  # (x1, y1, x2, y2) — top-left / bottom-right


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_base_point(bbox: BBox) -> Point:
    """Return the bottom-centre of a bounding box — the best ground-contact proxy.

    In perspective camera views the feet are anchored to the floor plane, so
    using ``(x_mid, y_max)`` produces more physically accurate crossing events
    than the centroid.

    Args:
        bbox: ``(x1, y1, x2, y2)`` in pixel coordinates (top-left, bottom-right).

    Returns:
        ``(x_mid, y_max)`` pixel coordinate.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def cross_product(o: Point, a: Point, b: Point) -> float:
    """Compute the signed 2-D cross product of vectors ``OA`` and ``OB``.

    The sign indicates on which side of the directed line ``O → A`` the point
    ``B`` lies:

    * positive → ``B`` is to the *left*  of ``O → A``
    * negative → ``B`` is to the *right* of ``O → A``
    * zero     → ``B`` is *collinear*    with ``O → A``

    Args:
        o: Origin point.
        a: First point defining the direction vector ``OA``.
        b: Point to test.

    Returns:
        Signed cross-product scalar.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(
    p1: Point,
    p2: Point,
    q1: Point,
    q2: Point,
) -> bool:
    """Return *True* if segment ``P1→P2`` crosses segment ``Q1→Q2``.

    Uses the cross-product / "straddle" test: two segments cross when each
    segment's endpoints straddle the infinite line defined by the other
    segment.  This handles all angles — not just horizontal or vertical lines.

    Args:
        p1, p2: Endpoints of the first segment (the object's displacement
                vector from the previous frame to the current frame).
        q1, q2: Endpoints of the second segment (the virtual tripwire).

    Returns:
        ``True`` if the segments cross, ``False`` otherwise.
    """
    d1 = cross_product(q1, q2, p1)
    d2 = cross_product(q1, q2, p2)
    d3 = cross_product(p1, p2, q1)
    d4 = cross_product(p1, p2, q2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear / on-segment edge cases — treat as non-crossing to avoid
    # double-counting objects that sit exactly on the line.
    return False


def side_of_line(point: Point, line_start: Point, line_end: Point) -> int:
    """Return the side of the directed line ``line_start → line_end`` that
    ``point`` is on, based on the sign of the 2-D cross product.

    .. note::
        This function returns the **sign of the cross product** without
        attaching a "left/right" label, because the interpretation depends on
        the coordinate system.  In standard maths coordinates (y-axis up) a
        positive result means the point is to the *left* of the directed line;
        in image/screen coordinates (y-axis down) a positive result means the
        point is *below* the line.

    Returns:
        +1 if the cross-product is positive,
        -1 if the cross-product is negative,
         0 if the point is collinear with the line.
    """
    cp = cross_product(line_start, line_end, point)
    if cp > 0:
        return 1
    if cp < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Velocity estimation
# ---------------------------------------------------------------------------

class VelocityEstimator:
    """Track per-object speed in pixels/second (and optionally m/s).

    Maintains a short rolling history of ``(timestamp, position)`` tuples for
    each ``track_id`` so that speed is computed over multiple frames rather
    than a single noisy frame-to-frame difference.

    Args:
        history_len: Number of recent positions to keep per track.
        homography:  Optional 3×3 NumPy array.  When supplied, positions are
                     projected to a top-down plane before computing speed,
                     which converts the unit to real-world metres/second
                     (assuming the homography maps pixels → metres).
    """

    def __init__(
        self,
        history_len: int = 10,
        homography: Optional[np.ndarray] = None,
    ) -> None:
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._homography = homography

    # ------------------------------------------------------------------
    def update(self, track_id: int, position: Point) -> None:
        """Record a new observation for ``track_id``."""
        self._history[track_id].append((time.monotonic(), position))

    # ------------------------------------------------------------------
    def get_speed(self, track_id: int) -> Optional[float]:
        """Return estimated speed for ``track_id``, or *None* if insufficient data.

        The speed is averaged over all consecutive pairs in the rolling window
        to reduce single-frame noise.

        Returns:
            Speed in pixels/second (or m/s if a homography was provided), or
            ``None`` when fewer than two observations exist.
        """
        history = self._history.get(track_id)
        if not history or len(history) < 2:
            return None

        speeds: List[float] = []
        entries = list(history)
        for i in range(1, len(entries)):
            t0, p0 = entries[i - 1]
            t1, p1 = entries[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            p0_proj = self._project(p0)
            p1_proj = self._project(p1)
            dist = float(np.linalg.norm(np.array(p1_proj) - np.array(p0_proj)))
            speeds.append(dist / dt)

        return float(np.mean(speeds)) if speeds else None

    # ------------------------------------------------------------------
    def remove(self, track_id: int) -> None:
        """Discard history for a track that has been lost."""
        self._history.pop(track_id, None)

    # ------------------------------------------------------------------
    def _project(self, point: Point) -> Point:
        if self._homography is None:
            return point
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        proj = cv2_perspective_transform(pt, self._homography)
        return (float(proj[0, 0, 0]), float(proj[0, 0, 1]))


def cv2_perspective_transform(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Apply a 3×3 homography matrix to an array of 2-D points.

    Args:
        points: Shape ``(N, 1, 2)`` float32.
        H:      3×3 float64 homography matrix.

    Returns:
        Transformed points, same shape as input.
    """
    import cv2  # imported locally to keep this module importable without OpenCV
    return cv2.perspectiveTransform(points, H)


# ---------------------------------------------------------------------------
# Directional buffer
# ---------------------------------------------------------------------------

class DirectionalBuffer:
    """Per-track temporal smoothing buffer.

    An object must be seen on the same side of the tripwire for at least
    ``required_frames`` consecutive frames before its side assignment is
    considered "confirmed."  This prevents flickering counts when a bounding
    box wavers right on the line.

    Args:
        required_frames: Minimum consecutive frames before a side is confirmed.
    """

    def __init__(self, required_frames: int = 3) -> None:
        self._required = required_frames
        # Maps track_id → (current_candidate_side, consecutive_frame_count)
        self._state: Dict[int, Tuple[int, int]] = {}
        # Maps track_id → last *confirmed* side
        self._confirmed: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(self, track_id: int, observed_side: int) -> Optional[int]:
        """Feed a new side observation and return the *confirmed* side (or None).

        Args:
            track_id:      The ByteTrack / BoT-SORT track identifier.
            observed_side: +1, -1, or 0 (from :func:`side_of_line`).

        Returns:
            The confirmed side once the streak reaches ``required_frames``,
            otherwise ``None``.
        """
        if observed_side == 0:
            # Collinear — don't reset the buffer, just wait.
            return self._confirmed.get(track_id)

        candidate, streak = self._state.get(track_id, (observed_side, 0))

        if observed_side == candidate:
            streak += 1
        else:
            # Side changed — restart streak for the new candidate.
            candidate = observed_side
            streak = 1

        self._state[track_id] = (candidate, streak)

        if streak >= self._required:
            self._confirmed[track_id] = candidate

        return self._confirmed.get(track_id)

    # ------------------------------------------------------------------
    def get_confirmed(self, track_id: int) -> Optional[int]:
        """Return the most recently confirmed side for ``track_id``, or None."""
        return self._confirmed.get(track_id)

    # ------------------------------------------------------------------
    def remove(self, track_id: int) -> None:
        """Clean up state for a lost track."""
        self._state.pop(track_id, None)
        self._confirmed.pop(track_id, None)


# ---------------------------------------------------------------------------
# High-level tripwire counter
# ---------------------------------------------------------------------------

class TripwireCounter:
    """State machine that counts objects crossing a virtual tripwire.

    Combines :class:`DirectionalBuffer` (temporal smoothing) and
    :class:`VelocityEstimator` into a single easy-to-use interface.

    Args:
        line_start:       One endpoint of the tripwire ``(x, y)``.
        line_end:         Other endpoint of the tripwire ``(x, y)``.
        buffer_in:        Frames needed to confirm "entry" side.
        buffer_out:       Frames needed to confirm "exit" side.
        velocity_enabled: Whether to estimate per-track speed.
        homography:       Optional 3×3 homography for real-world speed.
    """

    def __init__(
        self,
        line_start: Point,
        line_end: Point,
        buffer_in: int = 3,
        buffer_out: int = 3,
        velocity_enabled: bool = True,
        homography: Optional[np.ndarray] = None,
    ) -> None:
        self.line_start = line_start
        self.line_end   = line_end

        # Use the stricter of the two buffer values for the shared buffer.
        max_buffer = max(buffer_in, buffer_out)
        self._buffer   = DirectionalBuffer(required_frames=max_buffer)
        self._velocity = VelocityEstimator(homography=homography) if velocity_enabled else None

        # Confirmed side from the *previous* call to update()
        self._prev_confirmed: Dict[int, int] = {}

        self.count_in  = 0  # crossed from right → left  (side +1 → side -1)
        self.count_out = 0  # crossed from left  → right (side -1 → side +1)

    # ------------------------------------------------------------------
    def update(
        self,
        track_id: int,
        bbox: BBox,
    ) -> Optional[str]:
        """Process one detection and return a crossing event string if one occurred.

        Args:
            track_id: Unique track identifier.
            bbox:     ``(x1, y1, x2, y2)`` bounding box in pixel coordinates.

        Returns:
            ``"in"`` or ``"out"`` if a crossing was just confirmed, else ``None``.
        """
        base = get_base_point(bbox)
        current_side = side_of_line(base, self.line_start, self.line_end)

        # Update velocity estimator
        if self._velocity is not None:
            self._velocity.update(track_id, base)

        # Update temporal buffer
        confirmed_side = self._buffer.update(track_id, current_side)

        # Detect transitions in the *confirmed* side
        event: Optional[str] = None
        prev = self._prev_confirmed.get(track_id)
        if prev is not None and confirmed_side is not None and confirmed_side != prev:
            if prev == -1 and confirmed_side == 1:
                self.count_in += 1
                event = "in"
            elif prev == 1 and confirmed_side == -1:
                self.count_out += 1
                event = "out"

        if confirmed_side is not None:
            self._prev_confirmed[track_id] = confirmed_side

        return event

    # ------------------------------------------------------------------
    def get_speed(self, track_id: int) -> Optional[float]:
        """Return the current speed estimate for ``track_id`` (px/s or m/s)."""
        if self._velocity is None:
            return None
        return self._velocity.get_speed(track_id)

    # ------------------------------------------------------------------
    def remove_track(self, track_id: int) -> None:
        """Release all state for a track that has been lost."""
        self._buffer.remove(track_id)
        self._prev_confirmed.pop(track_id, None)
        if self._velocity is not None:
            self._velocity.remove(track_id)

    # ------------------------------------------------------------------
    @property
    def net_count(self) -> int:
        """Net occupancy: ``count_in - count_out``."""
        return self.count_in - self.count_out
