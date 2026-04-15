"""tracking.py — ByteTrack / BoT-SORT wrapper for GateKeeper.

Provides a unified :class:`Tracker` interface so that the rest of the pipeline
does not import ``supervision`` directly, and switching between ByteTrack and
BoT-SORT requires only a config change.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .detection import Detection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TrackedObject:
    """One tracked object returned by :class:`Tracker`.

    Attributes:
        track_id:   Persistent integer ID assigned by the tracker.
        bbox:       Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detection confidence for the associated detection.
        class_id:   COCO class identifier.
        class_name: Human-readable class label.
    """

    __slots__ = ("track_id", "bbox", "confidence", "class_id", "class_name")

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.track_id   = track_id
        self.bbox       = bbox
        self.confidence = confidence
        self.class_id   = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"TrackedObject(id={self.track_id}, class={self.class_name!r}, "
            f"conf={self.confidence:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}))"
        )


class Tracker:
    """Multi-object tracker wrapping the ``supervision`` ByteTrack implementation.

    Args:
        tracker_type:      ``"bytetrack"`` or ``"botsort"`` (case-insensitive).
        track_high_thresh: High-confidence detection threshold for ByteTrack.
        track_low_thresh:  Low-confidence detection threshold for ByteTrack.
        new_track_thresh:  Minimum score to initialise a new track.
        track_buffer:      Frames to keep a lost track alive before removal.
        match_thresh:      IoU matching threshold between tracks and detections.
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.60,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
    ) -> None:
        self._tracker_type    = tracker_type.lower()
        self._track_buffer    = track_buffer
        self._match_thresh    = match_thresh
        self._track_high_thresh = track_high_thresh
        self._track_low_thresh  = track_low_thresh
        self._new_track_thresh  = new_track_thresh

        self._tracker = self._build_tracker()

    # ------------------------------------------------------------------
    def _build_tracker(self):
        """Instantiate the underlying ``supervision`` tracker."""
        import supervision as sv  # noqa: PLC0415

        if self._tracker_type == "bytetrack":
            return sv.ByteTrack(
                track_activation_threshold=self._track_high_thresh,
                lost_track_buffer=self._track_buffer,
                minimum_matching_threshold=self._match_thresh,
                minimum_consecutive_frames=1,
            )
        if self._tracker_type == "botsort":
            # supervision >= 0.21 ships BotSort
            return sv.BotSort(
                track_activation_threshold=self._track_high_thresh,
                lost_track_buffer=self._track_buffer,
                minimum_matching_threshold=self._match_thresh,
            )
        raise ValueError(
            f"Unknown tracker type: {self._tracker_type!r}. "
            "Expected 'bytetrack' or 'botsort'."
        )

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """Update tracker state and return currently active tracks.

        Args:
            detections: List of :class:`~detection.Detection` objects for the
                        current frame.
            frame:      The raw BGR frame (required by BoT-SORT for Re-ID;
                        unused by ByteTrack).

        Returns:
            List of :class:`TrackedObject` instances, one per active track.
        """
        import supervision as sv  # noqa: PLC0415

        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy  = np.array([d.bbox       for d in detections], dtype=np.float32)
            confs = np.array([d.confidence for d in detections], dtype=np.float32)
            cls   = np.array([d.class_id   for d in detections], dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=cls)

        tracked = self._tracker.update_with_detections(sv_dets)

        # Build a class_id → class_name lookup from the current detections
        class_name_map: dict = {d.class_id: d.class_name for d in detections}

        results: List[TrackedObject] = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            box = tuple(float(v) for v in tracked.xyxy[i])
            conf_val = (
                float(tracked.confidence[i])
                if tracked.confidence is not None
                else 0.0
            )
            cid  = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            name = class_name_map.get(cid, str(cid))
            results.append(
                TrackedObject(
                    track_id=tid,
                    bbox=box,  # type: ignore[arg-type]
                    confidence=conf_val,
                    class_id=cid,
                    class_name=name,
                )
            )

        return results

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "Tracker":
        """Construct a :class:`Tracker` from a ``tracker`` config dict.

        Args:
            cfg: Dictionary corresponding to the ``tracker:`` section of
                 ``configs/default.yaml``.

        Returns:
            Configured :class:`Tracker` instance.
        """
        return cls(
            tracker_type=cfg.get("type", "bytetrack"),
            track_high_thresh=cfg.get("track_high_thresh", 0.50),
            track_low_thresh=cfg.get("track_low_thresh", 0.10),
            new_track_thresh=cfg.get("new_track_thresh", 0.60),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.80),
        )
