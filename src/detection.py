"""detection.py — YOLOv8 detection wrapper for GateKeeper.

Provides a thin, configuration-driven wrapper around Ultralytics YOLOv8 so
that the rest of the pipeline does not need to import ``ultralytics`` directly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Detection:
    """A single object detection result.

    Attributes:
        bbox:       Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detection confidence in ``[0, 1]``.
        class_id:   COCO class identifier (0 = person).
        class_name: Human-readable class label.
    """

    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.bbox       = bbox
        self.confidence = confidence
        self.class_id   = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"Detection(class={self.class_name!r}, conf={self.confidence:.2f}, "
            f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}))"
        )


class Detector:
    """YOLOv8 detector wrapping the Ultralytics API.

    Args:
        weights:    Path to the ``.pt`` model file, or a model name like
                    ``"yolov8n.pt"`` (auto-downloaded on first use).
        confidence: Minimum confidence threshold for returned detections.
        iou:        Non-maximum suppression IoU threshold.
        device:     PyTorch device string (``"cpu"``, ``"cuda:0"``, ``""`` for
                    auto-select).
        classes:    List of COCO class IDs to return.  ``None`` returns all.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.40,
        iou: float = 0.45,
        device: str = "",
        classes: Optional[List[int]] = None,
    ) -> None:
        self._conf    = confidence
        self._iou     = iou
        self._device  = device
        self._classes = classes
        self._model   = self._load(weights)

    # ------------------------------------------------------------------
    def _load(self, weights: str):
        """Load the YOLO model (deferred import keeps startup fast)."""
        from ultralytics import YOLO  # noqa: PLC0415
        return YOLO(weights)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Args:
            frame: OpenCV BGR image as a NumPy array (H × W × 3).

        Returns:
            List of :class:`Detection` objects filtered by confidence and class.
        """
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            classes=self._classes,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy  = boxes.xyxy.cpu().numpy()   # (N, 4)
            confs = boxes.conf.cpu().numpy()   # (N,)
            cls   = boxes.cls.cpu().numpy().astype(int)  # (N,)
            names = result.names

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                detections.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(confs[i]),
                        class_id=int(cls[i]),
                        class_name=str(names[cls[i]]),
                    )
                )

        return detections

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "Detector":
        """Construct a :class:`Detector` from a ``model`` config dict.

        Args:
            cfg: Dictionary corresponding to the ``model:`` section of
                 ``configs/default.yaml``.

        Returns:
            Configured :class:`Detector` instance.
        """
        return cls(
            weights=cfg.get("weights", "yolov8n.pt"),
            confidence=cfg.get("confidence", 0.40),
            iou=cfg.get("iou", 0.45),
            device=cfg.get("device", ""),
            classes=cfg.get("classes", None),
        )
