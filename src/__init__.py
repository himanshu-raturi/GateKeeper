"""GateKeeper — real-time people-counting with YOLOv8 + ByteTrack."""

from .logic import TripwireCounter, get_base_point, cross_product, segments_intersect

__all__ = [
    "TripwireCounter",
    "get_base_point",
    "cross_product",
    "segments_intersect",
]

# Optional imports — only available when OpenCV, ultralytics, and supervision
# are installed (i.e. in a full runtime environment, not during unit tests).
try:
    from .detection import Detector  # noqa: F401
    from .tracking import Tracker    # noqa: F401
    from .utils import draw_tripwire, draw_tracks, draw_hud  # noqa: F401

    __all__ += ["Detector", "Tracker", "draw_tripwire", "draw_tracks", "draw_hud"]
except ModuleNotFoundError:
    pass
