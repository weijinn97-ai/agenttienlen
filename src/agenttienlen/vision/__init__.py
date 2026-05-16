"""YOLOv8-based real-time card detection."""

from agenttienlen.vision.labels import (
    CARD_CLASS_NAMES,
    NUM_CLASSES,
    card_to_class_id,
    class_id_to_card,
)
from agenttienlen.vision.layout import ROI, RegionName, default_layout_1280x720

__all__ = [
    "CARD_CLASS_NAMES",
    "NUM_CLASSES",
    "ROI",
    "RegionName",
    "card_to_class_id",
    "class_id_to_card",
    "default_layout_1280x720",
]
