"""Region-based real-time card detection.

The top-level entrypoint is :class:`StructuredFramePipeline`, which
splits each captured frame into per-region crops via
:class:`LayoutRouter` and dispatches them to five pluggable readers
(:class:`HandReader`, :class:`TableReader`, :class:`OpponentReader`,
:class:`PlayerIdentityReader`, :class:`TurnDetector`). Each reader
returns its slice of :class:`StructuredFrame` data, which downstream
code (stabilizer + state machine + game state) interprets.
"""

from agenttienlen.vision.labels import (
    CARD_CLASS_NAMES,
    NUM_CLASSES,
    card_to_class_id,
    class_id_to_card,
)
from agenttienlen.vision.layout import (
    ROI,
    RegionName,
    default_layout_1280x720,
    full_layout_1280x720,
)
from agenttienlen.vision.layout_router import LayoutRouter, RegionCrop
from agenttienlen.vision.pipeline import StructuredFramePipeline
from agenttienlen.vision.readers import (
    HandReader,
    HandReadResult,
    OpponentReader,
    PlayerIdentityReader,
    TableReader,
    TableReadResult,
    TurnDetector,
    TurnReadResult,
)
from agenttienlen.vision.stabilizer import FrameStabilizer, StableResult
from agenttienlen.vision.structured_frame import (
    ButtonName,
    ButtonState,
    CardCandidate,
    PlayerProfile,
    Rect,
    Seat,
    SeatMap,
    StructuredFrame,
)

__all__ = [
    "CARD_CLASS_NAMES",
    "NUM_CLASSES",
    "ROI",
    "ButtonName",
    "ButtonState",
    "CardCandidate",
    "FrameStabilizer",
    "HandReadResult",
    "HandReader",
    "LayoutRouter",
    "OpponentReader",
    "PlayerIdentityReader",
    "PlayerProfile",
    "Rect",
    "RegionCrop",
    "RegionName",
    "Seat",
    "SeatMap",
    "StableResult",
    "StructuredFrame",
    "StructuredFramePipeline",
    "TableReadResult",
    "TableReader",
    "TurnDetector",
    "TurnReadResult",
    "card_to_class_id",
    "class_id_to_card",
    "default_layout_1280x720",
    "full_layout_1280x720",
]
