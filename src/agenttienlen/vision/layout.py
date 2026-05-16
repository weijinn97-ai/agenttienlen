"""Screen region (ROI) layout for grouping YOLO detections.

After YOLO outputs all card bounding boxes on a frame, we need to bucket each
detection into a semantic region:

- ``MY_HAND``     — the bot's own hand at the bottom of the screen.
- ``TABLE``       — the combo currently in the middle (last play).
- ``OPP_LEFT``    — left opponent's played cards.
- ``OPP_TOP``     — top opponent's played cards.
- ``OPP_RIGHT``   — right opponent's played cards.

A detection's region is decided by which ROI rectangle contains its centre.

Coordinates below are for 1280x720 captures, calibrated against the screenshots
in the project. Other resolutions should call :func:`scale_layout`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class RegionName(Enum):
    MY_HAND = "my_hand"
    TABLE = "table"
    OPP_LEFT = "opp_left"
    OPP_TOP = "opp_top"
    OPP_RIGHT = "opp_right"


@dataclass(frozen=True, slots=True)
class ROI:
    name: RegionName
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, cx: float, cy: float) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


def default_layout_1280x720() -> list[ROI]:
    """ROI rectangles tuned against the 1280x720 screenshots in this repo.

    Values are conservative bounding boxes; refine per-game via
    ``tools/calibrate.py``.
    """
    return [
        ROI(RegionName.MY_HAND, 200, 500, 1100, 720),
        ROI(RegionName.TABLE, 450, 340, 830, 500),
        ROI(RegionName.OPP_LEFT, 150, 200, 450, 340),
        ROI(RegionName.OPP_TOP, 500, 30, 850, 200),
        ROI(RegionName.OPP_RIGHT, 830, 200, 1130, 340),
    ]


def scale_layout(layout: Iterable[ROI], width: int, height: int) -> list[ROI]:
    """Linearly rescale ROIs to a different capture resolution."""
    sx = width / 1280
    sy = height / 720
    return [
        ROI(
            name=r.name,
            x1=int(r.x1 * sx),
            y1=int(r.y1 * sy),
            x2=int(r.x2 * sx),
            y2=int(r.y2 * sy),
        )
        for r in layout
    ]


def assign_region(
    cx: float,
    cy: float,
    layout: Iterable[ROI],
) -> RegionName | None:
    for roi in layout:
        if roi.contains(cx, cy):
            return roi.name
    return None
