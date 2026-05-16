"""Screen region (ROI) layout for grouping detections and routing crops.

Two layouts are exposed:

- :func:`default_layout_1280x720` — the **legacy** 5-region map used by
  :class:`agenttienlen.vision.yolo_detector.YoloCardDetector` to bucket
  raw YOLO detections into ``MY_HAND`` / ``TABLE`` / ``OPP_*``. Kept for
  backward compatibility.
- :func:`full_layout_1280x720` — the **spec** map used by
  :class:`agenttienlen.vision.layout_router.LayoutRouter` to crop a frame
  into per-region images for the multi-reader pipeline (avatar, name,
  back-card, play area, buttons, ...).

A detection's region is decided by which ROI rectangle contains its
centre. Coordinates below are tuned against the 1280x720 screenshots in
this repo; other resolutions should call :func:`scale_layout`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class RegionName(Enum):
    # Legacy regions (used by yolo_detector). KHÔNG đổi tên/giá trị.
    MY_HAND = "my_hand"
    TABLE = "table"
    OPP_LEFT = "opp_left"
    OPP_TOP = "opp_top"
    OPP_RIGHT = "opp_right"

    # Finer per-seat play areas (where each opponent's last play appears).
    OPP_PLAY_LEFT = "opp_play_left"
    OPP_PLAY_TOP = "opp_play_top"
    OPP_PLAY_RIGHT = "opp_play_right"

    # Per-seat back-card stacks (used by OpponentReader to count remaining).
    OPP_BACK_LEFT = "opp_back_left"
    OPP_BACK_TOP = "opp_back_top"
    OPP_BACK_RIGHT = "opp_back_right"

    # Per-seat avatar ROIs (used by PlayerIdentityReader + TurnDetector).
    AVATAR_ME = "avatar_me"
    AVATAR_LEFT = "avatar_left"
    AVATAR_TOP = "avatar_top"
    AVATAR_RIGHT = "avatar_right"

    # Per-seat display-name ROIs (used by PlayerIdentityReader for OCR).
    NAME_ME = "name_me"
    NAME_LEFT = "name_left"
    NAME_TOP = "name_top"
    NAME_RIGHT = "name_right"

    # Action buttons (used by TurnDetector).
    BUTTON_PLAY = "button_play"
    BUTTON_PASS = "button_pass"


@dataclass(frozen=True, slots=True)
class ROI:
    name: RegionName
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"ROI dimensions must be non-negative: {self!r}")

    def contains(self, cx: float, cy: float) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def default_layout_1280x720() -> list[ROI]:
    """Legacy ROI rectangles used by :class:`YoloCardDetector`.

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


def full_layout_1280x720() -> list[ROI]:
    """Spec ROI map: all regions used by the multi-reader pipeline.

    Calibrated against the 1280x720 reference screenshots (in-game
    captures with the "Nhất Ăn Tất" table). Each ROI is a generous
    bounding box — readers refine internally if needed.
    """
    return [
        # MY_HAND fan along the bottom edge.
        ROI(RegionName.MY_HAND, 135, 510, 1140, 720),
        # No central table in this variant — opponents' plays appear at
        # their seats — but keep a wide TABLE rect covering the union for
        # legacy / fallback readers.
        ROI(RegionName.TABLE, 170, 10, 1110, 340),
        # Legacy wide per-seat regions (used by yolo_detector grouping).
        ROI(RegionName.OPP_LEFT, 150, 200, 450, 340),
        ROI(RegionName.OPP_TOP, 500, 30, 850, 200),
        ROI(RegionName.OPP_RIGHT, 830, 200, 1130, 340),
        # Finer play-area rects (where opponents' last-played cards land).
        ROI(RegionName.OPP_PLAY_LEFT, 170, 240, 290, 340),
        ROI(RegionName.OPP_PLAY_TOP, 700, 10, 820, 180),
        ROI(RegionName.OPP_PLAY_RIGHT, 985, 240, 1110, 340),
        # Opponent back-card stacks (count only — no rank/suit reading).
        ROI(RegionName.OPP_BACK_LEFT, 130, 245, 205, 320),
        ROI(RegionName.OPP_BACK_TOP, 805, 50, 880, 160),
        ROI(RegionName.OPP_BACK_RIGHT, 1080, 245, 1150, 320),
        # Avatar circles per seat.
        ROI(RegionName.AVATAR_ME, 25, 545, 130, 660),
        ROI(RegionName.AVATAR_LEFT, 25, 195, 130, 300),
        ROI(RegionName.AVATAR_TOP, 870, 20, 985, 130),
        ROI(RegionName.AVATAR_RIGHT, 1150, 195, 1255, 300),
        # Display-name strips below each avatar.
        ROI(RegionName.NAME_ME, 25, 660, 130, 700),
        ROI(RegionName.NAME_LEFT, 25, 305, 130, 340),
        ROI(RegionName.NAME_TOP, 870, 130, 985, 162),
        ROI(RegionName.NAME_RIGHT, 1150, 305, 1255, 340),
        # Action buttons. When both visible they sit side-by-side; when
        # only "Đánh" is shown it is centred — these rects span both
        # positions so the detector picks up either layout.
        ROI(RegionName.BUTTON_PASS, 355, 370, 590, 455),
        ROI(RegionName.BUTTON_PLAY, 555, 370, 925, 455),
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
