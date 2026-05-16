"""Route a full frame into per-region crops for the multi-reader pipeline.

:class:`LayoutRouter` is the **single** entry point that turns a raw
1280x720 BGR/RGB ``np.ndarray`` into a ``dict[RegionName, RegionCrop]``.
It does NOT parse cards, OCR names, or detect buttons — that is the
responsibility of each downstream reader. Keeping the router this dumb
makes ROI calibration the only thing that needs to be tuned per game
variant.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agenttienlen.vision.layout import ROI, RegionName, full_layout_1280x720

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class RegionCrop:
    """A cropped view of one ROI inside the source frame.

    ``image`` is the cropped sub-array (typically a numpy view, no copy).
    ``offset`` is the ROI's top-left corner in the source frame so a
    reader can translate local detection coordinates back to the full
    frame (e.g. ``frame_x = local_x + offset_x``).
    """

    region: RegionName
    roi: ROI
    image: np.ndarray
    offset: tuple[int, int]

    @property
    def width(self) -> int:
        return self.roi.width

    @property
    def height(self) -> int:
        return self.roi.height


class LayoutRouter:
    """Crop a frame into per-region sub-images.

    The router clips ROIs against the actual frame bounds so an ROI that
    extends past the edge (e.g. due to mild calibration drift) still
    produces a non-empty crop.
    """

    def __init__(self, layout: Iterable[ROI] | None = None) -> None:
        self.layout: list[ROI] = list(layout) if layout is not None else full_layout_1280x720()

    def split(self, frame: np.ndarray) -> dict[RegionName, RegionCrop]:
        """Return one :class:`RegionCrop` per ROI in the layout.

        ROIs whose clipped area is empty (entirely outside the frame) are
        skipped — they would otherwise produce a 0-sized sub-array that
        readers cannot meaningfully consume.
        """
        if frame.ndim < 2:
            raise ValueError(f"frame must be 2D (HxW[xC]); got shape {frame.shape!r}")
        height, width = frame.shape[0], frame.shape[1]

        crops: dict[RegionName, RegionCrop] = {}
        for roi in self.layout:
            x1 = max(0, min(width, roi.x1))
            y1 = max(0, min(height, roi.y1))
            x2 = max(0, min(width, roi.x2))
            y2 = max(0, min(height, roi.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crops[roi.name] = RegionCrop(
                region=roi.name,
                roi=roi,
                image=frame[y1:y2, x1:x2],
                offset=(x1, y1),
            )
        return crops

    def crop(self, frame: np.ndarray, region: RegionName) -> RegionCrop | None:
        """Convenience: crop a single named region (or ``None`` if missing)."""
        for roi in self.layout:
            if roi.name is region:
                return self.split(frame).get(region)
        return None
