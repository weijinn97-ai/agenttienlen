"""Template-matching table reader.

Implements the :class:`~agenttienlen.vision.readers.TableReader` Protocol
by scanning the ``TABLE`` crop (or opponent-play-area crops) for face-up
cards using multi-scale template matching.

Face-up cards on the table are rendered at roughly 55-60% of the
template's original size, so the reader tests a range of scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agenttienlen.core.card import Rank, Suit
from agenttienlen.vision.readers import TableReadResult
from agenttienlen.vision.structured_frame import CardCandidate, Rect
from agenttienlen.vision.template_utils import TemplateStore

if TYPE_CHECKING:
    from agenttienlen.vision.layout_router import RegionCrop


@dataclass
class _TableMatch:
    """Internal: one detected card on the table."""

    x: int
    y: int
    rank: Rank
    suit: Suit
    score: float
    scale: float
    w: int
    h: int


class TemplateTableReader:
    """Concrete :class:`TableReader` backed by multi-scale template matching.

    Parameters
    ----------
    store:
        Preloaded :class:`TemplateStore` with 52 card templates.
    scales:
        Scale factors to try when matching templates against the table.
    threshold:
        Minimum ``TM_CCOEFF_NORMED`` score.
    nms_dist:
        Pixel distance for non-maximum suppression.
    """

    def __init__(
        self,
        store: TemplateStore,
        *,
        scales: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65),
        threshold: float = 0.72,
        nms_dist: int = 25,
    ) -> None:
        self.store = store
        self.scales = scales
        self.threshold = threshold
        self.nms_dist = nms_dist

    def read(self, crop: RegionCrop) -> TableReadResult:
        """Detect face-up cards in the ``TABLE`` region crop."""
        import cv2
        import numpy as np

        image = crop.image
        if image.size == 0:
            return TableReadResult()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        ox, oy = crop.offset

        raw_matches: list[_TableMatch] = []
        for entry in self.store.entries:
            for scale in self.scales:
                th = max(1, int(entry.gray.shape[0] * scale))
                tw = max(1, int(entry.gray.shape[1] * scale))
                if th > gray.shape[0] or tw > gray.shape[1]:
                    continue
                resized = cv2.resize(entry.gray, (tw, th))
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                locs = np.where(result >= self.threshold)
                for pt_y, pt_x in zip(locs[0], locs[1], strict=True):
                    raw_matches.append(
                        _TableMatch(
                            x=int(pt_x),
                            y=int(pt_y),
                            rank=entry.rank,
                            suit=entry.suit,
                            score=float(result[pt_y, pt_x]),
                            scale=scale,
                            w=tw,
                            h=th,
                        )
                    )

        if not raw_matches:
            return TableReadResult()

        # NMS: keep best match per spatial cluster
        kept = self._nms(raw_matches)

        candidates: list[CardCandidate] = []
        for m in kept:
            bbox = Rect(
                x=m.x + ox,
                y=m.y + oy,
                width=m.w,
                height=m.h,
            )
            candidates.append(
                CardCandidate(
                    rank=m.rank,
                    suit=m.suit,
                    confidence=m.score,
                    bbox=bbox,
                )
            )

        candidates.sort(key=lambda c: c.bbox.x)
        return TableReadResult(cards=candidates)

    def _nms(self, matches: list[_TableMatch]) -> list[_TableMatch]:
        """Non-maximum suppression over detected cards."""
        sorted_matches = sorted(matches, key=lambda m: -m.score)
        kept: list[_TableMatch] = []
        for m in sorted_matches:
            conflict = any(
                abs(m.x - k.x) < self.nms_dist and abs(m.y - k.y) < self.nms_dist for k in kept
            )
            if not conflict:
                kept.append(m)
        return kept
