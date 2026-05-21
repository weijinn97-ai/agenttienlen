"""Template-matching hand reader.

Implements the :class:`~agenttienlen.vision.readers.HandReader` Protocol
by scanning the ``MY_HAND`` crop for card corners using the same approach
as the legacy ``CardDetectorV2``:

1. **Top-crop grayscale matching** — slide each template's top 30%
   over the crop; high-scoring hits mark card slot positions.
2. **Spatial clustering + NMS** — group matches within ``nms_dist`` px
   into slots; keep the best match per slot.
3. **Suit refinement** — for each slot's top rank, compare all four
   suit variants in colour space; pick the best or mark ``None`` if
   ambiguous.

The reader is stateless — call :meth:`read` per frame. Temporal
stabilisation belongs to the downstream :class:`FrameStabilizer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agenttienlen.core.card import Rank, Suit
from agenttienlen.vision.readers import HandReadResult
from agenttienlen.vision.structured_frame import CardCandidate, Rect
from agenttienlen.vision.template_utils import TemplateStore

if TYPE_CHECKING:
    import numpy as np

    from agenttienlen.vision.layout_router import RegionCrop


# ---- Suit-colour family ----
_RED_SUITS = frozenset({Suit.HEARTS, Suit.DIAMONDS})
_BLACK_SUITS = frozenset({Suit.SPADES, Suit.CLUBS})


@dataclass
class _SlotMatch:
    """Internal: best template match at one card slot."""

    x: int
    y: int
    rank: Rank
    suit: Suit | None
    score: float
    suit_confidence: float


class TemplateHandReader:
    """Concrete :class:`HandReader` backed by template matching.

    Parameters
    ----------
    store:
        Preloaded :class:`TemplateStore` with 52 card templates.
    threshold:
        Minimum ``TM_CCOEFF_NORMED`` score to consider a match.
    nms_dist:
        Pixel distance for non-maximum suppression (same-slot merging).
    suit_margin:
        If the score gap between the top-1 and top-2 suit variants is
        below this, the suit is marked ``None`` (uncertain).
    """

    def __init__(
        self,
        store: TemplateStore,
        *,
        threshold: float = 0.82,
        nms_dist: int = 40,
        suit_margin: float = 0.008,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.nms_dist = nms_dist
        self.suit_margin = suit_margin

    def read(self, crop: RegionCrop) -> HandReadResult:
        """Detect cards in the ``MY_HAND`` region crop."""
        import cv2
        import numpy as np

        image = crop.image
        if image.size == 0:
            return HandReadResult()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        ox, oy = crop.offset

        # Stage 1: find all matches above threshold
        raw_matches: list[tuple[int, int, float, Rank, Suit]] = []
        for entry in self.store.entries:
            tc = entry.top_crop_gray
            if tc.shape[0] > gray.shape[0] or tc.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, tc, cv2.TM_CCOEFF_NORMED)
            locs = np.where(result >= self.threshold)
            for pt_y, pt_x in zip(locs[0], locs[1], strict=True):
                raw_matches.append(
                    (int(pt_x), int(pt_y), float(result[pt_y, pt_x]), entry.rank, entry.suit)
                )

        if not raw_matches:
            return HandReadResult()

        # Stage 2: cluster into slots by x-position, keep best per slot
        slots = self._cluster_slots(raw_matches)

        # Stage 3: for each slot, refine suit among same-rank variants
        candidates: list[CardCandidate] = []
        for slot in slots:
            refined = self._refine_suit(slot, image, gray)
            # Build CardCandidate in frame coordinates
            tmpl_h = self.store.entries[0].top_crop_gray.shape[0]
            tmpl_w = self.store.entries[0].top_crop_gray.shape[1]
            bbox = Rect(
                x=refined.x + ox,
                y=refined.y + oy,
                width=tmpl_w,
                height=tmpl_h * 3,  # approximate full card height
            )
            candidates.append(
                CardCandidate(
                    rank=refined.rank,
                    suit=refined.suit,
                    confidence=refined.score,
                    bbox=bbox,
                )
            )

        # Sort left-to-right by bbox.x
        candidates.sort(key=lambda c: c.bbox.x)
        return HandReadResult(candidates=candidates)

    # ---- internals ----

    def _cluster_slots(
        self,
        raw_matches: list[tuple[int, int, float, Rank, Suit]],
    ) -> list[_SlotMatch]:
        """NMS: group matches into spatial slots, keep best per slot."""
        # Sort by score descending
        sorted_matches = sorted(raw_matches, key=lambda m: -m[2])
        kept: list[_SlotMatch] = []
        for x, y, score, rank, suit in sorted_matches:
            conflict = any(
                abs(x - k.x) < self.nms_dist and abs(y - k.y) < self.nms_dist for k in kept
            )
            if not conflict:
                kept.append(
                    _SlotMatch(
                        x=x,
                        y=y,
                        rank=rank,
                        suit=suit,
                        score=score,
                        suit_confidence=score,
                    )
                )
        return kept

    def _refine_suit(
        self,
        slot: _SlotMatch,
        image_bgr: np.ndarray,
        image_gray: np.ndarray,
    ) -> _SlotMatch:
        """Try to disambiguate suit for a detected rank.

        Strategy: re-run template matching at the slot position for all
        four suits of the detected rank; pick the best if the margin is
        clear, otherwise mark suit as ``None``.
        """
        import cv2

        variants = self.store.by_rank.get(slot.rank, [])
        if len(variants) <= 1:
            return slot

        scores: list[tuple[Suit, float]] = []
        for entry in variants:
            tc = entry.top_crop_gray
            th, tw = tc.shape[:2]
            # Extract the same-sized patch at the slot position
            py1 = max(0, slot.y)
            py2 = min(image_gray.shape[0], slot.y + th)
            px1 = max(0, slot.x)
            px2 = min(image_gray.shape[1], slot.x + tw)
            if py2 - py1 < th or px2 - px1 < tw:
                continue
            patch = image_gray[py1:py2, px1:px2]
            score = cv2.matchTemplate(patch, tc, cv2.TM_CCOEFF_NORMED)
            scores.append((entry.suit, float(score[0, 0])))

        if not scores:
            return slot

        scores.sort(key=lambda s: -s[1])
        best_suit, best_score = scores[0]

        # Check if the suit call is confident
        if len(scores) >= 2:
            margin = best_score - scores[1][1]
            if margin < self.suit_margin:
                # Suit ambiguous — try colour heuristic as tiebreaker
                colour_suit = self._colour_heuristic(
                    image_bgr,
                    slot.x,
                    slot.y,
                    self.store.entries[0].top_crop_gray.shape,
                )
                if colour_suit is not None:
                    # Filter to same-colour family
                    family = _RED_SUITS if colour_suit in _RED_SUITS else _BLACK_SUITS
                    family_scores = [(s, sc) for s, sc in scores if s in family]
                    if family_scores:
                        best_suit = family_scores[0][0]
                        best_score = family_scores[0][1]
                        suit_conf = best_score * 0.85  # discount for heuristic
                        return _SlotMatch(
                            x=slot.x,
                            y=slot.y,
                            rank=slot.rank,
                            suit=best_suit,
                            score=slot.score,
                            suit_confidence=suit_conf,
                        )
                # Still ambiguous
                return _SlotMatch(
                    x=slot.x,
                    y=slot.y,
                    rank=slot.rank,
                    suit=None,
                    score=slot.score,
                    suit_confidence=0.0,
                )

        return _SlotMatch(
            x=slot.x,
            y=slot.y,
            rank=slot.rank,
            suit=best_suit,
            score=slot.score,
            suit_confidence=best_score,
        )

    @staticmethod
    def _colour_heuristic(
        image_bgr: np.ndarray,
        x: int,
        y: int,
        template_shape: tuple[int, ...],
    ) -> Suit | None:
        """Classify red vs black from the rank text area.

        Returns a representative suit from the detected colour family,
        or ``None`` if undetermined.
        """
        import numpy as np

        th, tw = template_shape[:2]
        # Rank-text area: top ~60% of the top-crop, left ~50%
        ry1 = max(0, y)
        ry2 = min(image_bgr.shape[0], y + int(th * 0.6))
        rx1 = max(0, x)
        rx2 = min(image_bgr.shape[1], x + int(tw * 0.5))
        if ry2 <= ry1 or rx2 <= rx1:
            return None

        region = image_bgr[ry1:ry2, rx1:rx2]
        b = region[:, :, 0].astype(np.float32)
        g = region[:, :, 1].astype(np.float32)
        r = region[:, :, 2].astype(np.float32)

        # Red text: pixels with R > 150, R - G > 40, R - B > 40
        red_mask = (r > 150) & ((r - g) > 40) & ((r - b) > 40)
        red_ratio = float(red_mask.sum()) / max(1, red_mask.size)

        # Black text: pixels with V < 80
        v = np.maximum(np.maximum(r, g), b)
        black_mask = v < 80
        black_ratio = float(black_mask.sum()) / max(1, black_mask.size)

        if red_ratio > 0.05:
            return Suit.HEARTS  # representative red suit
        if black_ratio > 0.05:
            return Suit.SPADES  # representative black suit
        return None
