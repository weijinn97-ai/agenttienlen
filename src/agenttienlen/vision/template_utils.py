"""Template loading and Vietnamese card-name → Card mapping.

Templates are cropped card images (52 files, e.g. ``3Bich.png``,
``AtRo.png``) extracted from the game client. This module loads them
into rank/suit-indexed structures so :class:`TemplateHandReader` and
:class:`TemplateTableReader` can use them for detection.

The naming convention in ``cards_output/``:

- Rank: ``3``-``10``, ``J``, ``Q``, ``K``, ``At`` (Ace)
- Suit: ``Bich`` (♠), ``Chuon`` (♣), ``Co`` (♥), ``Ro`` (♦)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agenttienlen.core.card import Card, Rank, Suit

if TYPE_CHECKING:
    import numpy as np

# ---- Vietnamese → Enum mappings ----

_VN_RANK: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "At": Rank.ACE,
}

_VN_SUIT: dict[str, Suit] = {
    "Bich": Suit.SPADES,
    "Chuon": Suit.CLUBS,
    "Co": Suit.HEARTS,
    "Ro": Suit.DIAMONDS,
}

# Precompiled regex: e.g. "10Bich", "AtRo", "JCo"
_TEMPLATE_NAME_RE = re.compile(r"^(10|[2-9]|J|Q|K|At)(Bich|Chuon|Co|Ro)$")


def parse_template_name(name: str) -> Card | None:
    """Parse a Vietnamese template name into a :class:`Card`.

    >>> parse_template_name("AtRo")
    Card(AD)
    >>> parse_template_name("10Bich")
    Card(10S)
    """
    m = _TEMPLATE_NAME_RE.match(name)
    if m is None:
        return None
    rank_str, suit_str = m.group(1), m.group(2)
    return Card(rank=_VN_RANK[rank_str], suit=_VN_SUIT[suit_str])


def card_to_template_name(card: Card) -> str:
    """Inverse of :func:`parse_template_name`.

    >>> card_to_template_name(Card.parse("AS"))
    'AtBich'
    """
    rank_str = next(k for k, v in _VN_RANK.items() if v == card.rank)
    suit_str = next(k for k, v in _VN_SUIT.items() if v == card.suit)
    return f"{rank_str}{suit_str}"


@dataclass
class TemplateEntry:
    """One loaded template image with its :class:`Card` identity."""

    card: Card
    image: np.ndarray  # BGR, full template
    gray: np.ndarray  # grayscale, full template
    top_crop_gray: np.ndarray  # grayscale, top ``crop_pct`` of template

    @property
    def rank(self) -> Rank:
        return self.card.rank

    @property
    def suit(self) -> Suit:
        return self.card.suit


@dataclass
class TemplateStore:
    """Collection of all 52 templates, indexed by rank for fast lookup."""

    entries: list[TemplateEntry] = field(default_factory=list)
    by_rank: dict[Rank, list[TemplateEntry]] = field(default_factory=dict)
    by_card: dict[Card, TemplateEntry] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        template_dir: str | Path,
        *,
        crop_top_pct: int = 30,
    ) -> TemplateStore:
        """Load all ``*.png`` files from *template_dir*.

        Parameters
        ----------
        template_dir:
            Directory containing ``<rank><suit>.png`` files.
        crop_top_pct:
            Percentage of template height to keep as the top crop
            (used for initial grayscale rank matching). Default 30%.
        """
        import cv2

        template_dir = Path(template_dir)
        entries: list[TemplateEntry] = []
        by_rank: dict[Rank, list[TemplateEntry]] = {}
        by_card: dict[Card, TemplateEntry] = {}

        for fn in sorted(template_dir.iterdir()):
            if fn.suffix.lower() != ".png":
                continue
            card = parse_template_name(fn.stem)
            if card is None:
                continue
            img = cv2.imread(str(fn))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h = gray.shape[0]
            top_h = max(1, int(h * crop_top_pct / 100))
            top_crop = gray[:top_h, :]

            entry = TemplateEntry(card=card, image=img, gray=gray, top_crop_gray=top_crop)
            entries.append(entry)
            by_rank.setdefault(card.rank, []).append(entry)
            by_card[card] = entry

        return cls(entries=entries, by_rank=by_rank, by_card=by_card)

    @property
    def count(self) -> int:
        return len(self.entries)
