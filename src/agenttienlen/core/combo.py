"""Combo (bộ bài) primitives.

A Combo is an immutable, validated bundle of cards plus a type tag. Use
:func:`agenttienlen.core.rules.classify` to construct one from raw cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agenttienlen.core.card import Card


class ComboType(Enum):
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    STRAIGHT = "straight"
    FOUR_OF_A_KIND = "four_of_a_kind"
    THREE_PAIRS = "three_pairs"
    FOUR_PAIRS = "four_pairs"


# Combos that "chop" (chặt) — only apply when out-of-type override is allowed.
BOMB_TYPES: frozenset[ComboType] = frozenset(
    {
        ComboType.FOUR_OF_A_KIND,
        ComboType.THREE_PAIRS,
        ComboType.FOUR_PAIRS,
    }
)


@dataclass(frozen=True, slots=True)
class Combo:
    """An ordered, validated collection of cards forming a legal play.

    ``cards`` is always sorted by strength ascending. ``key_card`` is the
    highest card and is used for comparisons of the same type/length.
    """

    type: ComboType
    cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        if not self.cards:
            raise ValueError("Combo must contain at least one card")
        # Defensive: ensure sorted-ness so equality and hashing are stable.
        sorted_cards = tuple(sorted(self.cards))
        if sorted_cards != self.cards:
            object.__setattr__(self, "cards", sorted_cards)

    @property
    def size(self) -> int:
        return len(self.cards)

    @property
    def key_card(self) -> Card:
        """Card used for same-type comparisons (always the highest)."""
        return self.cards[-1]

    @property
    def is_bomb(self) -> bool:
        return self.type in BOMB_TYPES

    def __str__(self) -> str:
        return f"{self.type.value}[" + " ".join(str(c) for c in self.cards) + "]"
