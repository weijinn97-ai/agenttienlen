"""YOLO class labels: 52 cards + 1 'back' class.

Class id = rank * 4 + suit_index (0..51), with 52 reserved for the card-back.
Suit order matches :class:`agenttienlen.core.card.Suit`.
"""

from __future__ import annotations

from agenttienlen.core.card import Card, Rank, Suit

NUM_CLASSES = 53
BACK_CLASS_ID = 52


def card_to_class_id(card: Card) -> int:
    return card.strength  # = rank * 4 + suit


def class_id_to_card(class_id: int) -> Card | None:
    """Return the Card for a class id, or None for the 'back' class."""
    if class_id == BACK_CLASS_ID:
        return None
    if not 0 <= class_id < 52:
        raise ValueError(f"class_id out of range: {class_id}")
    return Card.from_strength(class_id)


def _name(card: Card) -> str:
    return f"{card.rank.label}{card.suit.letter}"


CARD_CLASS_NAMES: list[str] = []
for rank in Rank:
    for suit in Suit:
        CARD_CLASS_NAMES.append(_name(Card(rank, suit)))
CARD_CLASS_NAMES.append("back")
assert len(CARD_CLASS_NAMES) == NUM_CLASSES
