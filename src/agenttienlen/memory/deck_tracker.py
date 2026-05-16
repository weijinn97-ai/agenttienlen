"""Track the status of every card in the 52-card deck across a round.

Each of the 52 cards is in exactly one of four buckets:

- ``IN_HAND``     — known to be in the bot's hand
- ``PLAYED``      — has been played to the table (any player)
- ``OPPONENT``    — known to be in a specific opponent's hand (rare; usually unknown)
- ``UNKNOWN``     — face-down in some opponent's hand

The tracker is updated as the vision module reports the bot's hand and as the
orchestrator reports plays from any seat.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from agenttienlen.core.card import Card, full_deck


class CardStatus(Enum):
    UNKNOWN = "unknown"
    IN_HAND = "in_hand"
    OPPONENT = "opponent"
    PLAYED = "played"


class DeckTracker:
    """Maintain status for all 52 cards. Cheap O(1) updates and lookups."""

    def __init__(self) -> None:
        self._status: dict[Card, CardStatus] = dict.fromkeys(full_deck(), CardStatus.UNKNOWN)

    # ---- updates ----

    def set_hand(self, hand: Iterable[Card]) -> None:
        """Mark exactly these cards as IN_HAND; everything else previously
        IN_HAND but not in the new hand becomes PLAYED (we played it)."""
        new_hand = set(hand)
        for card, status in self._status.items():
            if status == CardStatus.IN_HAND and card not in new_hand:
                self._status[card] = CardStatus.PLAYED
        for card in new_hand:
            self._status[card] = CardStatus.IN_HAND

    def mark_played(self, cards: Iterable[Card]) -> None:
        """Mark cards as played to the table (regardless of who played)."""
        for card in cards:
            self._status[card] = CardStatus.PLAYED

    def mark_opponent_known(self, cards: Iterable[Card]) -> None:
        """Rarely used: when vision sees a face-up opponent card before play."""
        for card in cards:
            if self._status[card] == CardStatus.UNKNOWN:
                self._status[card] = CardStatus.OPPONENT

    def reset(self) -> None:
        """Reset for a new round."""
        for card in self._status:
            self._status[card] = CardStatus.UNKNOWN

    # ---- queries ----

    def status(self, card: Card) -> CardStatus:
        return self._status[card]

    def hand(self) -> list[Card]:
        return sorted(c for c, s in self._status.items() if s == CardStatus.IN_HAND)

    def played(self) -> list[Card]:
        return sorted(c for c, s in self._status.items() if s == CardStatus.PLAYED)

    def unseen(self) -> list[Card]:
        """Cards not in our hand and not played yet — i.e. in some opponent's hand."""
        return sorted(
            c for c, s in self._status.items() if s in (CardStatus.UNKNOWN, CardStatus.OPPONENT)
        )

    def __repr__(self) -> str:
        return (
            f"DeckTracker(hand={len(self.hand())} "
            f"played={len(self.played())} "
            f"unseen={len(self.unseen())})"
        )
