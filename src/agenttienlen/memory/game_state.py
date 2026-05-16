"""High-level game state shared across vision → agent → io_ctrl."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agenttienlen.core.card import Card
from agenttienlen.core.combo import Combo
from agenttienlen.memory.deck_tracker import DeckTracker


class PlayerSeat(Enum):
    BOTTOM = "bottom"  # the bot
    LEFT = "left"
    TOP = "top"
    RIGHT = "right"


@dataclass(slots=True)
class GameState:
    """The full mutable state the agent consumes when deciding a move.

    Updated by the orchestrator after every vision frame.
    """

    hand: list[Card] = field(default_factory=list)
    """Cards currently in the bot's hand, sorted ascending."""

    current_combo: Combo | None = None
    """The combo on the table that the bot must beat, or None if leading."""

    last_player: PlayerSeat | None = None
    """Who played `current_combo`. None at start of round."""

    seat_card_counts: dict[PlayerSeat, int] = field(default_factory=dict)
    """How many cards each opponent still holds (from the badge on their avatar)."""

    passed_this_round: set[PlayerSeat] = field(default_factory=set)
    """Seats that have passed in the current trick."""

    is_first_round: bool = True
    """True only for the very first trick of the game (3♠ must lead)."""

    deck: DeckTracker = field(default_factory=DeckTracker)
    """52-card status tracker."""

    def is_leading(self) -> bool:
        """True when the bot is free to play any combo (new trick)."""
        return self.current_combo is None

    def opponents_remaining(self) -> int:
        """Count of opponents who still have cards."""
        return sum(
            1 for seat, n in self.seat_card_counts.items() if seat != PlayerSeat.BOTTOM and n > 0
        )
