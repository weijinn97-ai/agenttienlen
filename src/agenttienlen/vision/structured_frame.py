"""Structured output of the region-based vision pipeline.

This module defines the **contract** between the per-region readers and the
rest of the bot (memory / agent / orchestrator). Per the spec, vision does
NOT return a flat ``list[Card]``; it returns a :class:`StructuredFrame` that
keeps each region's data separate so :class:`GameState` and `TurnDetector`
can decide whose play each card belongs to.

All types here are pure data — no image, no model, no I/O. They are imported
freely by tests and by every reader implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from agenttienlen.core.card import Card, Rank, Suit


class Seat(Enum):
    """A physical seat at the 4-player table.

    Distinct from :class:`agenttienlen.memory.game_state.PlayerSeat` on
    purpose: this is the **vision-side** seat label (matching the spec's
    ``ME / OPP_LEFT / OPP_TOP / OPP_RIGHT`` terminology), while
    ``PlayerSeat`` is the **memory-side** label. The orchestrator maps
    between them.
    """

    ME = "me"
    OPP_LEFT = "opp_left"
    OPP_TOP = "opp_top"
    OPP_RIGHT = "opp_right"


class ButtonName(Enum):
    """Action buttons surfaced by :class:`TurnDetector`."""

    PLAY = "play"  # "Đánh"
    PASS = "pass"  # "Bỏ lượt"


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in pixel coordinates ``(x, y, width, height)``.

    ``x``/``y`` are the top-left corner. Used for every bounding box that
    flows out of vision (card detections, button areas, avatar crops, ...).
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Rect dimensions must be non-negative: {self!r}")

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True, slots=True)
class CardCandidate:
    """A single card detection from any region.

    ``rank`` / ``suit`` may be ``None`` if the reader could not identify
    one or both (e.g. corner classifier returned low confidence). Callers
    should treat partial candidates as **unknown** and not commit them to
    ``GameState`` until they stabilise.
    """

    rank: Rank | None
    suit: Suit | None
    confidence: float
    bbox: Rect

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")

    @property
    def is_known(self) -> bool:
        return self.rank is not None and self.suit is not None

    @property
    def card(self) -> Card | None:
        """Materialise a :class:`Card` if both rank and suit are known."""
        if self.rank is None or self.suit is None:
            return None
        return Card(rank=self.rank, suit=self.suit)


@dataclass(frozen=True, slots=True)
class ButtonState:
    """Visibility / enablement of an action button at one frame."""

    name: ButtonName
    visible: bool
    enabled: bool
    confidence: float
    bbox: Rect

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Display identity of one player as read by :class:`PlayerIdentityReader`.

    ``display_name`` is the cleaned label (after OCR + normalisation);
    ``raw_name_text`` keeps the raw OCR output for debugging when the
    cleaned version is wrong. Both may be ``None`` if OCR has not produced
    a stable reading yet.

    ``avatar_roi`` and ``name_roi`` are the frame-coordinate boxes the
    reader looked at, so logs can re-crop the same regions for replay.
    """

    seat: Seat
    display_name: str | None
    raw_name_text: str | None
    name_confidence: float
    avatar_roi: Rect
    name_roi: Rect
    is_bot: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.name_confidence <= 1.0:
            raise ValueError(f"name_confidence must be in [0, 1]: {self.name_confidence}")


@dataclass(frozen=True, slots=True)
class SeatMap:
    """Snapshot of the 4-seat ↔ display-name mapping for one hand.

    Captured by the orchestrator at the start of every new hand (or when
    seat changes are detected) and attached to every log event so replay
    can identify which real account each ``Seat`` corresponded to.
    """

    hand_id: str
    room_id: str | None
    players: Mapping[Seat, PlayerProfile]

    def display_name(self, seat: Seat) -> str | None:
        profile = self.players.get(seat)
        return profile.display_name if profile is not None else None


@dataclass(frozen=True, slots=True)
class StructuredFrame:
    """The full output of one vision tick, organised by region.

    The spec is strict about this: vision must NOT return a single flat
    list of cards. Each field below comes from a different reader looking
    at a different ROI, and downstream code decides ownership / legality.
    """

    frame_id: int
    timestamp_ms: int
    my_hand_candidates: list[CardCandidate] = field(default_factory=list)
    table_cards: list[CardCandidate] = field(default_factory=list)
    opponent_back_counts: Mapping[Seat, int] = field(default_factory=dict)
    player_profiles: Mapping[Seat, PlayerProfile] = field(default_factory=dict)
    ui_buttons: Mapping[ButtonName, ButtonState] = field(default_factory=dict)
    turn_indicators: Mapping[Seat, float] = field(default_factory=dict)
    seat_map: SeatMap | None = None
