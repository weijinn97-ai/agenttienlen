"""Reader contracts for the multi-region vision pipeline.

Each reader is a :class:`typing.Protocol` so individual reader PRs (the
real implementations of ``HandReader``, ``TableReader``, ``OpponentReader``,
``PlayerIdentityReader``, ``TurnDetector``) can be developed in parallel
against a stable interface. The protocols are intentionally narrow — a
reader receives crops and returns its slice of :class:`StructuredFrame`
data; it MUST NOT touch :class:`GameState`, decide who played a card, or
choose an action.

The result dataclasses below are used as the **return type** of each
reader. The pipeline (``StructuredFramePipeline``) assembles them into a
final :class:`StructuredFrame`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from agenttienlen.vision.layout_router import RegionCrop
from agenttienlen.vision.structured_frame import (
    ButtonName,
    ButtonState,
    CardCandidate,
    PlayerProfile,
    Seat,
)


@dataclass(frozen=True, slots=True)
class HandReadResult:
    """Output of :class:`HandReader`."""

    candidates: list[CardCandidate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TableReadResult:
    """Output of :class:`TableReader`. Combo parsing happens downstream."""

    cards: list[CardCandidate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TurnReadResult:
    """Output of :class:`TurnDetector`.

    ``turn_indicators`` is a per-seat confidence in ``[0, 1]`` that the
    seat is the **active** one — derived from highlight ring brightness,
    timer presence, etc. The orchestrator's state machine picks the most
    confident seat (if any crosses the threshold).
    """

    ui_buttons: Mapping[ButtonName, ButtonState] = field(default_factory=dict)
    turn_indicators: Mapping[Seat, float] = field(default_factory=dict)


class HandReader(Protocol):
    """Read the bot's own hand from the ``MY_HAND`` crop.

    Implementations must return one :class:`CardCandidate` per detected
    slot (left-to-right) — even when the rank/suit are uncertain, so the
    stabilizer can vote on the slot count too.
    """

    def read(self, crop: RegionCrop) -> HandReadResult: ...


class TableReader(Protocol):
    """Read the cards currently on the table (last play / combo)."""

    def read(self, crop: RegionCrop) -> TableReadResult: ...


class OpponentReader(Protocol):
    """Count back-card stacks per opponent seat.

    Receives a mapping of ``Seat → back-card crop`` (typically
    ``OPP_BACK_*`` ROIs). Must NOT attempt to identify rank/suit of
    face-down cards.
    """

    def read(self, crops: Mapping[Seat, RegionCrop]) -> Mapping[Seat, int]: ...


class PlayerIdentityReader(Protocol):
    """Resolve seat → :class:`PlayerProfile` from avatar + name crops.

    OCR / template matching lives in the implementation. Returning a
    ``PlayerProfile`` with ``display_name=None`` and a low
    ``name_confidence`` is the correct response when OCR is unsure —
    don't fabricate a name.
    """

    def read(
        self,
        avatar_crops: Mapping[Seat, RegionCrop],
        name_crops: Mapping[Seat, RegionCrop],
    ) -> Mapping[Seat, PlayerProfile]: ...


class TurnDetector(Protocol):
    """Detect action buttons and the active-seat highlight."""

    def read(
        self,
        button_crops: Mapping[ButtonName, RegionCrop],
        avatar_crops: Mapping[Seat, RegionCrop],
    ) -> TurnReadResult: ...
