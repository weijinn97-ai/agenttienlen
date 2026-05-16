"""Top-level vision pipeline: route → 5 readers → :class:`StructuredFrame`.

This is the skeleton that wires everything together. The actual readers
are :class:`typing.Protocol` instances — production code injects the
real implementations (HandReader corner classifier, TableReader YOLO,
...), while tests inject deterministic mock readers.

The pipeline:

1. Calls :meth:`LayoutRouter.split` to crop the frame into per-region
   sub-images.
2. Dispatches each region crop to the matching reader.
3. Assembles the per-reader results into one :class:`StructuredFrame`.

The pipeline **does not** stabilize, mutate :class:`GameState`, or pick
an action — those belong to the stabilizer and the orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from agenttienlen.vision.layout import RegionName
from agenttienlen.vision.layout_router import LayoutRouter, RegionCrop
from agenttienlen.vision.readers import (
    HandReader,
    HandReadResult,
    OpponentReader,
    PlayerIdentityReader,
    TableReader,
    TableReadResult,
    TurnDetector,
    TurnReadResult,
)
from agenttienlen.vision.structured_frame import (
    ButtonName,
    PlayerProfile,
    Seat,
    SeatMap,
    StructuredFrame,
)

if TYPE_CHECKING:
    import numpy as np


_OPP_BACK_BY_SEAT: dict[Seat, RegionName] = {
    Seat.OPP_LEFT: RegionName.OPP_BACK_LEFT,
    Seat.OPP_TOP: RegionName.OPP_BACK_TOP,
    Seat.OPP_RIGHT: RegionName.OPP_BACK_RIGHT,
}

_AVATAR_BY_SEAT: dict[Seat, RegionName] = {
    Seat.ME: RegionName.AVATAR_ME,
    Seat.OPP_LEFT: RegionName.AVATAR_LEFT,
    Seat.OPP_TOP: RegionName.AVATAR_TOP,
    Seat.OPP_RIGHT: RegionName.AVATAR_RIGHT,
}

_NAME_BY_SEAT: dict[Seat, RegionName] = {
    Seat.ME: RegionName.NAME_ME,
    Seat.OPP_LEFT: RegionName.NAME_LEFT,
    Seat.OPP_TOP: RegionName.NAME_TOP,
    Seat.OPP_RIGHT: RegionName.NAME_RIGHT,
}

_BUTTON_BY_NAME: dict[ButtonName, RegionName] = {
    ButtonName.PLAY: RegionName.BUTTON_PLAY,
    ButtonName.PASS: RegionName.BUTTON_PASS,
}


class StructuredFramePipeline:
    """Coordinate :class:`LayoutRouter` + 5 readers into one structured tick.

    Construct once at boot, call :meth:`read` per frame. The pipeline is
    stateless; if a reader needs to remember anything (templates, last
    OCR result), it owns that state internally.
    """

    def __init__(
        self,
        router: LayoutRouter,
        hand_reader: HandReader,
        table_reader: TableReader,
        opponent_reader: OpponentReader,
        identity_reader: PlayerIdentityReader,
        turn_detector: TurnDetector,
    ) -> None:
        self.router = router
        self.hand_reader = hand_reader
        self.table_reader = table_reader
        self.opponent_reader = opponent_reader
        self.identity_reader = identity_reader
        self.turn_detector = turn_detector

    def read(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: int,
        hand_id: str | None = None,
        room_id: str | None = None,
    ) -> StructuredFrame:
        """Run all five readers on ``frame`` and return a structured tick."""
        crops = self.router.split(frame)

        hand_result = self._read_hand(crops)
        table_result = self._read_table(crops)
        back_counts = self._read_opponents(crops)
        profiles = self._read_identities(crops)
        turn_result = self._read_turn(crops)

        seat_map = (
            SeatMap(hand_id=hand_id, room_id=room_id, players=profiles)
            if hand_id is not None
            else None
        )

        return StructuredFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            my_hand_candidates=list(hand_result.candidates),
            table_cards=list(table_result.cards),
            opponent_back_counts=back_counts,
            player_profiles=profiles,
            ui_buttons=dict(turn_result.ui_buttons),
            turn_indicators=dict(turn_result.turn_indicators),
            seat_map=seat_map,
        )

    # ---- helpers ----

    def _read_hand(self, crops: Mapping[RegionName, RegionCrop]) -> HandReadResult:
        crop = crops.get(RegionName.MY_HAND)
        if crop is None:
            return HandReadResult()
        return self.hand_reader.read(crop)

    def _read_table(self, crops: Mapping[RegionName, RegionCrop]) -> TableReadResult:
        crop = crops.get(RegionName.TABLE)
        if crop is None:
            return TableReadResult()
        return self.table_reader.read(crop)

    def _read_opponents(self, crops: Mapping[RegionName, RegionCrop]) -> Mapping[Seat, int]:
        seat_crops = {
            seat: crops[region] for seat, region in _OPP_BACK_BY_SEAT.items() if region in crops
        }
        if not seat_crops:
            return {}
        return self.opponent_reader.read(seat_crops)

    def _read_identities(
        self, crops: Mapping[RegionName, RegionCrop]
    ) -> Mapping[Seat, PlayerProfile]:
        avatar_crops = {
            seat: crops[region] for seat, region in _AVATAR_BY_SEAT.items() if region in crops
        }
        name_crops = {
            seat: crops[region] for seat, region in _NAME_BY_SEAT.items() if region in crops
        }
        if not avatar_crops and not name_crops:
            return {}
        return self.identity_reader.read(avatar_crops, name_crops)

    def _read_turn(self, crops: Mapping[RegionName, RegionCrop]) -> TurnReadResult:
        button_crops = {
            button: crops[region] for button, region in _BUTTON_BY_NAME.items() if region in crops
        }
        avatar_crops = {
            seat: crops[region] for seat, region in _AVATAR_BY_SEAT.items() if region in crops
        }
        if not button_crops and not avatar_crops:
            return TurnReadResult()
        return self.turn_detector.read(button_crops, avatar_crops)
