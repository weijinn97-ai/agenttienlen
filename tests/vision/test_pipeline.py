"""Tests for :class:`StructuredFramePipeline`."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from agenttienlen.core.card import Rank, Suit
from agenttienlen.vision.layout import RegionName, full_layout_1280x720
from agenttienlen.vision.layout_router import LayoutRouter, RegionCrop
from agenttienlen.vision.pipeline import StructuredFramePipeline
from agenttienlen.vision.readers import (
    HandReadResult,
    TableReadResult,
    TurnReadResult,
)
from agenttienlen.vision.structured_frame import (
    ButtonName,
    ButtonState,
    CardCandidate,
    PlayerProfile,
    Rect,
    Seat,
)


class StubHand:
    def __init__(self) -> None:
        self.calls: list[RegionName] = []

    def read(self, crop: RegionCrop) -> HandReadResult:
        self.calls.append(crop.region)
        return HandReadResult(
            candidates=[
                CardCandidate(Rank.THREE, Suit.SPADES, 0.95, Rect(0, 0, 30, 50)),
            ]
        )


class StubTable:
    def read(self, crop: RegionCrop) -> TableReadResult:
        return TableReadResult(
            cards=[CardCandidate(Rank.FOUR, Suit.HEARTS, 0.9, Rect(0, 0, 30, 50))]
        )


class StubOpponent:
    def __init__(self) -> None:
        self.last_keys: list[Seat] = []

    def read(self, crops: Mapping[Seat, RegionCrop]) -> Mapping[Seat, int]:
        self.last_keys = list(crops.keys())
        return {seat: 13 for seat in crops}


class StubIdentity:
    def __init__(self) -> None:
        self.last_avatar_keys: list[Seat] = []
        self.last_name_keys: list[Seat] = []

    def read(
        self,
        avatar_crops: Mapping[Seat, RegionCrop],
        name_crops: Mapping[Seat, RegionCrop],
    ) -> Mapping[Seat, PlayerProfile]:
        self.last_avatar_keys = list(avatar_crops.keys())
        self.last_name_keys = list(name_crops.keys())
        return {
            seat: PlayerProfile(
                seat=seat,
                display_name=f"User{seat.value}",
                raw_name_text=f"User{seat.value}",
                name_confidence=0.9,
                avatar_roi=Rect(0, 0, 100, 100),
                name_roi=Rect(0, 100, 100, 30),
            )
            for seat in avatar_crops
        }


class StubTurn:
    def read(
        self,
        button_crops: Mapping[ButtonName, RegionCrop],
        avatar_crops: Mapping[Seat, RegionCrop],
    ) -> TurnReadResult:
        buttons = {
            name: ButtonState(
                name=name,
                visible=True,
                enabled=True,
                confidence=0.92,
                bbox=Rect(0, 0, 200, 80),
            )
            for name in button_crops
        }
        indicators = {seat: 0.5 for seat in avatar_crops}
        return TurnReadResult(ui_buttons=buttons, turn_indicators=indicators)


def _pipeline() -> tuple[StructuredFramePipeline, StubHand, StubOpponent, StubIdentity]:
    router = LayoutRouter(full_layout_1280x720())
    hand = StubHand()
    opp = StubOpponent()
    ident = StubIdentity()
    pipe = StructuredFramePipeline(
        router=router,
        hand_reader=hand,
        table_reader=StubTable(),
        opponent_reader=opp,
        identity_reader=ident,
        turn_detector=StubTurn(),
    )
    return pipe, hand, opp, ident


def _frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class TestReadAssembly:
    def test_assembles_full_structured_frame(self) -> None:
        pipe, _hand, _opp, _ident = _pipeline()
        frame = pipe.read(_frame(), frame_id=1, timestamp_ms=10)

        assert frame.frame_id == 1
        assert frame.timestamp_ms == 10
        assert len(frame.my_hand_candidates) == 1
        assert frame.my_hand_candidates[0].rank is Rank.THREE
        assert len(frame.table_cards) == 1
        assert set(frame.opponent_back_counts) == {
            Seat.OPP_LEFT,
            Seat.OPP_TOP,
            Seat.OPP_RIGHT,
        }
        assert set(frame.player_profiles) == {
            Seat.ME,
            Seat.OPP_LEFT,
            Seat.OPP_TOP,
            Seat.OPP_RIGHT,
        }
        assert set(frame.ui_buttons) == {ButtonName.PLAY, ButtonName.PASS}
        assert set(frame.turn_indicators) == {
            Seat.ME,
            Seat.OPP_LEFT,
            Seat.OPP_TOP,
            Seat.OPP_RIGHT,
        }
        assert frame.seat_map is None

    def test_seat_map_when_hand_id_provided(self) -> None:
        pipe, *_ = _pipeline()
        frame = pipe.read(
            _frame(),
            frame_id=1,
            timestamp_ms=10,
            hand_id="room131957_g001",
            room_id="131957",
        )
        assert frame.seat_map is not None
        assert frame.seat_map.hand_id == "room131957_g001"
        assert frame.seat_map.display_name(Seat.OPP_TOP) == "Useropp_top"

    def test_dispatches_my_hand_crop_to_hand_reader(self) -> None:
        pipe, hand, _opp, _ident = _pipeline()
        pipe.read(_frame(), frame_id=1, timestamp_ms=10)
        assert hand.calls == [RegionName.MY_HAND]

    def test_passes_only_opponent_seats_to_opponent_reader(self) -> None:
        pipe, _hand, opp, _ident = _pipeline()
        pipe.read(_frame(), frame_id=1, timestamp_ms=10)
        assert set(opp.last_keys) == {Seat.OPP_LEFT, Seat.OPP_TOP, Seat.OPP_RIGHT}
        assert Seat.ME not in opp.last_keys


class TestMissingCrops:
    def test_empty_layout_returns_empty_results(self) -> None:
        # Router with no ROIs → no crops → readers return empty defaults.
        router = LayoutRouter([])
        pipe = StructuredFramePipeline(
            router=router,
            hand_reader=StubHand(),
            table_reader=StubTable(),
            opponent_reader=StubOpponent(),
            identity_reader=StubIdentity(),
            turn_detector=StubTurn(),
        )
        frame = pipe.read(_frame(), frame_id=1, timestamp_ms=10)
        assert frame.my_hand_candidates == []
        assert frame.table_cards == []
        assert frame.opponent_back_counts == {}
        assert frame.player_profiles == {}
        assert frame.ui_buttons == {}
        assert frame.turn_indicators == {}
