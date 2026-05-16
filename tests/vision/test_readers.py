"""Smoke tests for reader Protocols and their result dataclasses."""

from __future__ import annotations

from collections.abc import Mapping

from agenttienlen.core.card import Rank, Suit
from agenttienlen.vision.layout import ROI, RegionName
from agenttienlen.vision.layout_router import RegionCrop
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
    ButtonState,
    CardCandidate,
    PlayerProfile,
    Rect,
    Seat,
)


def _crop(region: RegionName = RegionName.MY_HAND) -> RegionCrop:
    import numpy as np

    roi = ROI(region, 0, 0, 10, 10)
    return RegionCrop(
        region=region,
        roi=roi,
        image=np.zeros((roi.height, roi.width, 3), dtype=np.uint8),
        offset=(0, 0),
    )


class StubHandReader:
    def read(self, crop: RegionCrop) -> HandReadResult:
        return HandReadResult(
            candidates=[
                CardCandidate(Rank.THREE, Suit.SPADES, 0.9, Rect(0, 0, 30, 50)),
            ]
        )


class StubTableReader:
    def read(self, crop: RegionCrop) -> TableReadResult:
        return TableReadResult(cards=[])


class StubOpponentReader:
    def read(self, crops: Mapping[Seat, RegionCrop]) -> Mapping[Seat, int]:
        return {seat: 13 for seat in crops}


class StubPlayerIdentityReader:
    def read(
        self,
        avatar_crops: Mapping[Seat, RegionCrop],
        name_crops: Mapping[Seat, RegionCrop],
    ) -> Mapping[Seat, PlayerProfile]:
        return {}


class StubTurnDetector:
    def read(
        self,
        button_crops: Mapping[ButtonName, RegionCrop],
        avatar_crops: Mapping[Seat, RegionCrop],
    ) -> TurnReadResult:
        return TurnReadResult(
            ui_buttons={
                ButtonName.PLAY: ButtonState(
                    name=ButtonName.PLAY,
                    visible=True,
                    enabled=True,
                    confidence=0.95,
                    bbox=Rect(700, 370, 200, 80),
                )
            },
            turn_indicators={Seat.ME: 0.9},
        )


class TestProtocols:
    def test_stub_implementations_satisfy_protocols(self) -> None:
        # Structural typing — assigning a stub to a Protocol-typed variable
        # must type-check at runtime as a plain assignment.
        hand: HandReader = StubHandReader()
        table: TableReader = StubTableReader()
        opp: OpponentReader = StubOpponentReader()
        ident: PlayerIdentityReader = StubPlayerIdentityReader()
        turn: TurnDetector = StubTurnDetector()

        assert hand.read(_crop()).candidates[0].card is not None
        assert table.read(_crop(RegionName.TABLE)).cards == []
        assert opp.read({Seat.OPP_LEFT: _crop(RegionName.OPP_BACK_LEFT)}) == {Seat.OPP_LEFT: 13}
        assert ident.read({}, {}) == {}
        assert turn.read({}, {}).ui_buttons[ButtonName.PLAY].enabled is True


class TestResultDefaults:
    def test_empty_defaults(self) -> None:
        assert HandReadResult().candidates == []
        assert TableReadResult().cards == []
        result = TurnReadResult()
        assert result.ui_buttons == {}
        assert result.turn_indicators == {}
