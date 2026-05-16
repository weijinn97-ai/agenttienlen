"""Tests for the structured-frame dataclasses (Rect, CardCandidate, ...)."""

from __future__ import annotations

import pytest

from agenttienlen.core.card import Card, Rank, Suit
from agenttienlen.vision.structured_frame import (
    ButtonName,
    ButtonState,
    CardCandidate,
    PlayerProfile,
    Rect,
    Seat,
    SeatMap,
    StructuredFrame,
)


class TestRect:
    def test_corners_and_centre(self) -> None:
        r = Rect(10, 20, 30, 40)
        assert r.x1 == 10
        assert r.y1 == 20
        assert r.x2 == 40
        assert r.y2 == 60
        assert r.cx == 25.0
        assert r.cy == 40.0
        assert r.area == 1200

    def test_zero_size_is_allowed(self) -> None:
        Rect(0, 0, 0, 0)

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Rect(0, 0, -1, 10)
        with pytest.raises(ValueError, match="non-negative"):
            Rect(0, 0, 10, -1)

    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [
            (10, 20, True),
            (40, 60, True),
            (9, 20, False),
            (41, 20, False),
            (25, 60.1, False),
        ],
    )
    def test_contains(self, x: float, y: float, expected: bool) -> None:
        assert Rect(10, 20, 30, 40).contains(x, y) is expected


class TestCardCandidate:
    def test_card_materialises_when_known(self) -> None:
        c = CardCandidate(Rank.ACE, Suit.HEARTS, 0.9, Rect(0, 0, 10, 10))
        assert c.is_known is True
        assert c.card == Card(Rank.ACE, Suit.HEARTS)

    def test_card_is_none_when_partial(self) -> None:
        assert CardCandidate(None, Suit.HEARTS, 0.5, Rect(0, 0, 1, 1)).card is None
        assert CardCandidate(Rank.TWO, None, 0.5, Rect(0, 0, 1, 1)).card is None
        assert CardCandidate(None, None, 0.5, Rect(0, 0, 1, 1)).card is None

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            CardCandidate(Rank.THREE, Suit.SPADES, -0.1, Rect(0, 0, 1, 1))
        with pytest.raises(ValueError):
            CardCandidate(Rank.THREE, Suit.SPADES, 1.1, Rect(0, 0, 1, 1))


class TestButtonState:
    def test_basic_attributes(self) -> None:
        bs = ButtonState(
            name=ButtonName.PLAY,
            visible=True,
            enabled=False,
            confidence=0.8,
            bbox=Rect(500, 370, 200, 80),
        )
        assert bs.name is ButtonName.PLAY
        assert bs.visible is True
        assert bs.enabled is False

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            ButtonState(ButtonName.PASS, True, True, 1.5, Rect(0, 0, 1, 1))


class TestPlayerProfileAndSeatMap:
    def _profile(self, seat: Seat, name: str | None = None) -> PlayerProfile:
        return PlayerProfile(
            seat=seat,
            display_name=name,
            raw_name_text=name,
            name_confidence=0.95 if name else 0.0,
            avatar_roi=Rect(0, 0, 100, 100),
            name_roi=Rect(0, 100, 100, 30),
        )

    def test_seat_map_display_name_lookup(self) -> None:
        seat_map = SeatMap(
            hand_id="room131957_game001",
            room_id="131957",
            players={
                Seat.OPP_LEFT: self._profile(Seat.OPP_LEFT, "User0252398"),
                Seat.OPP_TOP: self._profile(Seat.OPP_TOP, "User5197217"),
            },
        )
        assert seat_map.display_name(Seat.OPP_LEFT) == "User0252398"
        assert seat_map.display_name(Seat.OPP_TOP) == "User5197217"
        # Missing seat → None, not KeyError.
        assert seat_map.display_name(Seat.OPP_RIGHT) is None

    def test_name_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            PlayerProfile(
                seat=Seat.ME,
                display_name=None,
                raw_name_text=None,
                name_confidence=1.5,
                avatar_roi=Rect(0, 0, 1, 1),
                name_roi=Rect(0, 0, 1, 1),
            )


class TestStructuredFrame:
    def test_empty_defaults(self) -> None:
        frame = StructuredFrame(frame_id=42, timestamp_ms=1715840923123)
        assert frame.frame_id == 42
        assert frame.timestamp_ms == 1715840923123
        assert frame.my_hand_candidates == []
        assert frame.table_cards == []
        assert frame.opponent_back_counts == {}
        assert frame.player_profiles == {}
        assert frame.ui_buttons == {}
        assert frame.turn_indicators == {}
        assert frame.seat_map is None

    def test_full_population(self) -> None:
        candidate = CardCandidate(Rank.THREE, Suit.SPADES, 0.91, Rect(0, 0, 60, 100))
        frame = StructuredFrame(
            frame_id=1,
            timestamp_ms=1,
            my_hand_candidates=[candidate],
            table_cards=[candidate],
            opponent_back_counts={Seat.OPP_LEFT: 13, Seat.OPP_TOP: 12, Seat.OPP_RIGHT: 11},
            ui_buttons={
                ButtonName.PLAY: ButtonState(
                    ButtonName.PLAY, True, True, 0.99, Rect(700, 370, 200, 80)
                )
            },
            turn_indicators={Seat.ME: 0.88, Seat.OPP_LEFT: 0.05},
        )
        assert frame.my_hand_candidates[0].card == Card(Rank.THREE, Suit.SPADES)
        assert frame.opponent_back_counts[Seat.OPP_LEFT] == 13
        assert frame.ui_buttons[ButtonName.PLAY].enabled is True
