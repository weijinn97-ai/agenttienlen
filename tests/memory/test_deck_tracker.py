from __future__ import annotations

from agenttienlen.core.card import Card
from agenttienlen.memory.deck_tracker import CardStatus, DeckTracker


def H(s: str) -> list[Card]:
    return [Card.parse(t) for t in s.split()]


def test_initial_all_unknown() -> None:
    t = DeckTracker()
    assert len(t.unseen()) == 52
    assert t.hand() == []
    assert t.played() == []


def test_set_hand_marks_cards() -> None:
    t = DeckTracker()
    t.set_hand(H("3S 3D 4C"))
    assert sorted(t.hand()) == sorted(H("3S 3D 4C"))
    assert len(t.unseen()) == 49


def test_set_hand_diff_marks_played() -> None:
    t = DeckTracker()
    t.set_hand(H("3S 3D 4C 4H"))
    # Next frame: we played 3S 3D — hand is now 4C 4H.
    t.set_hand(H("4C 4H"))
    assert sorted(t.hand()) == sorted(H("4C 4H"))
    played = sorted(t.played())
    assert played == sorted(H("3S 3D"))


def test_mark_played_separately() -> None:
    t = DeckTracker()
    t.mark_played(H("5S 5D"))
    assert sorted(t.played()) == sorted(H("5S 5D"))


def test_unseen_is_neither_hand_nor_played() -> None:
    t = DeckTracker()
    t.set_hand(H("3S"))
    t.mark_played(H("4H"))
    assert all(t.status(c) == CardStatus.UNKNOWN for c in H("5S 5D 5C 5H"))


def test_reset() -> None:
    t = DeckTracker()
    t.set_hand(H("3S 3D"))
    t.mark_played(H("4H"))
    t.reset()
    assert len(t.unseen()) == 52
    assert t.hand() == []
    assert t.played() == []
