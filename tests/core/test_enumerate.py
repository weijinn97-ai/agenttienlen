from __future__ import annotations

from agenttienlen.core.card import Card
from agenttienlen.core.combo import ComboType
from agenttienlen.core.enumerate import enumerate_moves
from agenttienlen.core.rules import classify


def H(s: str) -> list[Card]:
    return [Card.parse(t) for t in s.split()]


def test_enumerate_includes_singles() -> None:
    hand = H("3S 4D 5C")
    moves = enumerate_moves(hand)
    singles = [m for m in moves if m.type == ComboType.SINGLE]
    assert len(singles) == 3
    assert {m.cards[0] for m in singles} == set(hand)


def test_enumerate_pairs() -> None:
    hand = H("3S 3D 4C 4H")
    moves = enumerate_moves(hand)
    pairs = [m for m in moves if m.type == ComboType.PAIR]
    assert len(pairs) == 2  # 3♠3♦ and 4♣4♥


def test_enumerate_straights_min_length_3() -> None:
    hand = H("3S 4D 5C 6H")
    moves = enumerate_moves(hand)
    straights = [m for m in moves if m.type == ComboType.STRAIGHT]
    sizes = sorted(m.size for m in straights)
    assert sizes == [3, 3, 4]


def test_enumerate_excludes_straights_with_two() -> None:
    hand = H("AD 2H 3S 4D")
    moves = enumerate_moves(hand)
    straights = [m for m in moves if m.type == ComboType.STRAIGHT]
    for s in straights:
        for c in s.cards:
            assert c.rank.label != "2"


def test_enumerate_four_of_a_kind() -> None:
    hand = H("5S 5C 5D 5H 7S")
    moves = enumerate_moves(hand)
    fours = [m for m in moves if m.type == ComboType.FOUR_OF_A_KIND]
    assert len(fours) == 1


def test_enumerate_three_pairs() -> None:
    hand = H("3S 3D 4C 4H 5D 5S")
    moves = enumerate_moves(hand)
    tp = [m for m in moves if m.type == ComboType.THREE_PAIRS]
    assert len(tp) == 1


def test_enumerate_four_pairs() -> None:
    hand = H("3S 3D 4C 4H 5D 5S 6C 6H")
    moves = enumerate_moves(hand)
    fp = [m for m in moves if m.type == ComboType.FOUR_PAIRS]
    assert len(fp) == 1


def test_enumerate_against_prev_filters_correctly() -> None:
    hand = H("5S 5D 6C 6H 7D 7S 2H")
    prev = classify(H("4S 4H"))
    assert prev is not None
    moves = enumerate_moves(hand, prev)
    # Only pairs higher than 4-4 (= 5-5, 6-6, 7-7) and chops.
    assert all(m.type == ComboType.PAIR for m in moves)
    assert {m.key_card.rank.label for m in moves} == {"5", "6", "7"}


def test_enumerate_returns_chops_against_single_two() -> None:
    hand = H("3S 3D 4C 4H 5C 5S 6C 6H 9S 9C 9D 9H")
    prev = classify(H("2H"))
    assert prev is not None
    moves = enumerate_moves(hand, prev)
    # 9999 (tứ quý), 3-3-4-4-5-5 (3 đôi thông), and 3-3-4-4-5-5-6-6 (4 đôi thông).
    types = {m.type for m in moves}
    assert ComboType.FOUR_OF_A_KIND in types
    assert ComboType.THREE_PAIRS in types
    assert ComboType.FOUR_PAIRS in types
