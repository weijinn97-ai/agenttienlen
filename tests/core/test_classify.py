from __future__ import annotations

from agenttienlen.core.card import Card
from agenttienlen.core.combo import ComboType
from agenttienlen.core.rules import classify


def H(s: str) -> list[Card]:
    return [Card.parse(t) for t in s.split()]


def test_single() -> None:
    c = classify(H("3S"))
    assert c is not None
    assert c.type == ComboType.SINGLE


def test_pair() -> None:
    c = classify(H("7C 7D"))
    assert c is not None
    assert c.type == ComboType.PAIR


def test_triple() -> None:
    c = classify(H("9S 9C 9D"))
    assert c is not None
    assert c.type == ComboType.TRIPLE


def test_four_of_a_kind() -> None:
    c = classify(H("5S 5C 5D 5H"))
    assert c is not None
    assert c.type == ComboType.FOUR_OF_A_KIND
    assert c.is_bomb


def test_straight_3() -> None:
    c = classify(H("4D 5C 6H"))
    assert c is not None
    assert c.type == ComboType.STRAIGHT
    assert c.size == 3


def test_straight_long() -> None:
    c = classify(H("3S 4D 5C 6H 7S 8C 9D 10H JC QD KS AH"))
    assert c is not None
    assert c.type == ComboType.STRAIGHT
    assert c.size == 12


def test_straight_cannot_include_2() -> None:
    assert classify(H("AS 2H 3D")) is None
    assert classify(H("KS AS 2H")) is None


def test_three_pairs_consecutive() -> None:
    c = classify(H("3S 3D 4C 4H 5D 5S"))
    assert c is not None
    assert c.type == ComboType.THREE_PAIRS
    assert c.is_bomb


def test_three_pairs_not_consecutive() -> None:
    # 3-3 5-5 6-6: gap → not a 3 đôi thông; also not any other combo
    assert classify(H("3S 3D 5C 5H 6D 6S")) is None


def test_three_pairs_cannot_include_2() -> None:
    assert classify(H("KC KH AS AH 2C 2H")) is None


def test_four_pairs_consecutive() -> None:
    c = classify(H("3S 3D 4C 4H 5D 5S 6C 6H"))
    assert c is not None
    assert c.type == ComboType.FOUR_PAIRS


def test_invalid_garbage_returns_none() -> None:
    assert classify(H("3S 7H")) is None  # two unrelated singles
    assert classify(H("3S 4S 5S 7S")) is None  # broken straight (skipped 6)


def test_empty_returns_none() -> None:
    assert classify([]) is None


def test_unsorted_input_classifies_correctly() -> None:
    c = classify(H("6H 4D 5C"))
    assert c is not None
    assert c.type == ComboType.STRAIGHT
    assert [str(card) for card in c.cards] == ["4♦", "5♣", "6♥"]
