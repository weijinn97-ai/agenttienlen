from __future__ import annotations

import pytest

from agenttienlen.core.card import Card, Rank, Suit, full_deck, parse_hand


def test_card_strength_full_range() -> None:
    deck = full_deck()
    assert len(deck) == 52
    assert {c.strength for c in deck} == set(range(52))


def test_card_ordering() -> None:
    assert Card(Rank.THREE, Suit.SPADES) < Card(Rank.THREE, Suit.HEARTS)
    assert Card(Rank.TWO, Suit.SPADES) > Card(Rank.ACE, Suit.HEARTS)
    assert Card(Rank.TWO, Suit.HEARTS) == Card.from_strength(51)


def test_card_parse_roundtrip() -> None:
    for c in full_deck():
        assert Card.parse(f"{c.rank.label}{c.suit.letter}") == c


def test_card_parse_examples() -> None:
    assert Card.parse("3S") == Card(Rank.THREE, Suit.SPADES)
    assert Card.parse("10H") == Card(Rank.TEN, Suit.HEARTS)
    assert Card.parse("JD") == Card(Rank.JACK, Suit.DIAMONDS)
    assert Card.parse("2C") == Card(Rank.TWO, Suit.CLUBS)


def test_card_str_uses_symbol() -> None:
    assert str(Card(Rank.ACE, Suit.HEARTS)) == "A♥"


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        Card.parse("XX")
    with pytest.raises(ValueError):
        Card.parse("3")
    with pytest.raises(ValueError):
        Card.from_strength(-1)
    with pytest.raises(ValueError):
        Card.from_strength(52)


def test_parse_hand_whitespace_and_commas() -> None:
    hand = parse_hand("3S 3H, JS, 2H")
    assert hand == [
        Card(Rank.THREE, Suit.SPADES),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.TWO, Suit.HEARTS),
    ]
