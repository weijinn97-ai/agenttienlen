"""Tests for template_utils: name parsing and template loading."""

from __future__ import annotations

import pytest

from agenttienlen.core.card import Card, Rank, Suit
from agenttienlen.vision.template_utils import (
    card_to_template_name,
    parse_template_name,
)


class TestParseTemplateName:
    """Verify Vietnamese card-name → Card mapping."""

    @pytest.mark.parametrize(
        ("name", "rank", "suit"),
        [
            ("3Bich", Rank.THREE, Suit.SPADES),
            ("3Chuon", Rank.THREE, Suit.CLUBS),
            ("3Co", Rank.THREE, Suit.HEARTS),
            ("3Ro", Rank.THREE, Suit.DIAMONDS),
            ("10Bich", Rank.TEN, Suit.SPADES),
            ("10Co", Rank.TEN, Suit.HEARTS),
            ("JBich", Rank.JACK, Suit.SPADES),
            ("QRo", Rank.QUEEN, Suit.DIAMONDS),
            ("KChuon", Rank.KING, Suit.CLUBS),
            ("AtBich", Rank.ACE, Suit.SPADES),
            ("AtRo", Rank.ACE, Suit.DIAMONDS),
            ("2Co", Rank.TWO, Suit.HEARTS),
        ],
    )
    def test_parse_valid(self, name: str, rank: Rank, suit: Suit) -> None:
        card = parse_template_name(name)
        assert card is not None
        assert card.rank == rank
        assert card.suit == suit

    @pytest.mark.parametrize("name", ["", "XBich", "3X", "hello", "11Bich", "1Bich"])
    def test_parse_invalid_returns_none(self, name: str) -> None:
        assert parse_template_name(name) is None


class TestCardToTemplateName:
    """Verify Card → Vietnamese template-name roundtrip."""

    @pytest.mark.parametrize(
        ("card_str", "expected"),
        [
            ("3S", "3Bich"),
            ("10H", "10Co"),
            ("JD", "JRo"),
            ("AS", "AtBich"),
            ("2C", "2Chuon"),
            ("KC", "KChuon"),
        ],
    )
    def test_roundtrip(self, card_str: str, expected: str) -> None:
        card = Card.parse(card_str)
        name = card_to_template_name(card)
        assert name == expected
        assert parse_template_name(name) == card
