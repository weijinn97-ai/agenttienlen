"""Card, Rank, Suit primitives for Tien Len Mien Nam.

Rank order (low→high): 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2
Suit order (low→high): Spades ♠, Clubs ♣, Diamonds ♦, Hearts ♥

Card strength = rank.value * 4 + suit.value, giving a total order on 52 cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    SPADES = 0
    CLUBS = 1
    DIAMONDS = 2
    HEARTS = 3

    @property
    def symbol(self) -> str:
        return _SUIT_SYMBOL[self]

    @property
    def letter(self) -> str:
        return _SUIT_LETTER[self]

    @classmethod
    def from_letter(cls, letter: str) -> Suit:
        try:
            return _LETTER_TO_SUIT[letter.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown suit letter: {letter!r}") from exc


_SUIT_SYMBOL: dict[Suit, str] = {
    Suit.SPADES: "♠",
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
}
_SUIT_LETTER: dict[Suit, str] = {
    Suit.SPADES: "S",
    Suit.CLUBS: "C",
    Suit.DIAMONDS: "D",
    Suit.HEARTS: "H",
}
_LETTER_TO_SUIT: dict[str, Suit] = {v: k for k, v in _SUIT_LETTER.items()}


class Rank(IntEnum):
    THREE = 0
    FOUR = 1
    FIVE = 2
    SIX = 3
    SEVEN = 4
    EIGHT = 5
    NINE = 6
    TEN = 7
    JACK = 8
    QUEEN = 9
    KING = 10
    ACE = 11
    TWO = 12

    @property
    def label(self) -> str:
        return _RANK_LABEL[self]

    @classmethod
    def from_label(cls, label: str) -> Rank:
        try:
            return _LABEL_TO_RANK[label.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown rank label: {label!r}") from exc


_RANK_LABEL: dict[Rank, str] = {
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.TWO: "2",
}
_LABEL_TO_RANK: dict[str, Rank] = {v: k for k, v in _RANK_LABEL.items()}


@dataclass(frozen=True, order=True, slots=True)
class Card:
    """A single playing card. Ordered by strength (rank, then suit)."""

    rank: Rank
    suit: Suit

    @property
    def strength(self) -> int:
        """Total order over 52 cards. 3♠ = 0, 2♥ = 51."""
        return int(self.rank) * 4 + int(self.suit)

    def __str__(self) -> str:
        return f"{self.rank.label}{self.suit.symbol}"

    def __repr__(self) -> str:
        return f"Card({self.rank.label}{self.suit.letter})"

    @classmethod
    def parse(cls, text: str) -> Card:
        """Parse strings like '3S', '10H', 'JD', '2C'."""
        s = text.strip().upper()
        if len(s) < 2:
            raise ValueError(f"Invalid card text: {text!r}")
        rank_part, suit_part = s[:-1], s[-1]
        return cls(rank=Rank.from_label(rank_part), suit=Suit.from_letter(suit_part))

    @classmethod
    def from_strength(cls, strength: int) -> Card:
        if not 0 <= strength < 52:
            raise ValueError(f"strength out of range: {strength}")
        return cls(rank=Rank(strength // 4), suit=Suit(strength % 4))


def full_deck() -> list[Card]:
    """Return all 52 cards in strength order (3♠ → 2♥)."""
    return [Card.from_strength(i) for i in range(52)]


def parse_hand(text: str) -> list[Card]:
    """Parse a whitespace- or comma-separated list of card tokens."""
    tokens = [t for t in text.replace(",", " ").split() if t]
    return [Card.parse(t) for t in tokens]
