"""Tien Len Mien Nam rules: combo classification, beats(), can_chop().

This file is the *single source of truth* for the rule engine. The agent and
test suite both consume it. Encoded variant: **Nhất Ăn Tất** (most common SGK
interpretation):

Beats (same type):
  - Single | Pair | Triple: compare highest card (rank then suit).
  - Straight: same length only; compare highest card.
  - Tứ quý (four of a kind): compare rank.
  - 3 đôi thông / 4 đôi thông: same length; compare highest pair.

Chop (chặt — cross-type overrides):
  - Lẻ 2  → tứ quý, 3 đôi thông, 4 đôi thông.
  - Đôi 2 → tứ quý, 4 đôi thông.
  - Tứ quý → tứ quý lớn hơn (same-type), 4 đôi thông.
  - 3 đôi thông → 3 đôi thông lớn hơn (same-type), tứ quý, 4 đôi thông.
  - 4 đôi thông → 4 đôi thông lớn hơn (same-type only).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from agenttienlen.core.card import Card, Rank
from agenttienlen.core.combo import Combo, ComboType


def classify(cards: Sequence[Card]) -> Combo | None:
    """Return the Combo formed by `cards`, or None if not a legal play.

    Caller may pass cards in any order.
    """
    if not cards:
        return None
    sorted_cards = tuple(sorted(cards))
    n = len(sorted_cards)

    if n == 1:
        return Combo(ComboType.SINGLE, sorted_cards)

    # All same rank → pair / triple / four-of-a-kind.
    if all(c.rank == sorted_cards[0].rank for c in sorted_cards):
        if n == 2:
            return Combo(ComboType.PAIR, sorted_cards)
        if n == 3:
            return Combo(ComboType.TRIPLE, sorted_cards)
        if n == 4:
            return Combo(ComboType.FOUR_OF_A_KIND, sorted_cards)
        return None

    # Straights/consecutive pairs cannot contain 2.
    if any(c.rank == Rank.TWO for c in sorted_cards):
        return None

    if n >= 3 and _is_straight(sorted_cards):
        return Combo(ComboType.STRAIGHT, sorted_cards)

    if n == 6 and _is_consecutive_pairs(sorted_cards, count=3):
        return Combo(ComboType.THREE_PAIRS, sorted_cards)

    if n == 8 and _is_consecutive_pairs(sorted_cards, count=4):
        return Combo(ComboType.FOUR_PAIRS, sorted_cards)

    return None


def beats(new: Combo, prev: Combo) -> bool:
    """Return True if `new` legally beats `prev`."""
    if new.type == prev.type and new.size == prev.size:
        if new.type == ComboType.FOUR_OF_A_KIND:
            return new.key_card.rank > prev.key_card.rank
        return new.key_card > prev.key_card
    return can_chop(new, prev)


def can_chop(new: Combo, prev: Combo) -> bool:
    """Cross-type chop (chặt) override. Same-type comparisons go via :func:`beats`."""
    if new.type == prev.type:
        return False

    if _is_single_two(prev):
        return new.type in (
            ComboType.FOUR_OF_A_KIND,
            ComboType.THREE_PAIRS,
            ComboType.FOUR_PAIRS,
        )
    if _is_pair_twos(prev):
        return new.type in (ComboType.FOUR_OF_A_KIND, ComboType.FOUR_PAIRS)
    if prev.type == ComboType.FOUR_OF_A_KIND:
        return new.type == ComboType.FOUR_PAIRS
    if prev.type == ComboType.THREE_PAIRS:
        return new.type in (ComboType.FOUR_OF_A_KIND, ComboType.FOUR_PAIRS)
    return False


def _is_single_two(combo: Combo) -> bool:
    return combo.type == ComboType.SINGLE and combo.cards[0].rank == Rank.TWO


def _is_pair_twos(combo: Combo) -> bool:
    return combo.type == ComboType.PAIR and combo.cards[0].rank == Rank.TWO


def _is_straight(sorted_cards: Sequence[Card]) -> bool:
    """Sorted cards form ≥3 consecutive ranks with no duplicates."""
    ranks = [int(c.rank) for c in sorted_cards]
    if len(set(ranks)) != len(ranks):
        return False
    return all(curr_rank - prev_rank == 1 for prev_rank, curr_rank in itertools.pairwise(ranks))


def _is_consecutive_pairs(sorted_cards: Sequence[Card], count: int) -> bool:
    """Sorted cards split into `count` consecutive pairs of identical rank."""
    if len(sorted_cards) != 2 * count:
        return False
    pair_ranks: list[int] = []
    for i in range(count):
        a, b = sorted_cards[2 * i], sorted_cards[2 * i + 1]
        if a.rank != b.rank:
            return False
        pair_ranks.append(int(a.rank))
    for prev_rank, curr_rank in itertools.pairwise(pair_ranks):
        if curr_rank - prev_rank != 1:
            return False
    return True
