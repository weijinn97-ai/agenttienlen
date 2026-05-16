"""Enumerate every legal move available from a hand.

Used by the heuristic agent (and any future search). Given a hand of cards and
an optional ``prev`` combo to beat, return all combos that are legal to play.
If ``prev`` is None, all legal combos (any type) are returned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, pairwise

from agenttienlen.core.card import Card, Rank
from agenttienlen.core.combo import Combo, ComboType
from agenttienlen.core.rules import beats, classify


def _group_by_rank(hand: Sequence[Card]) -> dict[Rank, list[Card]]:
    grouped: dict[Rank, list[Card]] = defaultdict(list)
    for c in hand:
        grouped[c.rank].append(c)
    for rank in grouped:
        grouped[rank].sort()
    return grouped


def enumerate_singles(hand: Sequence[Card]) -> list[Combo]:
    return [Combo(ComboType.SINGLE, (c,)) for c in sorted(hand)]


def enumerate_pairs(hand: Sequence[Card]) -> list[Combo]:
    out: list[Combo] = []
    for cards in _group_by_rank(hand).values():
        if len(cards) >= 2:
            for pair in combinations(cards, 2):
                out.append(Combo(ComboType.PAIR, tuple(sorted(pair))))
    return out


def enumerate_triples(hand: Sequence[Card]) -> list[Combo]:
    out: list[Combo] = []
    for cards in _group_by_rank(hand).values():
        if len(cards) >= 3:
            for trio in combinations(cards, 3):
                out.append(Combo(ComboType.TRIPLE, tuple(sorted(trio))))
    return out


def enumerate_fours(hand: Sequence[Card]) -> list[Combo]:
    out: list[Combo] = []
    for cards in _group_by_rank(hand).values():
        if len(cards) == 4:
            out.append(Combo(ComboType.FOUR_OF_A_KIND, tuple(sorted(cards))))
    return out


def enumerate_straights(hand: Sequence[Card], min_len: int = 3) -> list[Combo]:
    """All straights of length ≥ ``min_len``, no 2s, no duplicates."""
    grouped = _group_by_rank(hand)
    valid_ranks = sorted(r for r in grouped if r != Rank.TWO)
    if len(valid_ranks) < min_len:
        return []

    # Partition into maximal consecutive runs so we don't emit duplicates.
    runs: list[list[Rank]] = [[valid_ranks[0]]]
    for prev, curr in pairwise(valid_ranks):
        if int(curr) - int(prev) == 1:
            runs[-1].append(curr)
        else:
            runs.append([curr])

    out: list[Combo] = []
    for run in runs:
        if len(run) < min_len:
            continue
        for length in range(min_len, len(run) + 1):
            for start in range(len(run) - length + 1):
                slice_ranks = run[start : start + length]
                choices = [grouped[r] for r in slice_ranks]
                out.extend(
                    Combo(ComboType.STRAIGHT, tuple(sorted(combo)))
                    for combo in _cartesian_pick(choices)
                )
    return out


def enumerate_consecutive_pairs(hand: Sequence[Card], count: int) -> list[Combo]:
    """All ``count`` consecutive pairs (3 đôi thông or 4 đôi thông), no 2s."""
    if count < 3:
        return []
    grouped = _group_by_rank(hand)
    pair_capable = sorted(r for r in grouped if r != Rank.TWO and len(grouped[r]) >= 2)
    out: list[Combo] = []
    n = len(pair_capable)
    combo_type = ComboType.THREE_PAIRS if count == 3 else ComboType.FOUR_PAIRS
    for i in range(n - count + 1):
        window = pair_capable[i : i + count]
        if not _is_consecutive(window):
            continue
        # Pick any 2 of each rank as a pair.
        per_rank: list[list[tuple[Card, ...]]] = [list(combinations(grouped[r], 2)) for r in window]
        for choice in _cartesian_pick(per_rank):
            cards: list[Card] = []
            for pair in choice:
                cards.extend(pair)
            out.append(Combo(combo_type, tuple(sorted(cards))))
    return out


def _is_consecutive(ranks: Sequence[Rank]) -> bool:
    return all(int(curr) - int(prev) == 1 for prev, curr in pairwise(ranks))


def _cartesian_pick(groups: Sequence[Sequence[object]]) -> list[tuple[object, ...]]:
    """Cross product of groups, lazily realized into a list."""
    out: list[tuple[object, ...]] = [()]
    for group in groups:
        out = [(*prefix, item) for prefix in out for item in group]
    return out


def enumerate_moves(
    hand: Sequence[Card],
    prev: Combo | None = None,
    *,
    include_bombs: bool = True,
) -> list[Combo]:
    """Return every Combo legal from `hand`, optionally restricted to ones that beat `prev`.

    When ``prev`` is None, returns *every* legal combo in the hand.
    When ``prev`` is provided, returns only combos that beat it (same-type or chop).

    ``include_bombs`` (default True) controls whether tứ quý / 3 đôi thông /
    4 đôi thông are included when ``prev`` is None. Set False to filter out
    blind bombing when leading a new round.
    """
    all_combos: list[Combo] = []
    all_combos.extend(enumerate_singles(hand))
    all_combos.extend(enumerate_pairs(hand))
    all_combos.extend(enumerate_triples(hand))
    all_combos.extend(enumerate_straights(hand))
    if include_bombs or prev is not None:
        all_combos.extend(enumerate_fours(hand))
        all_combos.extend(enumerate_consecutive_pairs(hand, count=3))
        all_combos.extend(enumerate_consecutive_pairs(hand, count=4))

    if prev is None:
        return all_combos

    # Sanity: ensure each combo classifies cleanly, then filter by beats().
    legal: list[Combo] = []
    for combo in all_combos:
        if classify(combo.cards) is None:
            continue
        if beats(combo, prev):
            legal.append(combo)
    return legal
