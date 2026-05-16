"""beats() and can_chop() — covers same-type and cross-type rules."""

from __future__ import annotations

from agenttienlen.core.card import Card
from agenttienlen.core.rules import beats, can_chop, classify


def C(s: str) -> object:
    combo = classify([Card.parse(t) for t in s.split()])
    assert combo is not None, f"Could not classify: {s}"
    return combo


# ---- Same-type beats ----


def test_single_higher_rank_beats_lower() -> None:
    assert beats(C("KH"), C("KS"))  # type: ignore[arg-type]
    assert beats(C("4S"), C("3H"))  # type: ignore[arg-type]
    assert not beats(C("3S"), C("3H"))  # type: ignore[arg-type]


def test_pair_compares_highest_card() -> None:
    assert beats(C("7D 7H"), C("7S 7C"))  # type: ignore[arg-type]
    assert not beats(C("7S 7C"), C("7D 7H"))  # type: ignore[arg-type]


def test_triple_beats_by_rank() -> None:
    assert beats(C("4S 4C 4D"), C("3S 3C 3D"))  # type: ignore[arg-type]
    # Same rank triples don't exist with distinct suits but if ranks tied
    # they cannot legally co-occur in a 4-suit deck (only 3 of same rank
    # at most after a 4th is impossible without four_of_a_kind).


def test_straight_same_length_beats() -> None:
    assert beats(C("5S 6D 7C"), C("3S 4D 5C"))  # type: ignore[arg-type]


def test_straight_different_length_does_not_beat() -> None:
    # 5-card straight does NOT beat 3-card straight (must be same length).
    assert not beats(C("3S 4D 5C 6H 7C"), C("3S 4D 5C"))  # type: ignore[arg-type]


def test_four_of_a_kind_higher_rank_beats() -> None:
    assert beats(C("4S 4C 4D 4H"), C("3S 3C 3D 3H"))  # type: ignore[arg-type]


# ---- Chops (cross-type) ----


def test_four_of_a_kind_chops_single_two() -> None:
    assert can_chop(C("3S 3C 3D 3H"), C("2H"))  # type: ignore[arg-type]
    assert beats(C("3S 3C 3D 3H"), C("2H"))  # type: ignore[arg-type]


def test_three_pairs_chops_single_two() -> None:
    assert beats(C("3S 3C 4D 4H 5C 5S"), C("2H"))  # type: ignore[arg-type]


def test_three_pairs_does_not_chop_pair_twos() -> None:
    # Standard MN rule: 3 đôi thông KHÔNG chặt được đôi 2.
    assert not beats(C("3S 3C 4D 4H 5C 5S"), C("2S 2H"))  # type: ignore[arg-type]


def test_four_of_a_kind_chops_pair_twos() -> None:
    assert beats(C("3S 3C 3D 3H"), C("2S 2H"))  # type: ignore[arg-type]


def test_four_pairs_chops_pair_twos() -> None:
    assert beats(C("3S 3C 4D 4H 5C 5S 6C 6H"), C("2S 2H"))  # type: ignore[arg-type]


def test_four_pairs_chops_four_of_a_kind() -> None:
    assert beats(C("3S 3C 4D 4H 5C 5S 6C 6H"), C("AS AC AD AH"))  # type: ignore[arg-type]


def test_four_pairs_chops_three_pairs() -> None:
    # 4 đôi thông chặt 3 đôi thông bất kỳ (rule: 4 đôi thông is the strongest).
    assert beats(
        C("3S 3C 4D 4H 5C 5S 6C 6H"),
        C("QS QC KD KH AC AS"),
    )  # type: ignore[arg-type]


def test_lower_three_pairs_does_not_beat_higher() -> None:
    a = C("3S 3C 4D 4H 5C 5S")
    b = C("4S 4D 5C 5H 6C 6S")
    assert not beats(a, b)  # type: ignore[arg-type]
    assert beats(b, a)  # type: ignore[arg-type]


def test_pair_doesnt_chop_pair_with_lower() -> None:
    assert not beats(C("3S 3C"), C("KS KC"))  # type: ignore[arg-type]


def test_single_2_not_chopped_by_pair_or_straight() -> None:
    assert not beats(C("3S 3D"), C("2H"))  # type: ignore[arg-type]
    assert not beats(C("3S 4S 5S"), C("2H"))  # type: ignore[arg-type]
