from __future__ import annotations

from agenttienlen.agent.heuristic import HeuristicPolicy
from agenttienlen.agent.policy import Pass, Play
from agenttienlen.core.card import Card, Rank, Suit
from agenttienlen.core.combo import ComboType
from agenttienlen.core.rules import classify
from agenttienlen.memory.game_state import GameState


def H(s: str) -> list[Card]:
    return [Card.parse(t) for t in s.split()]


def make_state(
    hand: str,
    current: str | None = None,
    *,
    first_round: bool = False,
) -> GameState:
    state = GameState()
    state.hand = sorted(H(hand))
    if current is not None:
        combo = classify(H(current))
        assert combo is not None
        state.current_combo = combo
    state.is_first_round = first_round
    return state


# ---- Leading ----


def test_first_round_must_include_three_of_spades() -> None:
    policy = HeuristicPolicy()
    state = make_state("3S 5D 7H KC", first_round=True)
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert Card(Rank.THREE, Suit.SPADES) in action.combo.cards


def test_first_round_prefers_smallest_single_with_three_spades() -> None:
    policy = HeuristicPolicy()
    state = make_state("3S 5D 7H KC", first_round=True)
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert action.combo.type == ComboType.SINGLE
    assert action.combo.cards[0] == Card(Rank.THREE, Suit.SPADES)


def test_lead_prefers_low_single_over_pair() -> None:
    policy = HeuristicPolicy()
    state = make_state("4S 7C 7D")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert action.combo.type == ComboType.SINGLE
    assert action.combo.cards[0].rank == Rank.FOUR


def test_lead_never_bombs_when_smaller_options_exist() -> None:
    policy = HeuristicPolicy()
    # Hand contains a tứ quý but plenty of singles too — never lead with bomb.
    state = make_state("3D 5C 7H 9S 9C 9D 9H")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert not action.combo.is_bomb


# ---- Responding ----


def test_respond_with_smallest_beat() -> None:
    policy = HeuristicPolicy()
    state = make_state("4S 5D 6C 7H 9S", current="3H")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert action.combo.type == ComboType.SINGLE
    assert action.combo.cards[0].rank == Rank.FOUR


def test_pass_when_no_legal_response() -> None:
    policy = HeuristicPolicy()
    state = make_state("3S 4D 5C", current="KH")
    action = policy.decide(state)
    assert isinstance(action, Pass)


def test_does_not_waste_bomb_when_non_bomb_beat_exists() -> None:
    policy = HeuristicPolicy()
    # Hand has both a tứ quý 9 and a single 2♥ that beats the prev single 2♠.
    # Bot should pick the cheaper non-bomb response.
    state = make_state("9S 9C 9D 9H 2H", current="2S")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert not action.combo.is_bomb
    assert action.combo.type == ComboType.SINGLE


def test_chops_single_two_with_bomb() -> None:
    policy = HeuristicPolicy()
    state = make_state("9S 9C 9D 9H", current="2H")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert action.combo.type == ComboType.FOUR_OF_A_KIND


def test_chops_pair_twos_with_four_of_a_kind() -> None:
    policy = HeuristicPolicy()
    state = make_state("KS KC KD KH", current="2S 2H")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert action.combo.type == ComboType.FOUR_OF_A_KIND


def test_pair_responds_to_pair() -> None:
    policy = HeuristicPolicy()
    state = make_state("7S 7C 9D 9H KS", current="5S 5D")
    action = policy.decide(state)
    assert isinstance(action, Play)
    assert action.combo.type == ComboType.PAIR
    assert action.combo.key_card.rank == Rank.SEVEN
