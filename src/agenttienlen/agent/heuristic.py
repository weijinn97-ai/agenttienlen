"""Heuristic policy: "chuẩn sách giáo khoa" defaults.

Guidelines this policy follows (in priority order):

1. When leading the very first trick, the play **must include 3♠**.
2. When leading any other trick, prefer to dump low *singles* first to clear
   the hand, but never break pairs/triples/straights if a free single is
   available.
3. When responding, play the **smallest combo that beats** the current one.
4. Never waste a bomb (tứ quý, 3 đôi thông, 4 đôi thông) to chop a non-2
   single / non-2 pair — save bombs for chopping 2s or for own bombs.
5. Save 2s for the end of the round; only lead a 2 when nothing smaller is
   available.

The policy is intentionally simple, deterministic, and easy to test. It is the
baseline; future iterations can layer search / RL on top via the same
:class:`Policy` Protocol.
"""

from __future__ import annotations

from agenttienlen.agent.policy import Action, Pass, Play
from agenttienlen.core.card import Card, Rank, Suit
from agenttienlen.core.combo import Combo, ComboType
from agenttienlen.core.enumerate import enumerate_moves
from agenttienlen.memory.game_state import GameState

THREE_OF_SPADES = Card(Rank.THREE, Suit.SPADES)


class HeuristicPolicy:
    """A small, well-tested default policy. See module docstring for rules."""

    def decide(self, state: GameState) -> Action:
        if state.is_leading():
            return self._lead(state)
        return self._respond(state)

    # ---- leading ----

    def _lead(self, state: GameState) -> Action:
        if not state.hand:
            return Pass(reason="empty hand")

        all_moves = enumerate_moves(state.hand, prev=None, include_bombs=False)

        if state.is_first_round and THREE_OF_SPADES in state.hand:
            three_spades_moves = [m for m in all_moves if THREE_OF_SPADES in m.cards]
            if three_spades_moves:
                return Play(self._pick_smallest_lead(three_spades_moves))
            # Fallback: lead the lone 3♠.
            return Play(Combo(ComboType.SINGLE, (THREE_OF_SPADES,)))

        if not all_moves:
            # Hand contains only bombs and 2s. Lead the smallest single 2.
            twos = [c for c in state.hand if c.rank == Rank.TWO]
            if twos:
                return Play(Combo(ComboType.SINGLE, (min(twos),)))
            # As a last resort, lead a bomb.
            with_bombs = enumerate_moves(state.hand, prev=None, include_bombs=True)
            return Play(min(with_bombs, key=_lead_priority))

        return Play(self._pick_smallest_lead(all_moves))

    @staticmethod
    def _pick_smallest_lead(moves: list[Combo]) -> Combo:
        return min(moves, key=_lead_priority)

    # ---- responding ----

    def _respond(self, state: GameState) -> Action:
        prev = state.current_combo
        assert prev is not None  # leading is handled above
        legal = enumerate_moves(state.hand, prev=prev)
        if not legal:
            return Pass(reason="no legal beat")

        non_bombs = [m for m in legal if not m.is_bomb]
        if non_bombs:
            return Play(min(non_bombs, key=lambda m: m.key_card.strength))

        # Only bomb options remain.
        if _is_chopping_two(prev):
            # Chopping a 2 is always worth it.
            return Play(min(legal, key=_bomb_priority))
        return Pass(reason="hold bomb")


def _lead_priority(combo: Combo) -> tuple[int, int, int]:
    """Sort key when leading: prefer non-bomb singles, then by key card."""
    is_bomb = 1 if combo.is_bomb else 0
    is_two = 1 if combo.cards[0].rank == Rank.TWO else 0
    # Singles first (lower size = lower priority value).
    type_priority = _LEAD_TYPE_ORDER[combo.type]
    return (is_bomb, is_two, type_priority * 100 + combo.key_card.strength)


def _bomb_priority(combo: Combo) -> tuple[int, int]:
    """Sort key when forced to bomb: prefer cheaper bomb, by type then rank."""
    return (_BOMB_TYPE_ORDER[combo.type], combo.key_card.strength)


def _is_chopping_two(prev: Combo) -> bool:
    return prev.type in (ComboType.SINGLE, ComboType.PAIR) and prev.cards[0].rank == Rank.TWO


_LEAD_TYPE_ORDER: dict[ComboType, int] = {
    ComboType.SINGLE: 0,
    ComboType.PAIR: 1,
    ComboType.STRAIGHT: 2,
    ComboType.TRIPLE: 3,
    ComboType.FOUR_OF_A_KIND: 9,
    ComboType.THREE_PAIRS: 9,
    ComboType.FOUR_PAIRS: 9,
}

# Prefer 3 đôi thông over tứ quý over 4 đôi thông when forced (3 đôi thông has
# more redundancy; tứ quý is a tight rank lock; 4 đôi thông is the strongest
# and most precious resource).
_BOMB_TYPE_ORDER: dict[ComboType, int] = {
    ComboType.THREE_PAIRS: 0,
    ComboType.FOUR_OF_A_KIND: 1,
    ComboType.FOUR_PAIRS: 2,
    # Non-bombs should never reach this map, but make it total to satisfy mypy.
    ComboType.SINGLE: 100,
    ComboType.PAIR: 100,
    ComboType.TRIPLE: 100,
    ComboType.STRAIGHT: 100,
}
