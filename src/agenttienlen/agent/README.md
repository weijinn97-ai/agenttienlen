# `agent` — decision engine

The agent decides whether to **Play** a `Combo` or **Pass**, given a
`GameState`. Implementations satisfy the :class:`Policy` Protocol:

```python
class Policy(Protocol):
    def decide(self, state: GameState) -> Action: ...
```

`Action = Play | Pass`.

## Current policy: `HeuristicPolicy` (chuẩn SGK)

Strategy (priority order):

1. **First trick** → must include 3♠.
2. **Leading another trick** → dump low singles first; never break a pair/triple/straight if a free single is available; save 2s for the end.
3. **Responding** → smallest combo that beats the current play.
4. **Don't waste bombs** — only chop with a bomb when responding to a 2 (lẻ 2 / đôi 2) or facing a same-type bomb.
5. **Save 2s** — only lead a 2 when forced.

Determined / unit-tested. Drop-in for future search / RL / NN policies via the
same Protocol.

## Adding a new policy

1. Create `src/agenttienlen/agent/<my_policy>.py`.
2. Implement `decide(state) -> Action`.
3. Register in `agent/__init__.py` if you want it discoverable.
4. Add tests under `tests/agent/test_<my_policy>.py`.

## Notes / future work

- Currently the policy has no opponent model; `state.deck` is available
  (`DeckTracker`) so future versions can reason about unseen cards.
- Pure function: no mutation of `state`, no side effects.
