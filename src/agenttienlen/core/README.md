# `core` — pure game logic

No I/O, no vision, no agent code. Just the rule engine.

## Public API

```python
from agenttienlen.core import (
    Card, Rank, Suit,
    Combo, ComboType,
    classify, beats, can_chop,
    enumerate_moves,
)
```

### `Card`
- `Card(rank, suit)` — `rank: Rank`, `suit: Suit`.
- `Card.parse("3S")`, `Card.parse("10H")`, `Card.parse("JD")`, `Card.parse("2C")`.
- `card.strength` — integer 0–51. `3♠` = 0, `2♥` = 51.
- Sortable via `<`/`>`.

### `Combo`
- Built via `classify(cards)` → `Combo | None`.
- `combo.type: ComboType` — one of `SINGLE`, `PAIR`, `TRIPLE`, `STRAIGHT`, `FOUR_OF_A_KIND`, `THREE_PAIRS`, `FOUR_PAIRS`.
- `combo.cards` — `tuple[Card, ...]` sorted ascending.
- `combo.key_card` — highest card (used for comparisons).
- `combo.is_bomb` — True for tứ quý / 3 đôi thông / 4 đôi thông.

### `beats(new, prev) -> bool`
True iff `new` is a legal response that beats `prev`. Handles both same-type
comparisons and cross-type chops (chặt).

### `can_chop(new, prev) -> bool`
Cross-type override only. Returns False when types match. Same-type comparisons
go via `beats`.

### `enumerate_moves(hand, prev=None) -> list[Combo]`
Every legal play from `hand`. If `prev` is given, only combos that beat it.

## Encoded rules

See module docstring of [`rules.py`](rules.py) for the full chop table. Quick
reference (chuẩn SGK MN, Nhất Ăn Tất):

| `prev` | Beaten by |
|---|---|
| Single 2 | tứ quý, 3 đôi thông, 4 đôi thông |
| Pair 2 | tứ quý, 4 đôi thông |
| Tứ quý | larger tứ quý (same type), 4 đôi thông |
| 3 đôi thông | larger 3 đôi thông (same type), tứ quý, 4 đôi thông |
| 4 đôi thông | larger 4 đôi thông only |

Suit order: ♠ < ♣ < ♦ < ♥. Rank order: 3 < 4 < … < A < 2.

## Why this module is independent

- No external dependencies beyond `dataclasses` / `enum` / `itertools`.
- Deterministic and side-effect free.
- ~150 LoC, fully covered by `tests/core/`.

Other modules (`memory`, `agent`) import from here; `vision` and `io_ctrl` do not.
