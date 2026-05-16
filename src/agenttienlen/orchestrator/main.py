"""Main bot loop.

This is the orchestration entrypoint exposed as the ``agenttienlen`` CLI. It
ties together the vision, memory, agent and io_ctrl modules.

The loop in :func:`_live_run` is intentionally minimal — the real intelligence
lives inside :mod:`agenttienlen.agent` and :mod:`agenttienlen.core`. Heavy
imports (ultralytics, torch, adbutils, cv2) are deferred so that running
``--help`` or ``--dry-run`` does not require the full ML stack.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from agenttienlen.agent.heuristic import HeuristicPolicy
from agenttienlen.agent.policy import Pass, Play, Policy
from agenttienlen.memory.game_state import GameState
from agenttienlen.orchestrator.config import BotConfig, load_config

logger = logging.getLogger("agenttienlen")


def _select_policy(name: str) -> Policy:
    if name == "heuristic":
        return HeuristicPolicy()
    raise ValueError(f"Unknown policy: {name}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenttienlen", description="Real-time Tien Len bot")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't connect to ADB / load YOLO; print intended actions only.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="Stop after N decision ticks (0 = run forever).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = load_config(args.config) if args.config else BotConfig()
    policy = _select_policy(cfg.policy_name)

    if args.dry_run:
        return _dry_run(policy)

    return _live_run(cfg, policy, max_ticks=args.max_ticks)


def _dry_run(policy: Policy) -> int:
    """Run the loop without vision / ADB. Useful for smoke-testing the agent."""
    state = GameState()
    logger.info("Dry run: empty state, policy=%s", type(policy).__name__)
    action = policy.decide(state)
    logger.info("Decision: %s", action)
    return 0


def _live_run(cfg: BotConfig, policy: Policy, *, max_ticks: int) -> int:
    """Live loop: capture → detect → update state → decide → act."""
    from agenttienlen.io_ctrl.actions import GameActions
    from agenttienlen.io_ctrl.adb import AdbController
    from agenttienlen.vision.yolo_detector import YoloCardDetector

    adb = AdbController(serial=cfg.adb_serial)
    detector = YoloCardDetector(
        weights=cfg.weights_path,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
    )
    actions = GameActions(tap=adb.tap)
    state = GameState()
    logger.info("Bot online. weights=%s policy=%s", cfg.weights_path, type(policy).__name__)

    tick = 0
    while True:
        tick += 1
        try:
            frame = adb.screencap()
            result = detector.infer(frame)
            _update_state_from_frame(state, result)
            action = policy.decide(state)
            logger.info("tick=%d action=%s", tick, action)
            _execute(actions, action, state)
        except Exception as e:
            logger.exception("loop error: %s", e)
        if max_ticks and tick >= max_ticks:
            break
        time.sleep(cfg.tick_seconds)
    return 0


def _update_state_from_frame(state: GameState, result: object) -> None:
    """Refresh ``state`` from a vision FrameResult.

    Intentionally light: a richer implementation tracks turn order, last
    player, and trick boundaries. That logic lives in a follow-up
    `turn_detector.py` once calibrated against a real session.
    """
    from agenttienlen.vision.layout import RegionName
    from agenttienlen.vision.yolo_detector import FrameResult

    if not isinstance(result, FrameResult):
        return
    state.hand = sorted(result.cards_in(RegionName.MY_HAND))
    state.deck.set_hand(state.hand)


def _execute(actions: object, action: Play | Pass, state: GameState) -> None:
    """Translate a Play/Pass into taps.

    Assumes the on-screen hand is sorted left→right matching ``state.hand``.
    """
    from agenttienlen.io_ctrl.actions import GameActions

    if not isinstance(actions, GameActions):
        return
    if isinstance(action, Pass):
        actions.click_pass()
        return
    total = len(state.hand)
    indices = [state.hand.index(c) for c in action.combo.cards if c in state.hand]
    actions.click_cards(indices, total)
    actions.click_play()


if __name__ == "__main__":
    sys.exit(main())
