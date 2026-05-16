"""Configuration loading for the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class BotConfig:
    # Vision
    weights_path: Path = Path("weights/best.pt")
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5

    # Capture / control
    adb_serial: str | None = None
    capture_width: int = 1280
    capture_height: int = 720

    # Decision loop
    tick_seconds: float = 1.0
    pass_timeout_seconds: float = 12.0

    # Strategy
    policy_name: str = "heuristic"

    # Buttons / hand layout (overridable for non-1280x720 resolutions)
    extras: dict[str, object] = field(default_factory=dict)


def load_config(path: str | Path) -> BotConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping in {p}")

    cfg = BotConfig()
    for key, value in data.items():
        if key == "weights_path":
            cfg.weights_path = Path(value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extras[key] = value
    return cfg
