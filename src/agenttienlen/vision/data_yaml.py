"""Generate `data.yaml` for YOLOv8 training from the canonical class list."""

from __future__ import annotations

from pathlib import Path

from agenttienlen.vision.labels import CARD_CLASS_NAMES, NUM_CLASSES


def write_data_yaml(
    dataset_root: Path,
    output: Path | None = None,
    train_subdir: str = "images/train",
    val_subdir: str = "images/val",
    test_subdir: str = "images/test",
) -> Path:
    """Write a YOLOv8-compatible ``data.yaml`` next to the dataset.

    Returns the path to the written file. Idempotent.
    """
    dataset_root = dataset_root.resolve()
    output = output or (dataset_root / "data.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"path: {dataset_root}",
        f"train: {train_subdir}",
        f"val: {val_subdir}",
        f"test: {test_subdir}",
        f"nc: {NUM_CLASSES}",
        "names:",
    ]
    for i, name in enumerate(CARD_CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
