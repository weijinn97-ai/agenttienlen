"""Thin wrapper over ultralytics YOLOv8 for card detection.

This file isolates the heavy `ultralytics` import behind a lazy
:meth:`YoloCardDetector.load` call. Imports of :mod:`agenttienlen.vision` do
not pull in torch / ultralytics, so unit tests of the rest of the codebase
stay light.

Workflow:

1. Train a YOLOv8 model on the 53-class dataset (see ``vision/README.md``).
2. Drop ``best.pt`` into ``weights/best.pt``.
3. ``YoloCardDetector(weights="weights/best.pt").infer(frame)`` returns
   :class:`Detection` records grouped by ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agenttienlen.core.card import Card
from agenttienlen.vision.labels import BACK_CLASS_ID, class_id_to_card
from agenttienlen.vision.layout import ROI, RegionName, assign_region, default_layout_1280x720

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    card: Card | None  # None == card-back
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    region: RegionName | None

    @property
    def is_back(self) -> bool:
        return self.card is None


@dataclass(slots=True)
class FrameResult:
    detections: list[Detection]

    def by_region(self, region: RegionName) -> list[Detection]:
        return [d for d in self.detections if d.region == region]

    def cards_in(self, region: RegionName) -> list[Card]:
        return [d.card for d in self.by_region(region) if d.card is not None]

    def backs_in(self, region: RegionName) -> int:
        return sum(1 for d in self.by_region(region) if d.is_back)


class YoloCardDetector:
    """Lazy-loaded YOLOv8 wrapper. Construct cheap; first :meth:`infer` loads weights."""

    def __init__(
        self,
        weights: str | Path = "weights/best.pt",
        *,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        layout: list[ROI] | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.layout = layout or default_layout_1280x720()
        self._model: object | None = None  # lazily set

    def load(self) -> None:
        """Eagerly load the YOLOv8 weights (otherwise loaded on first infer)."""
        from ultralytics import YOLO  # heavy import deferred

        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.weights}. "
                "Train a model and place best.pt there — see vision/README.md."
            )
        self._model = YOLO(str(self.weights))

    def infer(self, frame: np.ndarray) -> FrameResult:
        """Run YOLO on a BGR/RGB numpy frame and return parsed detections."""
        if self._model is None:
            self.load()
        assert self._model is not None
        results = self._model.predict(  # type: ignore[attr-defined]
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        return self._parse(results[0])

    def _parse(self, result: object) -> FrameResult:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return FrameResult(detections=[])
        xywh = boxes.xywh.cpu().numpy()  # type: ignore[attr-defined]
        cls = boxes.cls.cpu().numpy().astype(int)  # type: ignore[attr-defined]
        conf = boxes.conf.cpu().numpy()  # type: ignore[attr-defined]
        detections: list[Detection] = []
        for (cx, cy, w, h), class_id, score in zip(xywh, cls, conf, strict=True):
            card = None if class_id == BACK_CLASS_ID else class_id_to_card(int(class_id))
            region = assign_region(float(cx), float(cy), self.layout)
            detections.append(
                Detection(
                    card=card,
                    confidence=float(score),
                    cx=float(cx),
                    cy=float(cy),
                    width=float(w),
                    height=float(h),
                    region=region,
                )
            )
        return FrameResult(detections=detections)
