# `vision` — YOLOv8 card detector

Detect *all* card faces (and card-backs) anywhere on the game screen, then
bucket them into semantic regions (my hand, table, each opponent).

## Modules

| File | Purpose |
|---|---|
| `labels.py` | 53-class mapping: 52 cards + `back`. Stable across all sessions. |
| `layout.py` | ROI rectangles for 1280×720; `scale_layout()` for other sizes. |
| `yolo_detector.py` | Lazy YOLOv8 wrapper → `FrameResult` with `Detection` records. |
| `data_yaml.py` | Generate `data.yaml` for training. |

## Class list

```
class_id = rank * 4 + suit
  rank: 0=3, 1=4, …, 11=A, 12=2
  suit: 0=♠, 1=♣, 2=♦, 3=♥
class_id 52 = back (card face-down)
```

Total 53 classes.

## Training a YOLOv8 model

1. Collect screenshots from the target game (a few hundred frames cover most
   layouts; aim for at least 10 instances of each rank+suit).
2. Label using e.g. [Roboflow](https://roboflow.com) or
   [label-studio](https://labelstud.io). YOLO TXT format:
   `<class_id> <cx> <cy> <w> <h>` (normalized 0–1).
3. Split into `images/train`, `images/val`, `images/test` + matching `labels/*`.
4. Generate `data.yaml`:
   ```python
   from pathlib import Path
   from agenttienlen.vision.data_yaml import write_data_yaml
   write_data_yaml(Path("dataset"))
   ```
5. Train:
   ```bash
   yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=80 imgsz=640
   ```
6. Copy `runs/detect/train/weights/best.pt` → `weights/best.pt`.

## Using the detector

```python
import cv2
from agenttienlen.vision.yolo_detector import YoloCardDetector

detector = YoloCardDetector(weights="weights/best.pt")
frame = cv2.imread("capture.png")
result = detector.infer(frame)

my_hand_cards = result.cards_in(RegionName.MY_HAND)
table_cards = result.cards_in(RegionName.TABLE)
opp_top_backs = result.backs_in(RegionName.OPP_TOP)
```

## ROI calibration

`default_layout_1280x720()` is tuned against the project screenshots. Refine
in `tools/calibrate.py` for your specific app.

## Dataset directory (gitignored)

```
dataset/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── data.yaml         # generated via write_data_yaml()
weights/
└── best.pt           # output of YOLOv8 training
```
