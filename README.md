# agenttienlen

Real-time Tien Len Mien Nam bot — YOLOv8 vision + rule-based decision agent.

> **Trạng thái**: v0.1 — skeleton, core game logic + memory + agent heuristic chuẩn SGK đã sẵn sàng. Vision (YOLOv8) và emulator I/O cần dataset/calibration thực tế.

## Kiến trúc

```
src/agenttienlen/
├── core/           # Pure game logic: Card, Combo, Rules (chuẩn SGK)
├── memory/         # Game state + deck tracker (52-card memory)
├── vision/         # YOLOv8 card detector (53 classes: 52 cards + back)
├── io_ctrl/        # ADB tap/click for MEmu/BlueStacks/LDPlayer
├── agent/          # Decision: heuristic policy (chuẩn SGK), pluggable
└── orchestrator/   # Main loop: capture → detect → decide → act
```

Mỗi module có **interface rõ ràng**, có thể phát triển song song bởi nhiều agent (Devin) khác nhau.

## Luật được encode (Tiến Lên Miền Nam — Nhất Ăn Tất)

- **Thứ tự lá**: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2
- **Thứ tự chất**: ♠ < ♣ < ♦ < ♥
- **Bộ**: lẻ, đôi, sám, sảnh (≥3, không chứa 2), tứ quý, 3 đôi thông, 4 đôi thông
- **Chặt heo**:
  - Lẻ 2 / Đôi 2 ← tứ quý, 3 đôi thông, 4 đôi thông
  - Tứ quý ← tứ quý lớn hơn, 4 đôi thông
  - 3 đôi thông ← 3 đôi thông lớn hơn, tứ quý, 4 đôi thông
  - 4 đôi thông ← 4 đôi thông lớn hơn
- **3♠ đi trước** ván đầu
- **4 đôi thông chặt không theo vòng** (Nhất Ăn Tất)

## Cài đặt

```bash
# Yêu cầu Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Cài thêm khi cần vision + emulator
pip install -e ".[dev,vision,emulator]"
```

## Lệnh thường dùng

```bash
# Unit tests
pytest

# Lint
ruff check src tests
ruff format --check src tests

# Type check
mypy src

# Chạy bot (sau khi có YOLO weights + ADB)
agenttienlen --config config.yaml
```

## Cấu trúc dataset cho YOLOv8

```
dataset/
├── images/
│   ├── train/   *.jpg
│   ├── val/     *.jpg
│   └── test/    *.jpg
├── labels/
│   ├── train/   *.txt   # YOLO format: <class_id> <cx> <cy> <w> <h>
│   ├── val/     *.txt
│   └── test/    *.txt
└── data.yaml    # train/val/test paths + names
```

53 class labels:
- 0–51: từng lá theo thứ tự `rank * 4 + suit_index`
  - rank: 0=3, 1=4, …, 11=A, 12=2
  - suit_index: 0=♠, 1=♣, 2=♦, 3=♥
- 52: `back` (lá úp)

## Tài liệu module

- [src/agenttienlen/core/README.md](src/agenttienlen/core/README.md) — Card / Combo / Rules
- [src/agenttienlen/agent/README.md](src/agenttienlen/agent/README.md) — Chiến thuật heuristic
- [src/agenttienlen/vision/README.md](src/agenttienlen/vision/README.md) — Hướng dẫn train YOLOv8

## License

MIT
