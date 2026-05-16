# AGENTS.md — Sổ tay cho các Devin agent cùng làm việc

> Đây là handbook **bắt buộc đọc** trước khi bất kỳ Devin agent nào bắt đầu code
> trên repo này. Mục tiêu: nhiều agent làm song song mà không merge conflict,
> không relitigate quyết định cũ, không phá kiến trúc.

---

## 0. TL;DR (đọc 30 giây)

- Repo: https://github.com/weijinn97-ai/agenttienlen — bot Tiến Lên Miền Nam realtime
- Stack: Python 3.11+, ruff (lint+format), pytest, mypy strict, GitHub Actions CI
- Kiến trúc: **6 module độc lập** — một agent claim một module hoặc một task.
- Vision: **YOLOv8** (đã chốt, KHÔNG đổi sang template matching).
- Agent: **heuristic chuẩn SGK** (đã chốt, KHÔNG dùng ML/RL ở v1).
- Branch: `devin/<timestamp>-<slug>`; KHÔNG push thẳng `main`; mọi thay đổi qua PR.
- Lint + test phải pass trước khi mở PR: `ruff check src tests && ruff format --check src tests && pytest -ra`.

---

## 1. Mục tiêu dự án

Bot chơi Tiến Lên Miền Nam (luật **Nhất Ăn Tất**) tự động, real-time:

1. **Nhìn**: nhận diện toàn bộ lá bài trên màn hình (tay mình + giữa bàn + 3 đối thủ) bằng YOLOv8.
2. **Ghi nhớ**: track 52-card deck (lá nào đã đánh, lá nào còn) + game state.
3. **Quyết định**: heuristic chuẩn SGK (đánh / chặt / bỏ lượt / giữ bom).
4. **Tap**: tự bấm bài qua ADB vào emulator (MEmu).
5. **Modular**: nhiều Devin agent làm song song không đụng nhau.

---

## 2. Trạng thái hiện tại (v0.1 — skeleton trên `main`)

| Module | Trạng thái | Test | Ghi chú |
|---|---|---|---|
| `core/` | ✅ STABLE | 41 tests pass | Card, Combo, beats, enumerate_moves — luật SGK |
| `memory/` | ✅ STABLE | 6 tests pass | GameState, DeckTracker, PlayerSeat |
| `agent/` | ✅ STABLE | 11 tests pass | HeuristicPolicy chuẩn SGK |
| `vision/` | 🟡 SKELETON | — | YOLO wrapper sẵn, **cần dataset + train** |
| `io_ctrl/` | 🟡 SKELETON | — | ADB tap sẵn, **cần calibrate trên MEmu thật** |
| `orchestrator/` | 🟡 SKELETON | — | Main loop sẵn, **cần state machine** |

**62 unit tests pass** trên `main`. CI (GitHub Actions) chạy Python 3.11 + 3.12.

Repo bắt đầu từ commit `731cebb` (initial skeleton). Đọc git log để hiểu lịch sử.

---

## 3. Kiến trúc 6 module

```
src/agenttienlen/
├── core/           # Pure game logic — KHÔNG đụng trừ khi user cập nhật luật
├── memory/         # GameState + DeckTracker
├── vision/         # YOLOv8 card detector
├── agent/          # Decision engine (heuristic + future minimax)
├── io_ctrl/        # ADB tap / click
└── orchestrator/   # Main loop: capture → detect → decide → act
```

Interface giữa các module qua **dataclass + Protocol**, KHÔNG qua global state.
Mỗi module có `README.md` riêng (xem `src/agenttienlen/<module>/README.md`).

### Liên lạc giữa các module

```
[Frame 1280x720 BGR numpy]
        │
        ▼
vision/yolo_detector.YoloCardDetector.infer(frame)
        │ → FrameResult { detections, by_region(), cards_in() }
        ▼
orchestrator/main._update_state_from_frame()
        │ → GameState { hand, current_combo, last_player, seat_card_counts, ... }
        ▼
agent/heuristic.HeuristicPolicy.decide(state)
        │ → Action (Play(combo) | Pass(reason))
        ▼
orchestrator/main._execute()
        │ → click coordinates
        ▼
io_ctrl/actions.GameActions.click_card_at / click_play / click_pass
        │
        ▼
ADB tap → Emulator (MEmu, 1280x720)
```

---

## 4. Quyết định đã chốt (KHÔNG relitigate)

| # | Quyết định | Lý do |
|---|---|---|
| 1 | Repo riêng `agenttienlen` (không reuse `Tienlenchuan`) | User yêu cầu tách hẳn |
| 2 | Python 3.11+, ruff, mypy strict | Type-safe, stable |
| 3 | **YOLOv8** cho vision (KHÔNG template matching) | Cards fan-out overlap, multi-size (3 scales: 75x125 / 50x85 / 40x70 px) |
| 4 | Heuristic agent chuẩn SGK ở v1 (KHÔNG ML/RL) | Đủ tốt, dễ debug, không cần training data lớn |
| 5 | MEmu emulator + ADB | User chỉ định |
| 6 | Synthetic dataset từ 52 ảnh lá + augment | Tiết kiệm label tay |
| 7 | Heuristic priority: 3♠ đi trước → dump lẻ nhỏ → không phá đôi/sám → chặt 2 bằng bom | Theo CLAUDE.md user |
| 8 | YOLO 53 classes (52 lá + 1 back) | Cần track lá úp để đếm lá đối thủ |

Nếu một quyết định trên thay đổi → update `AGENTS.md` trong cùng PR.

---

## 5. Roadmap (task để claim)

Một agent comment vào PR hoặc chat user để **claim** task trước khi code, tránh đụng.

### 🔥 Ưu tiên cao — sau khi user cấp 52 ảnh template

| Task ID | Mô tả | File output | Khó | ~Time |
|---|---|---|---|---|
| `vision-extract` | `scripts/extract_cards_from_screenshot.py` — cắt lá từ screenshot game | scripts/* | Easy | 2-3h |
| `vision-synth` | `scripts/synthetic_generator.py` — render 10K ảnh fan-out (3 scales: hand 75x125, table 50x85, opp 40x70) + multi-background | scripts/*, dataset/* | Medium | 4-6h |
| `vision-train` | `scripts/train_yolov8n.py` Windows/CUDA, output `weights/best.pt` + benchmark FPS | scripts/*, weights/* | Easy | 2-3h |
| `vision-calibrate` | Đo ROI thực từ screenshots, update `vision/layout.py` | src/agenttienlen/vision/layout.py | Easy | 1h |

### 🟡 Ưu tiên trung — chạy song song với vision

| Task ID | Mô tả | Khó | ~Time |
|---|---|---|---|
| `io-calibrate` | Đo tap coordinates trên MEmu thật, update `io_ctrl/actions.py` | Easy | 1-2h |
| `io-screencap` | Test capture frame ổn định qua ADB screencap | Easy | 1h |
| `orch-statemachine` | Detect game flow (chờ lượt / game mới / network error / disconnect) | Hard | 6-8h |
| `orch-config` | Mở rộng `config.example.yaml` với calibration values | Easy | 1h |

### 🟢 Ưu tiên thấp — sau khi v1 chạy được

| Task ID | Mô tả | Khó | ~Time |
|---|---|---|---|
| `agent-v2-minimax` | Minimax 2-ply lookahead | Hard | 8-10h |
| `agent-v2-rollout` | Monte Carlo rollout với deck unseen | Hard | 10-12h |
| `vision-v2-onnx` | Export ONNX cho deploy CPU | Easy | 2h |
| `tools-replay` | Tool replay từ log để debug | Medium | 4h |
| `docs-strategy` | Documentation chiến thuật SGK đầy đủ | Easy | 2h |

---

## 6. Quy tắc làm việc

### 6.1. Branch + PR

- Branch name: `devin/<timestamp>-<slug>` (vd: `devin/1778922000-vision-synthetic`)
- `<timestamp>` = `$(date +%s)` để tránh collision giữa các agent
- `<slug>` = task ID hoặc mô tả ngắn (lowercase, hyphen)
- KHÔNG push trực tiếp lên `main`. Mọi thay đổi qua PR.
- KHÔNG force-push `main`. Trên feature branch có thể `--force-with-lease` sau rebase.
- Một PR = một task (1 module hoặc 1 chủ đề nhỏ). KHÔNG gộp nhiều task.
- PR title: `<task-id>: <mô tả ngắn>` (vd: `vision-synth: synthetic dataset generator`).

### 6.2. Lint + test bắt buộc trước khi push

```bash
ruff check src tests
ruff format --check src tests
pytest -ra
```

Cả 3 phải exit 0. CI sẽ chạy lại nhưng đừng đẩy lỗi lên CI.

### 6.3. Coding conventions

- **Surgical changes**: chỉ sửa file liên quan tới task. Không refactor râu ria.
- **Comment tối thiểu**: chỉ comment khi logic phức tạp. KHÔNG comment kể chuyện diff.
- **Vietnamese OK** trong docstring/comment/README; **English** trong code identifiers (class, function, variable).
- KHÔNG dùng `Any`, `getattr`, `setattr`. Đọc type hints kỹ.
- KHÔNG hard-code workaround. Nếu test sai luật, raise lên user.
- Imports đặt trên đầu file. KHÔNG import lồng trong function/class.

### 6.4. Không được làm

- ❌ Reuse code từ repo cũ `Tienlenchuan` của user
- ❌ Amend commit (chỉ thêm commit mới)
- ❌ `git add .` (dễ commit file rác)
- ❌ Commit `.env`, secrets, weights nặng (>10MB), dataset images
- ❌ Sửa test cho pass thay vì sửa code (trừ khi user yêu cầu)
- ❌ Đổi quyết định trong mục 4 không hỏi user trước

### 6.5. Khi gặp conflict với agent khác

- 2 PR cùng động vào 1 file → PR nhỏ hơn merge trước, PR lớn hơn rebase sau
- 2 agent cùng claim 1 task → agent nào comment vào chat user trước thắng
- Module boundary mới phát sinh → discuss trong PR comment, KHÔNG tự quyết

---

## 7. Setup môi trường

### 7.1. Linux/macOS (agent dev)

```bash
git clone https://github.com/weijinn97-ai/agenttienlen.git
cd agenttienlen
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Tuỳ task, cài thêm:
pip install -e ".[dev,vision]"     # ultralytics, torch — cho vision module
pip install -e ".[dev,emulator]"   # adbutils — cho io_ctrl
```

Devin có blueprint sẵn — session mới sẽ tự setup. Xem `.devin/blueprint.yaml` (nếu user đã approve) hoặc đọc lệnh từ `pyproject.toml`.

### 7.2. Windows (cho user train YOLO trên RTX 3050)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,vision]"
```

Train YOLOv8n trên RTX 3050 6GB: ~15-25 phút cho 100 epochs trên 10K ảnh synthetic.

---

## 8. File map

```
agenttienlen/
├── AGENTS.md                          ← Bạn đang đọc
├── README.md                          ← Mô tả + setup user-facing
├── pyproject.toml                     ← Deps + ruff/pytest/mypy config
├── config.example.yaml                ← Config mẫu cho orchestrator
├── .github/workflows/ci.yml           ← CI
├── src/agenttienlen/
│   ├── core/         Card, Combo, beats, enumerate (STABLE)
│   ├── memory/       GameState, DeckTracker (STABLE)
│   ├── vision/       YOLO detector skeleton + labels + ROI layout
│   ├── agent/        HeuristicPolicy + Policy Protocol (STABLE)
│   ├── io_ctrl/      ADB tap + GameActions
│   └── orchestrator/ Main loop + config
└── tests/            62 tests cho core/memory/agent
```

Sau khi vision có dataset, thêm:

```
scripts/
├── extract_cards_from_screenshot.py
├── synthetic_generator.py
└── train_yolov8n.py
dataset/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── data.yaml
weights/
└── best.pt
```

---

## 9. Workflow đặt task mới (cho user)

User muốn thêm task → comment vào chat:
1. Tên task (vd: `vision-synth`)
2. Mô tả 1-2 câu
3. Output expected (file gì, behavior gì)
4. Acceptance criteria (test gì pass, demo gì OK)

Agent claim task → comment "tôi claim `<task-id>`" → bắt đầu code.

---

## 10. Liên hệ

- **User**: `trungduongtube2075` (Vietnamese — comm tiếng Việt)
- **GitHub org**: `weijinn97-ai`
- **Repo**: https://github.com/weijinn97-ai/agenttienlen

Đọc xong AGENTS.md → bắt đầu code module/task được phân công. Không cần đọc lại các thread cũ trừ khi user yêu cầu.
