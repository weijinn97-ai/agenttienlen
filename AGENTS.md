# AGENTS.md — Sổ tay cho các Devin agent cùng làm việc

> Đây là handbook **bắt buộc đọc** trước khi bất kỳ Devin agent nào bắt đầu code
> trên repo này. Mục tiêu: nhiều agent làm song song mà không merge conflict,
> không relitigate quyết định cũ, không phá kiến trúc.

---

## 0. TL;DR (đọc 30 giây)

- Repo: https://github.com/weijinn97-ai/agenttienlen — bot Tiến Lên Miền Nam realtime
- Stack: Python 3.11+, ruff (lint+format), pytest, mypy strict, GitHub Actions CI
- Kiến trúc: **6 module độc lập** — một agent claim một module hoặc một task.
- Vision: **multi-reader theo ROI** (xem §3.5) — `HandReader` (corner classifier) + `TableReader` (YOLO) + `OpponentReader` (back counter) + `PlayerIdentityReader` (OCR), KHÔNG template matching, KHÔNG single global YOLO.
- Agent: **heuristic chuẩn SGK** (đã chốt, KHÔNG dùng ML/RL ở v1).
- Branch: `devin/<timestamp>-<slug>`; KHÔNG push thẳng `main`; mọi thay đổi qua PR.
- Lint + test phải pass trước khi mở PR: `ruff check src tests && ruff format --check src tests && pytest -ra`.

---

## 1. Mục tiêu dự án

Bot chơi Tiến Lên Miền Nam (luật **Nhất Ăn Tất**) tự động, real-time:

1. **Nhìn**: chia màn hình theo ROI → mỗi vùng có reader riêng (tay mình: corner classifier, giữa bàn: YOLO, đối thủ: back-card counter, identity: OCR tên) → trả `StructuredFrame`.
2. **Ghi nhớ**: track 52-card deck (lá nào đã đánh, lá nào còn) + game state + `SeatMap` (seat ↔ display name).
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
├── vision/         # Multi-reader (Hand corner classifier + Table YOLO + Opponent counter + Identity OCR)
├── agent/          # Decision engine (heuristic + future minimax)
├── io_ctrl/        # ADB tap / click
└── orchestrator/   # Main loop: capture → detect → decide → act
```

Interface giữa các module qua **dataclass + Protocol**, KHÔNG qua global state.
Mỗi module có `README.md` riêng (xem `src/agenttienlen/<module>/README.md`).

### 3.1. Liên lạc giữa các module

```
[Frame 1280x720 BGR numpy]
        │
        ▼
vision/layout_router.split(frame)
        │ → {MY_HAND, TABLE, OPP_LEFT, OPP_TOP, OPP_RIGHT, BUTTONS, TURN_INDICATOR} crops
        ▼
vision/pipeline.StructuredFramePipeline.read(crops)
        │   ├─ HandReader: segment slots → corner classifier (rank+suit) → my_hand_candidates
        │   ├─ TableReader: YOLO detect các lá ngửa → table_cards + combo
        │   ├─ OpponentReader: đếm back-card per seat → opponent_back_counts
        │   ├─ PlayerIdentityReader: OCR tên/avatar → player_profiles, SeatMap
        │   └─ TurnDetector: button/highlight/timer → ui_buttons, turn_indicators
        │ → StructuredFrame (xem §3.5)
        ▼
vision/stabilizer (temporal voting qua N frame, xem §3.3)
        │ → StableStructuredFrame (mỗi vùng có cờ ổn định riêng)
        ▼
orchestrator/state_machine.transition(stable_frame)
        │ → State (xem §3.2) — agent chỉ được gọi khi state ∈ {MY_TURN_FREE, MY_TURN_RESPONSE}
        ▼
memory/game_state._update_from_stable_frame()
        │ → GameState { hand, current_combo, last_play (player+cards), seat_map, opponent_card_count, ... }
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
        │
        ▼
io_ctrl/verification.verify_action_applied()
        │ → OK → loop tiếp; FAIL → state = RECOVERING
        ▼
tools/replay_logger.log(frame, structured_frame, stable, state, decision, taps, verify, seat_map)
```

> **Nguyên tắc**: "Nhận diện vị trí trước, nhận diện quân bài sau, rồi dùng state machine để hiểu lá đó thuộc về ai."
> Vision KHÔNG trả `List[Card]` phẳng. Vision trả `StructuredFrame` chia theo vùng. Việc gán lá cho người chơi thuộc về `GameState` + `TurnDetector`.

### 3.2. Orchestrator state machine 7-state (safety spec)

Quyết định nguy hiểm nhất của bot real-time là **click sai thời điểm** (chia bài, animate, mạng lag).
Mọi quyết định `agent.decide()` chỉ được gọi khi state cho phép.

| State | Điều kiện nhận biết | Có gọi `agent.decide`? | Hành động |
|---|---|---|---|
| `WAITING_FOR_GAME` | Chưa đủ bài hoặc chưa vào ván | Không | Chỉ capture, không click |
| `DEALING_OR_SORTING` | Bài đang chia / animate / xếp lại | Không | Chờ stabilizer ổn định trở lại |
| `WAITING_TURN` | Không có nút `Đánh`/`Bỏ`, không phải lượt mình | Không | Update bài giữa bàn + đối thủ + DeckTracker |
| `MY_TURN_FREE` | Tới lượt mình, bàn trống / được đi vòng mới | **Có** | Agent chọn combo chủ động (lead) |
| `MY_TURN_RESPONSE` | Tới lượt mình, cần đè combo hiện tại | **Có** | Agent chọn combo đè / chặt / bỏ |
| `EXECUTING_ACTION` | Đã chọn action, đang tap | Không | Tap lá + tap `Đánh`/`Bỏ`, chờ verification |
| `RECOVERING` | Detection mâu thuẫn / mạng lag / tap fail / timeout | Không | Không đánh bừa; thu thập thêm frame, so state cũ, hoặc reset ván nếu quá lâu |

Guards bắt buộc:
- Transition vào `MY_TURN_*` phải có **stable frame ≥ 3** đồng thuận có nút `Đánh`/`Bỏ`.
- Sau `EXECUTING_ACTION`, nếu state không đổi trong T=2s → `RECOVERING`.
- `RECOVERING` quá T=10s → log fatal, dừng bot, yêu cầu user can thiệp.

### 3.3. Frame stabilizer (temporal voting)

YOLO flicker ở các thời điểm: chia bài, lá bay vào table, popup, animate xếp lá.
Không được biến mỗi frame raw thành state change.

| Tín hiệu | Cách stabilize đề xuất | Khi không ổn định |
|---|---|---|
| Bài trên tay | Cùng số lượng + cùng card set qua **2-3 frame liên tiếp** | Không update hand; giữ state cũ |
| Combo giữa bàn | Cùng combo hoặc cùng vùng detection qua **2 frame** | Chưa set `current_combo` |
| Nút `Đánh` / `Bỏ` | Phát hiện ổn định **3 frame** hoặc OCR/UI detector xác nhận | Chưa chuyển sang `MY_TURN_*` |
| Số bài đối thủ (back) | Back-card count ổn định qua **2 frame** | Không update seat count |

Thư mục code dự kiến: `src/agenttienlen/vision/stabilizer.py` — input: chuỗi `FrameResult`, output: `StableFrame` hoặc `None`.

### 3.4. Logging / replay (debug bắt buộc)

Bot real-time thua/đánh sai → nhìn lại màn hình không đủ. Mỗi action phải log đủ để **replay offline** không cần MEmu.

Mỗi tick log:
- `hand_id`, `frame_id`, `timestamp_ms` (để truy vết theo ván)
- Raw frame (jpg, downsampled OK)
- `StructuredFrame` (per-region, kèm bbox + confidence mỗi candidate)
- Stabilized state (sau temporal voting)
- `GameState` (hand, current_combo, last_play, seat counts, …)
- `SeatMap` snapshot (seat ↔ display_name + name_confidence + avatar_roi crop)
- Legal moves từ `enumerate_moves()`
- Policy decision + lý do chọn
- Tap coordinates
- Post-tap verification kết quả

Mỗi event quan trọng (vd `opponent_play_detected`) phải kèm `seat` + `display_name` + `name_confidence` + `hand_id` + `frame_id` để người xem log truy nguyên đúng người chơi.

Thư mục code dự kiến: `tools/replay_logger.py` + `tools/replay_viewer.py`.

### 3.5. Vision multi-reader theo ROI (KHÔNG single global YOLO)

Lý do KHÔNG dùng 1 YOLOv8 global phủ cả màn hình:

- Bài trên tay 60-80% bị che (fan-out overlap), chỉ lộ corner ~20×25 px → YOLO dễ nhầm / miss.
- 3 size scale khác nhau (hand 75×125 / table 50×85 / opponent 40×70 px) → detector chung khó tối ưu.
- Background, ROI, ngữ nghĩa (lá của ai) khác nhau giữa vùng → tách reader dễ debug + thay thế từng module.

5 reader độc lập + 1 stabilizer:

| Reader | Input | Output | Approach |
|---|---|---|---|
| `LayoutRouter` | Full frame | ROI crops (`MY_HAND`, `TABLE`, `OPP_*`, `BUTTONS`, `TURN_INDICATOR`) | Hard-coded ROI từ calibration |
| `HandReader` | `MY_HAND` ROI | `my_hand_candidates: list[CardCandidate]` | Segment slot theo trục X → crop corner trang trên-trái → rank classifier (13) + suit classifier (4) |
| `TableReader` | `TABLE` ROI | `table_cards: list[CardCandidate]` + `combo` | YOLOv8 detect 52 cards (không cần back ở table) → parse combo |
| `OpponentReader` | `OPP_*` ROIs | `opponent_back_counts: dict[Seat, int]` | Edge/contour hoặc template back → đếm số lá, KHÔNG đọc rank/suit |
| `PlayerIdentityReader` | Avatar/name ROIs | `player_profiles: dict[Seat, PlayerProfile]` + `SeatMap` | OCR tên + crop avatar; chốt khi `WAITING_FOR_GAME`/`DEALING_OR_SORTING` |
| `TurnDetector` | `BUTTONS` + `TURN_INDICATOR` ROIs | `ui_buttons`, `turn_indicators` | Phone-icon highlight / nu00fat `Đánh`/`Bỏ` / timer |

`StructuredFrame` (output của vision/pipeline):

```python
@dataclass
class StructuredFrame:
    my_hand_candidates: list[CardCandidate]
    table_cards: list[CardCandidate]
    opponent_back_counts: dict[Seat, int]
    player_profiles: dict[Seat, PlayerProfile]
    ui_buttons: dict[str, ButtonState]
    turn_indicators: dict[Seat, float]
    timestamp_ms: int
    frame_id: int
```

`SeatMap` (bắt buộc có trong mọi log event):

```python
@dataclass
class PlayerProfile:
    seat: Seat
    display_name: str | None
    raw_name_text: str | None
    name_confidence: float
    avatar_roi: Rect
    name_roi: Rect
    is_bot: bool = False

@dataclass
class SeatMap:
    hand_id: str
    room_id: str | None
    players: dict[Seat, PlayerProfile]
```

Nguyên tắc: `Seat` quyết định logic game; `display_name` chỉ dùng cho log/replay/debug.
Không dùng `display_name` để quyết định lượt hay chiến thuật.

### 3.6. Module contracts (để agent không đè nhau)

| Module | Input | Output | KHÔNG được làm |
|---|---|---|---|
| `vision/layout_router` | Screenshot | ROI crops | Parse card. |
| `vision/hand_reader` | `MY_HAND` ROI | `my_hand_candidates` | Quyết định nước đánh. |
| `vision/table_reader` | `TABLE` ROI | `table_cards`, `combo` | Tự gán người chơi. |
| `vision/opponent_reader` | Opponent ROIs | `back_counts` | Đoán rank/suit lá úp. |
| `vision/identity_reader` | Avatar/name ROIs | `PlayerProfile`, `SeatMap` | Dùng tên để quyết định lượt / strategy. |
| `vision/turn_detector` | UI/seat/button ROIs | `current_turn`, confidence | Mutate `GameState` trực tiếp. |
| `vision/stabilizer` | Observations | Stable observations | Sinh quyết định đánh. |
| `memory/game_state` | Stable obs + `SeatMap` | `GameState` chính thức | Xử lý ảnh raw. |
| `tools/replay_logger` | Frame, detections, state, action, `SeatMap` | Debug bundle theo ván | Sửa state hoặc chọn action. |
| `agent/heuristic` | `GameState` | `Action` | Đọc screenshot trực tiếp. |
| `io_ctrl/executor` | `Action` | Tap events + result | Chọn strategy. |

---

## 4. Quyết định đã chốt (KHÔNG relitigate)

| # | Quyết định | Lý do |
|---|---|---|
| 1 | Repo riêng `agenttienlen` (không reuse `Tienlenchuan`) | User yêu cầu tách hẳn |
| 2 | Python 3.11+, ruff, mypy strict | Type-safe, stable |
| 3 | **Vision multi-reader theo ROI** (xem §3.5): HandReader corner classifier, TableReader YOLOv8, OpponentReader back counter, PlayerIdentityReader OCR. KHÔNG template matching, KHÔNG single global YOLO. | Cards fan-out overlap, multi-size (3 scales: 75x125 / 50x85 / 40x70 px) |
| 4 | Heuristic agent chuẩn SGK ở v1 (KHÔNG ML/RL) | Đủ tốt, dễ debug, không cần training data lớn |
| 5 | MEmu emulator + ADB | User chỉ định |
| 6 | Synthetic dataset từ 52 ảnh lá + 1 back + augment | Tiết kiệm label tay; dùng cho cả TableReader YOLO và HandReader corner classifier |
| 7 | Heuristic priority: 3♠ đi trước → dump lẻ nhỏ → không phá đôi/sám → chặt 2 bằng bom | Theo CLAUDE.md user |
| 8 | **Multi-model** thay vì single YOLO 53 classes: TableReader YOLO 52 classes (lá ngửa giữa bàn), HandReader rank-classifier 13 + suit-classifier 4 (corner crop), OpponentReader đếm back không cần model | HandReader corner ~20x25 px → classifier robust hơn detector |
| 9 | `SeatMap` (seat ↔ display_name) gắn vào mọi log event | Replay/debug bắt buộc biết chính xác người chơi nào |

Nếu một quyết định trên thay đổi → update `AGENTS.md` trong cùng PR.

---

## 5. Roadmap (task để claim)

Một agent comment vào PR hoặc chat user để **claim** task trước khi code, tránh đụng.

### 🔥 Ưu tiên cao — vision pipeline multi-reader (xem §3.5)

Thứ tự tối ưu: `vision-calibrate` → `vision-layout-router` → `vision-extract` → (4 reader song song) → `vision-pipeline-integrate`.

| Task ID | Mô tả | File output | Khó | ~Time |
|---|---|---|---|---|
| `vision-calibrate` | Đo ROI thực từ screenshots (MY_HAND, TABLE, OPP_*, BUTTONS, TURN_INDICATOR, AVATAR_*) → `vision/layout.py` | `src/agenttienlen/vision/layout.py` | Easy | 1-2h |
| `vision-layout-router` | `LayoutRouter.split(frame)` trả ROI crops + `Rect` per region | `src/agenttienlen/vision/layout_router.py` + tests | Easy | 2h |
| `vision-extract` | `scripts/extract_cards_from_screenshot.py` — cắt 52 lá (nguồn cho synth) + corner crops + 1 back | `scripts/extract_cards_from_screenshot.py` | Easy | 2-3h |
| `vision-hand-reader` | Slot segmenter + corner crop + **rank-13 classifier + suit-4 classifier** + stabilizer | `src/agenttienlen/vision/hand_segmenter.py`, `card_corner_classifier.py`, `hand_reader.py` + tests | Medium | 6-8h |
| `vision-table-reader` | `TableReader` — YOLOv8 detect lá ngửa (52 classes) + combo parse từ `core/classify.py` | `src/agenttienlen/vision/table_reader.py` + `scripts/train_table_yolo.py` + tests | Medium | 4-6h |
| `vision-opponent-reader` | `OpponentReader` — đếm back-card per seat (edge/contour) | `src/agenttienlen/vision/opponent_reader.py` + tests | Easy | 2-3h |
| `vision-identity-reader` | `PlayerIdentityReader` — OCR display_name + crop avatar → `SeatMap` | `src/agenttienlen/vision/identity_reader.py` + tests | Medium | 4-5h |
| `vision-turn-detector` | Detect nút `Đánh`/`Bỏ`/`Bỏ lượt` + highlight seat + timer | `src/agenttienlen/vision/turn_detector.py` + tests | Medium | 3-4h |
| `vision-pipeline-integrate` | `StructuredFramePipeline.read()` ghép 5 reader + stabilizer | `src/agenttienlen/vision/pipeline.py` + tests | Easy | 2-3h |
| `vision-synth-corner` | Synth corner crops cho rank-13 + suit-4 classifier (augment: rotation, blur, lighting, partial occlusion) | `scripts/synthetic_corner_generator.py` + dataset | Medium | 3-4h |
| `vision-synth-table` | Synth full-card images cho TableReader YOLO | `scripts/synthetic_table_generator.py` + dataset | Medium | 3-4h |

### 🟡 Ưu tiên trung — orchestrator safety + io reliability (chạy song song với vision)

Các task này không bị block bởi vision, có thể claim song song từ đầu.

Đây là layer an toàn để bot **không click bừa** — phải có TRƯỚC khi cho bot chạy thật.
Spec chi tiết: §3.2 (state machine), §3.3 (stabilizer), §3.4 (replay logger).

| Task ID | Mô tả | File output | Khó | ~Time |
|---|---|---|---|---|
| `orch-statemachine-spec` | Implement 7-state orchestrator (xem §3.2), transition guards, timeout, recovery | `src/agenttienlen/orchestrator/state_machine.py` + tests | Hard | 6-8h |
| `vision-stabilizer` | Temporal voting (N=2-3 frame) cho hand/table/back-card/button (xem §3.3) | `src/agenttienlen/vision/stabilizer.py` + tests | Medium | 3-4h |
| `tools-replay` | Replay log frame + detection + state + action để debug offline (xem §3.4) | `tools/replay_logger.py`, `tools/replay_viewer.py` | Medium | 4-5h |
| `io-tap-verification` | Sau tap, verify state thay đổi đúng; fail → transition `RECOVERING` | `src/agenttienlen/io_ctrl/verification.py` + orchestrator integration | Easy | 2-3h |
| `orch-timing-budget` | Đo + log latency ms/stage (capture/vision/decide/tap), cảnh báo khi vượt budget | `src/agenttienlen/orchestrator/timing.py` + docs | Easy | 1-2h |
| `io-calibrate` | Đo tap coordinates trên MEmu thật, update `io_ctrl/actions.py` | `src/agenttienlen/io_ctrl/actions.py` | Easy | 1-2h |
| `io-screencap` | Test capture frame ổn định qua ADB screencap | `src/agenttienlen/io_ctrl/adb.py` + tests | Easy | 1h |
| `orch-config` | Mở rộng `config.example.yaml` với calibration + state machine values | `config.example.yaml` | Easy | 1h |

### 🟢 Ưu tiên thấp — sau khi v1 chạy được

| Task ID | Mô tả | Khó | ~Time |
|---|---|---|---|
| `agent-v2-minimax` | Minimax 2-ply lookahead | Hard | 8-10h |
| `agent-v2-rollout` | Monte Carlo rollout với deck unseen | Hard | 10-12h |
| `vision-v2-onnx` | Export ONNX cho deploy CPU | Easy | 2h |
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

Sau khi vision multi-reader có dataset + weights, thêm:

```
src/agenttienlen/vision/
├── layout_router.py             (vision-layout-router)
├── hand_segmenter.py            (vision-hand-reader)
├── card_corner_classifier.py    (vision-hand-reader)
├── hand_reader.py               (vision-hand-reader)
├── table_reader.py              (vision-table-reader)
├── opponent_reader.py           (vision-opponent-reader)
├── identity_reader.py           (vision-identity-reader)
├── turn_detector.py             (vision-turn-detector)
├── stabilizer.py                (vision-stabilizer)
└── pipeline.py                  (vision-pipeline-integrate)

scripts/
├── extract_cards_from_screenshot.py  (vision-extract)
├── synthetic_corner_generator.py     (vision-synth-corner)
├── synthetic_table_generator.py      (vision-synth-table)
├── train_corner_classifier.py        (vision-hand-reader)
└── train_table_yolo.py               (vision-table-reader)

dataset/
├── corner/{train,val,test}/<class>/*.jpg   (rank+suit classifier)
├── table/{images,labels}/{train,val,test}/*  (table YOLO)
└── data.yaml

weights/
├── corner_rank.pt
├── corner_suit.pt
└── table_yolo.pt
```

Sau khi orchestrator safety + replay làm xong, thêm:

```
src/agenttienlen/
├── orchestrator/state_machine.py   (§3.2)
├── orchestrator/timing.py          (orch-timing-budget)
├── vision/stabilizer.py            (§3.3)
└── io_ctrl/verification.py         (io-tap-verification)
tools/
├── replay_logger.py                (§3.4)
└── replay_viewer.py
logs/<game_session_id>/             (replay data, gitignored)
├── frames/*.jpg
└── tick_*.json
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
