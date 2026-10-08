"""DQN 신호 제어 시각화 (tkinter).

한 번의 '실행(run)'은 고정된 설정으로 정해진 시간만큼 시뮬레이션을 돌린다.
실행 중에는 실험 조건을 바꿀 수 없고(그래야 평균 지표가 하나의 조건을 대표한다),
실행이 끝나면 설정을 수정해 새 실행을 시작할 수 있다.
지난 실행 결과는 아래 '실행 기록'에 남아 조건별 비교에 쓴다.

- 4방향 × (직진·좌회전) = 8개 차로에 차가 줄 서고, 초록불에 교차로를 통과해 빠져나간다.
- 한국처럼 우측통행. 좌회전 차로가 중앙선 쪽, 직진 차로가 바깥쪽이다.
- 차 색깔은 '같이 초록불을 받는 페이즈'별로 같다. 같은 색 차들이 함께 움직인다.
- DQN 제어 / 고정주기 신호를 바꿔가며 같은 조건에서 성능 비교 가능

사용법:
    python visualize.py                     # results/dqn_normal.pt 사용
    python visualize.py --preset rush_hour
    python visualize.py --model results/dqn_normal.pt

학습된 모델이 없으면 먼저:  python train_dqn.py
"""

from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from crossway_environment import STAY, SWITCH, CrosswayEnvironment
from env_config import APPROACH_NAMES, APPROACHES, MOVEMENTS, PRESETS

# ── 화면 레이아웃 상수 ───────────────────────────────────────────────
CANVAS_W, CANVAS_H = 720, 720
CX, CY = CANVAS_W // 2, CANVAS_H // 2   # 교차로 중심
ROAD_HALF = 70                          # 도로 반쪽 폭 (중앙선 ~ 도로 가장자리)
LANE_LEFT = 16                          # 중앙선에서 좌회전 차로 중심까지
LANE_STRAIGHT = 44                      # 중앙선에서 직진 차로 중심까지
CAR_LEN, CAR_W = 26, 16
CAR_GAP = 6                             # 정차 중인 차량 간격
QUEUE_START = 14                        # 정지선에서 첫 차까지 거리 (신호등 자리)
MAX_DRAWN_CARS = 7                      # 한 차로에 그릴 최대 차량 수 (넘치면 숫자로 표시)
CAR_SPEED = 30                          # 교차로를 지나는 차의 이동 속도 (픽셀/프레임)
EXIT_DISTANCE = CANVAS_W * 0.75         # 이만큼 움직이면 화면 밖으로 나간 것으로 본다

BG = "#1e222a"
ROAD = "#3a3f4b"
LINE = "#c8cdd6"
GREEN, RED, YELLOW = "#3ddc84", "#e5484d", "#f5c518"
PANEL_BG = "#282c34"
TEXT = "#e6e9ef"
MUTED = "#9aa4b2"

# 페이즈별 차 색깔 (페이즈를 8개까지 늘려도 되도록 넉넉히)
PHASE_COLORS = ["#5aa9e6", "#f2a65a", "#7cd6c1", "#d38ce8",
                "#f28b82", "#a3d977", "#fdd663", "#8ab4f8"]
RIGHT_TURN_COLOR = "#9aa4b2"

# 진입 방향별 진행 방향 벡터 (화면 좌표: x는 오른쪽, y는 아래쪽이 +)
HEADING = {"N": (0, 1), "S": (0, -1), "E": (-1, 0), "W": (1, 0)}

PRESET_LABELS = {"night": "심야", "normal": "일반", "rush_hour": "러시아워", "ns_heavy": "남북 편중"}


def _right_of(h):
    """진행 방향 h의 오른쪽 방향 (우측통행이므로 차로는 이쪽에 있다)."""
    return (-h[1], h[0])


def _left_of(h):
    return (h[1], -h[0])


def _stop_point(approach, lane_offset):
    """해당 진입로·차로의 정지선 위치."""
    h = HEADING[approach]
    r = _right_of(h)
    return (CX - h[0] * ROAD_HALF + r[0] * lane_offset,
            CY - h[1] * ROAD_HALF + r[1] * lane_offset)


def _car_size(h):
    return (CAR_W, CAR_LEN) if h[0] == 0 else (CAR_LEN, CAR_W)


@dataclass
class MovingCar:
    """교차로를 통과 중인 차량 애니메이션."""

    approach: str           # 들어온 방향 N/S/E/W
    turn: str               # straight / left / right
    color: str
    distance: float = 0.0   # 정지선에서부터 움직인 거리(픽셀)

    def position(self):
        """현재 위치 (x, y)와 진행 방향을 계산한다. 회전 차량은 꺾이는 지점에서 방향을 바꾼다."""
        h = HEADING[self.approach]
        if self.turn == "straight":
            sx, sy = _stop_point(self.approach, LANE_STRAIGHT)
            return sx + h[0] * self.distance, sy + h[1] * self.distance, h

        if self.turn == "left":
            # 좌회전 차로에서 출발해, 나갈 도로의 바깥 차로 위치까지 간 뒤 왼쪽으로 꺾는다.
            sx, sy = _stop_point(self.approach, LANE_LEFT)
            turn_at, new_h = ROAD_HALF + LANE_STRAIGHT, _left_of(h)
        else:
            # 우회전은 직진 차로에서 출발해 바로 오른쪽으로 꺾는다.
            sx, sy = _stop_point(self.approach, LANE_STRAIGHT)
            turn_at, new_h = ROAD_HALF - LANE_STRAIGHT, _right_of(h)

        if self.distance < turn_at:
            return sx + h[0] * self.distance, sy + h[1] * self.distance, h
        x = sx + h[0] * turn_at + new_h[0] * (self.distance - turn_at)
        y = sy + h[1] * turn_at + new_h[1] * (self.distance - turn_at)
        return x, y, new_h


@dataclass
class RunResult:
    """한 번의 실행 설정과 결과. 실행 기록 표에 쌓인다."""

    index: int
    control: str
    demand_ns: float        # 남북 진입로 평균 교통량 (대/시간)
    demand_ew: float        # 동서 진입로 평균 교통량 (대/시간)
    duration: int
    avg_queue: float
    avg_wait: float
    throughput: float
    passed: int
    switches: int


class SignalControlApp:
    def __init__(self, root, model_path: Path | None, preset: str):
        self.root = root
        self.root.title("강화학습 기반 교차로 신호 제어 시뮬레이션")
        self.root.configure(bg=BG)

        self.env = CrosswayEnvironment(preset=preset, episode_seconds=None, seed=0)
        self.agent = self._load_agent(model_path)
        # 이동류마다 자기가 속한 (첫) 페이즈의 색을 쓴다.
        self.movement_color = {}
        for i, (_, movements) in enumerate(self.env.config.phases):
            for m in movements:
                self.movement_color.setdefault(m, PHASE_COLORS[i % len(PHASE_COLORS)])

        self.moving_cars: list[MovingCar] = []
        self.history: list[RunResult] = []
        self.finished = True     # 시작 전에는 '종료' 상태 = 설정 수정 가능
        self.tick_ms = 120

        # 이번 실행에 실제로 적용된 설정 (실행 중 슬라이더를 움직여도 영향받지 않도록 스냅샷)
        self.active: dict = {}
        self.step_count = 0
        self.queue_sum = 0
        self.switch_count = 0

        self._build_ui(preset)
        self._set_controls_enabled(True)
        self._render()
        self._loop()

    # ── 모델 로드 ──────────────────────────────────────────────────
    def _load_agent(self, model_path):
        if model_path is None or not Path(model_path).exists():
            print(f"[경고] 학습된 모델을 찾을 수 없습니다: {model_path}")
            print("       고정주기 신호로만 실행됩니다. 먼저 'python train_dqn.py'를 실행하세요.")
            return None

        from dqn_agent import DQNAgent

        try:
            agent = DQNAgent(self.env).load(model_path)
        except ValueError as e:
            print(f"[경고] {e}")
            print("       고정주기 신호로만 실행됩니다.")
            return None
        print(f"[정보] DQN 모델 로드 완료: {model_path}")
        return agent

    # ── UI 구성 ────────────────────────────────────────────────────
    def _build_ui(self, preset):
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)

        panel = tk.Frame(self.root, bg=PANEL_BG, padx=16, pady=10)
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.state_label = tk.Label(panel, text="", bg=PANEL_BG, fg=TEXT,
                                    font=("맑은 고딕", 13, "bold"))
        self.state_label.pack(anchor="w")

        self.status = tk.Label(panel, text="", bg=PANEL_BG, fg=TEXT, justify="left",
                               font=("Consolas", 10))
        self.status.pack(anchor="w", pady=(4, 6))

        tk.Frame(panel, bg="#3a3f4b", height=1).pack(fill=tk.X, pady=(0, 6))

        # ── 실험 조건: 실행 중에는 잠기고, 끝나면 수정 가능 ──
        tk.Label(panel, text="실험 조건 (실행 종료 후 수정 가능)", bg=PANEL_BG, fg=TEXT,
                 font=("맑은 고딕", 11, "bold")).pack(anchor="w")
        tk.Label(panel, text="진입로별 교통량 (대/시간)", bg=PANEL_BG, fg=MUTED,
                 font=("맑은 고딕", 9)).pack(anchor="w")

        # 4방향 교통량 슬라이더를 2×2로 배치
        grid = tk.Frame(panel, bg=PANEL_BG)
        grid.pack(anchor="w")
        self.demand = {}
        self.locked_sliders = []
        for i, approach in enumerate(APPROACHES):
            cell = tk.Frame(grid, bg=PANEL_BG)
            cell.grid(row=i // 2, column=i % 2, padx=(0, 8))
            var, scale = self._add_slider(cell, f"{APPROACH_NAMES[approach]}에서 진입",
                                          0, 1800, 50, self.env.arrival_per_hour[approach],
                                          length=130)
            self.demand[approach] = var
            self.locked_sliders.append(scale)

        # 프리셋 버튼 (실험 조건이므로 함께 잠근다)
        preset_row = tk.Frame(panel, bg=PANEL_BG)
        preset_row.pack(anchor="w", pady=(2, 4))
        self.preset_buttons = []
        for name in PRESETS:
            b = tk.Button(preset_row, text=PRESET_LABELS.get(name, name), relief="flat",
                          bg="#3a3f4b", fg=TEXT, activebackground="#4a5060", padx=6,
                          command=lambda n=name: self._apply_preset(n))
            b.pack(side=tk.LEFT, padx=2)
            self.preset_buttons.append(b)

        cfg = self.env.config
        self.pass_rate, s1 = self._add_slider(panel, "통과 속도 (차로당 대/초)",
                                              0.1, 3.0, 0.1, cfg.pass_rate)
        self.hold_time, s2 = self._add_slider(
            panel, f"고정주기 유지시간 (초, {cfg.min_green}~{cfg.max_green})",
            cfg.min_green, cfg.max_green, 1, cfg.min_green)
        self.duration, s3 = self._add_slider(panel, "실행 시간 (초)", 60, 1800, 60, 600)
        self.locked_sliders += [s1, s2, s3]

        self.use_dqn = tk.BooleanVar(value=self.agent is not None)
        self.dqn_check = tk.Checkbutton(
            panel, text="DQN 제어 (끄면 고정주기 신호)", variable=self.use_dqn,
            bg=PANEL_BG, fg=TEXT, selectcolor="#3a3f4b", activebackground=PANEL_BG,
            activeforeground=TEXT, font=("맑은 고딕", 10))
        self.dqn_check.pack(anchor="w")

        # ── 실행 제어 ──
        btn_row = tk.Frame(panel, bg=PANEL_BG)
        btn_row.pack(anchor="w", pady=(6, 4))
        self.start_btn = tk.Button(btn_row, text="설정 적용하고 시작", width=18, relief="flat",
                                   bg=GREEN, fg="#11141a", font=("맑은 고딕", 10, "bold"),
                                   command=self._start_run)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn = tk.Button(btn_row, text="중단", width=7, relief="flat",
                                  bg="#3a3f4b", fg=TEXT, command=self._stop_run)
        self.stop_btn.pack(side=tk.LEFT)

        # 재생 속도는 실험 조건이 아니라 '보기' 설정이므로 실행 중에도 조절 가능.
        self.speed, _ = self._add_slider(panel, "재생 속도 (ms/step) — 실행 중 조절 가능",
                                         10, 500, 10, self.tick_ms)

        # ── 실행 기록 ──
        tk.Frame(panel, bg="#3a3f4b", height=1).pack(fill=tk.X, pady=(4, 4))
        tk.Label(panel, text="실행 기록 (교통량은 진입로당 대/시간)", bg=PANEL_BG, fg=TEXT,
                 font=("맑은 고딕", 10, "bold")).pack(anchor="w")
        self.history_box = tk.Text(panel, height=9, width=50, bg="#1e222a", fg=TEXT,
                                   font=("Consolas", 9), relief="flat", state="disabled")
        self.history_box.pack(anchor="w", pady=(4, 0))
        self._refresh_history()

    def _add_slider(self, parent, label, lo, hi, step, initial, length=250):
        tk.Label(parent, text=label, bg=PANEL_BG, fg=MUTED,
                 font=("맑은 고딕", 9)).pack(anchor="w")
        var = tk.DoubleVar(value=initial)
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient=tk.HORIZONTAL,
                         variable=var, length=length, bg=PANEL_BG, fg=TEXT,
                         troughcolor="#1e222a", highlightthickness=0, activebackground=GREEN)
        scale.pack(anchor="w", pady=(0, 1))
        return var, scale

    def _set_controls_enabled(self, enabled: bool):
        """실험 조건 위젯을 잠그거나 푼다."""
        state = "normal" if enabled else "disabled"
        for widget in self.locked_sliders + self.preset_buttons + [self.dqn_check]:
            widget.config(state=state)
        self.start_btn.config(state=state,
                              text="설정 적용하고 시작" if not self.history else "설정 적용하고 재실행")
        self.stop_btn.config(state="disabled" if enabled else "normal")

    # ── 설정 프리셋 ────────────────────────────────────────────────
    def _apply_preset(self, name):
        for approach, value in PRESETS[name].items():
            self.demand[approach].set(value)

    # ── 실행 시작 / 종료 ───────────────────────────────────────────
    def _start_run(self):
        # 슬라이더 값을 이 시점에 한 번만 읽어 고정한다. 실행 중에는 바뀌지 않는다.
        self.active = {
            "demand": {a: self.demand[a].get() for a in APPROACHES},
            "pass_rate": self.pass_rate.get(),
            "hold_time": int(self.hold_time.get()),
            "duration": int(self.duration.get()),
            "use_dqn": bool(self.use_dqn.get() and self.agent is not None),
        }

        self.env.reset()
        self.env.arrival_per_hour = dict(self.active["demand"])
        self.env.pass_rate = self.active["pass_rate"]

        self.moving_cars.clear()
        self.step_count = 0
        self.queue_sum = 0
        self.switch_count = 0
        self.finished = False
        self._set_controls_enabled(False)

    def _stop_run(self):
        if not self.finished:
            self._finish_run(interrupted=True)

    def _finish_run(self, interrupted=False):
        self.finished = True
        self._set_controls_enabled(True)

        if self.step_count == 0:
            return

        demand = self.active["demand"]
        self.history.append(RunResult(
            index=len(self.history) + 1,
            control=("DQN" if self.active["use_dqn"] else f"고정{self.active['hold_time']}초")
                    + ("*" if interrupted else ""),
            demand_ns=(demand["N"] + demand["S"]) / 2,
            demand_ew=(demand["E"] + demand["W"]) / 2,
            duration=self.step_count,
            avg_queue=self.queue_sum / self.step_count,
            avg_wait=self.env.avg_wait,
            throughput=self.env.total_passed / self.step_count * 60,
            passed=self.env.total_passed,
            switches=self.switch_count,
        ))
        self._refresh_history()

    def _refresh_history(self):
        self.history_box.config(state="normal")
        self.history_box.delete("1.0", tk.END)

        header = (f"{'#':>2} {'제어':<8}{'남북':>6}{'동서':>6}"
                  f"{'대기열':>7}{'대기(초)':>9}{'전환':>5}\n")
        self.history_box.insert(tk.END, header)
        self.history_box.insert(tk.END, "-" * 50 + "\n")

        if not self.history:
            self.history_box.insert(tk.END, "\n  아직 실행 기록이 없습니다.\n"
                                            "  설정을 정하고 '시작'을 누르세요.\n")
        for r in self.history[-6:]:
            self.history_box.insert(
                tk.END,
                f"{r.index:>2} {r.control:<8}{r.demand_ns:>6.0f}{r.demand_ew:>6.0f}"
                f"{r.avg_queue:>8.2f}{r.avg_wait:>9.1f}{r.switches:>5d}\n")

        if any(r.control.endswith("*") for r in self.history):
            self.history_box.insert(tk.END, "* 중간에 중단된 실행\n")

        self.history_box.config(state="disabled")

    # ── 시뮬레이션 한 스텝 ─────────────────────────────────────────
    def _select_action(self, state):
        if self.active.get("use_dqn"):
            return self.agent.select_action(state, greedy=True)
        # 고정주기 신호: 정해진 시간마다 기계적으로 전환
        return SWITCH if self.env.phase_time >= self.active["hold_time"] else STAY

    def _sim_step(self):
        _, _, _, info = self.env.step(self._select_action(self.env.get_state()))

        # 통과한 차량을 애니메이션 대상으로 등록
        for movement, _wait in info.departures:
            approach, turn = movement.split("_")
            self.moving_cars.append(MovingCar(approach, turn, self.movement_color[movement]))
        for approach in info.right_turns:
            self.moving_cars.append(MovingCar(approach, "right", RIGHT_TURN_COLOR))

        self.step_count += 1
        self.queue_sum += info.total_queue
        self.switch_count += info.switched

        if self.step_count >= self.active["duration"]:
            self._finish_run()

    def _advance_animation(self):
        for car in self.moving_cars:
            car.distance += CAR_SPEED
        self.moving_cars = [c for c in self.moving_cars if c.distance < EXIT_DISTANCE]

    # ── 렌더링 ─────────────────────────────────────────────────────
    def _draw_roads(self):
        c = self.canvas
        c.create_rectangle(0, CY - ROAD_HALF, CANVAS_W, CY + ROAD_HALF, fill=ROAD, width=0)
        c.create_rectangle(CX - ROAD_HALF, 0, CX + ROAD_HALF, CANVAS_H, fill=ROAD, width=0)

        # 중앙선 (교차로 내부는 비움)
        for start, end in [(0, CX - ROAD_HALF), (CX + ROAD_HALF, CANVAS_W)]:
            c.create_line(start, CY, end, CY, fill=YELLOW, width=2)
        for start, end in [(0, CY - ROAD_HALF), (CY + ROAD_HALF, CANVAS_H)]:
            c.create_line(CX, start, CX, end, fill=YELLOW, width=2)

        # 진입로마다 좌회전/직진 차로 경계선과 정지선
        lane_divider = (LANE_LEFT + LANE_STRAIGHT) / 2
        for approach in APPROACHES:
            h = HEADING[approach]
            r = _right_of(h)
            # 정지선: 중앙선부터 도로 가장자리까지
            x0 = CX - h[0] * ROAD_HALF
            y0 = CY - h[1] * ROAD_HALF
            c.create_line(x0, y0, x0 + r[0] * ROAD_HALF, y0 + r[1] * ROAD_HALF,
                          fill=LINE, width=3)
            # 차로 경계선: 정지선에서 화면 끝까지
            dx, dy = x0 + r[0] * lane_divider, y0 + r[1] * lane_divider
            c.create_line(dx, dy, dx - h[0] * CANVAS_W, dy - h[1] * CANVAS_H,
                          fill=LINE, dash=(8, 8))

    def _queue_pos(self, movement, index):
        """대기열에서 index번째 차의 위치와 크기."""
        approach, turn = movement.split("_")
        h = HEADING[approach]
        sx, sy = _stop_point(approach, LANE_LEFT if turn == "left" else LANE_STRAIGHT)
        back = QUEUE_START + CAR_LEN / 2 + index * (CAR_LEN + CAR_GAP)
        return sx - h[0] * back, sy - h[1] * back, h

    def _draw_queues(self):
        """대기 중인 차량을 정지선 뒤로 줄 세워 그린다."""
        for movement, count in self.env.queue_snapshot().items():
            color = self.movement_color[movement]
            for i in range(min(count, MAX_DRAWN_CARS)):
                x, y, h = self._queue_pos(movement, i)
                self._car(x, y, h, color)

            # 화면에 다 못 그리는 차는 숫자로 표기
            if count > MAX_DRAWN_CARS:
                x, y, _ = self._queue_pos(movement, MAX_DRAWN_CARS)
                self.canvas.create_text(x, y, text=f"+{count - MAX_DRAWN_CARS}",
                                        fill=color, font=("Consolas", 10, "bold"))

    def _draw_moving_cars(self):
        for car in self.moving_cars:
            x, y, h = car.position()
            self._car(x, y, h, car.color)

    def _car(self, x, y, heading, color):
        w, h = _car_size(heading)
        self.canvas.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                                     fill=color, outline="#11141a", width=1)

    def _draw_signals(self):
        """차로마다 정지선 위에 신호등을 그린다."""
        green = set(self.env.green_movements())
        # 최소 유지시간을 못 채워 전환이 막힌 상태는 노란색으로 표시
        locked = self.env.phase_time < self.env.min_green
        for movement in MOVEMENTS:
            approach, turn = movement.split("_")
            x, y = _stop_point(approach, LANE_LEFT if turn == "left" else LANE_STRAIGHT)
            h = HEADING[approach]
            x, y = x - h[0] * 6, y - h[1] * 6
            if movement in green:
                color = YELLOW if locked else GREEN
            else:
                color = RED
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="#11141a")

    def _draw_hud(self):
        c = self.canvas
        c.create_text(16, 18, anchor="w", fill=TEXT, font=("맑은 고딕", 12, "bold"),
                      text=f"신호: {self.env.phase_name()} ({self.env.phase_time}초)")

        if self.finished:
            label, color = "대기 중 — 설정 후 시작", YELLOW
        else:
            mode = "DQN 제어" if self.active["use_dqn"] else f"고정주기 {self.active['hold_time']}초"
            label, color = f"실행 중 · {mode}", GREEN
        c.create_text(16, 42, anchor="w", fill=color, font=("맑은 고딕", 10), text=label)

        # 진행률 바
        if not self.finished:
            ratio = min(self.step_count / self.active["duration"], 1.0)
            c.create_rectangle(16, 58, 266, 66, outline="#3a3f4b")
            c.create_rectangle(16, 58, 16 + 250 * ratio, 66, fill=GREEN, width=0)

        # 범례: 페이즈별 색 + 우회전
        legend = [(name, PHASE_COLORS[i % len(PHASE_COLORS)])
                  for i, (name, _) in enumerate(self.env.config.phases)]
        legend.append(("우회전 (상시 통과)", RIGHT_TURN_COLOR))
        top = CANVAS_H - 16 - 18 * (len(legend) - 1)
        for i, (name, color) in enumerate(legend):
            y = top + i * 18
            c.create_rectangle(16, y - 6, 30, y + 6, fill=color, outline="")
            c.create_text(38, y, anchor="w", fill=TEXT, text=name, font=("맑은 고딕", 9))

    def _render(self):
        self.canvas.delete("all")
        self._draw_roads()
        self._draw_queues()
        self._draw_moving_cars()
        self._draw_signals()
        self._draw_hud()

        if self.finished:
            self.state_label.config(text="⏸  대기 중 (설정 수정 가능)", fg=YELLOW)
        else:
            remain = self.active["duration"] - self.step_count
            self.state_label.config(text=f"▶  실행 중 — {remain}초 남음", fg=GREEN)

        avg_queue = self.queue_sum / self.step_count if self.step_count else 0
        throughput = self.env.total_passed / self.step_count * 60 if self.step_count else 0
        self.status.config(
            text=(f"경과 시간    {self.step_count:5d} 초\n"
                  f"통과 차량    {self.env.total_passed:5d} 대  ({throughput:.1f} 대/분)\n"
                  f"현재 대기열  {self.env.total_queue:5d} 대  "
                  f"(남북 {self.env.ns_queue}, 동서 {self.env.ew_queue})\n"
                  f"평균 대기열  {avg_queue:7.2f} 대\n"
                  f"평균 대기    {self.env.avg_wait:7.1f} 초\n"
                  f"신호 전환    {self.switch_count:5d} 회"))

    # ── 메인 루프 ──────────────────────────────────────────────────
    def _loop(self):
        self.tick_ms = int(self.speed.get())
        if not self.finished:
            self._sim_step()
        self._advance_animation()
        self._render()
        self.root.after(self.tick_ms, self._loop)


def main():
    parser = argparse.ArgumentParser(description="DQN 교차로 신호 제어 시각화")
    parser.add_argument("--preset", default="normal", choices=list(PRESETS))
    parser.add_argument("--model", default=None, help="학습된 모델 경로 (.pt)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else \
        Path(__file__).parent / "results" / f"dqn_{args.preset}.pt"

    root = tk.Tk()
    SignalControlApp(root, model_path, args.preset)
    root.mainloop()


if __name__ == "__main__":
    main()
