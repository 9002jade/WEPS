"""DQN 신호 제어 시각화 (tkinter).

한 번의 '실행(run)'은 고정된 설정으로 정해진 시간만큼 시뮬레이션을 돌린다.
실행 중에는 실험 조건을 바꿀 수 없고(그래야 평균 지표가 하나의 조건을 대표한다),
실행이 끝나면 설정을 수정해 새 실행을 시작할 수 있다.
지난 실행 결과는 아래 '실행 기록'에 남아 조건별 비교에 쓴다.

- 차량이 대기줄에 서고, 초록불에 교차로를 통과해 빠져나가는 모습까지 렌더링
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

from crossway_environment import (
    MAX_GREEN_TIME,
    MIN_GREEN_TIME,
    MOVEMENT_BY_PHASE,
    PRESETS,
    STAY,
    SWITCH,
    CrosswayEnvironment,
)

# ── 화면 레이아웃 상수 ───────────────────────────────────────────────
CANVAS_W, CANVAS_H = 720, 720
CX, CY = CANVAS_W // 2, CANVAS_H // 2   # 교차로 중심
ROAD_HALF = 70                          # 교차로 반쪽 폭
LANE_OFFSET = 34                        # 중앙선에서 차로 중심까지 거리
CAR_LEN, CAR_W = 26, 16
CAR_GAP = 6                             # 정차 중인 차량 간격
MAX_DRAWN_CARS = 12                     # 한 대기열에 그릴 최대 차량 수

BG = "#1e222a"
ROAD = "#3a3f4b"
LINE = "#c8cdd6"
GREEN, RED, YELLOW = "#3ddc84", "#e5484d", "#f5c518"
PANEL_BG = "#282c34"
TEXT = "#e6e9ef"
MUTED = "#9aa4b2"

# 이동류별 색상 (직진/좌회전 구분)
CAR_COLORS = {
    "ns_straight": "#5aa9e6",
    "ns_left": "#7cd6c1",
    "ew_straight": "#f2a65a",
    "ew_left": "#d38ce8",
}
RIGHT_TURN_COLOR = "#9aa4b2"


@dataclass
class MovingCar:
    """교차로를 통과 중인 차량 애니메이션."""

    movement: str
    progress: float = 0.0   # 0.0(정지선) -> 1.0(화면 밖)
    color: str = "#ffffff"


@dataclass
class RunResult:
    """한 번의 실행 설정과 결과. 실행 기록 표에 쌓인다."""

    index: int
    control: str
    arrival_ns: float
    arrival_ew: float
    duration: int
    avg_queue: float
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

        agent = DQNAgent()
        agent.load(model_path)
        print(f"[정보] DQN 모델 로드 완료: {model_path}")
        return agent

    # ── UI 구성 ────────────────────────────────────────────────────
    def _build_ui(self, preset):
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)

        panel = tk.Frame(self.root, bg=PANEL_BG, padx=16, pady=12)
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.state_label = tk.Label(panel, text="", bg=PANEL_BG, fg=TEXT,
                                    font=("맑은 고딕", 13, "bold"))
        self.state_label.pack(anchor="w")

        self.status = tk.Label(panel, text="", bg=PANEL_BG, fg=TEXT, justify="left",
                               font=("Consolas", 10))
        self.status.pack(anchor="w", pady=(6, 10))

        tk.Frame(panel, bg="#3a3f4b", height=1).pack(fill=tk.X, pady=(0, 8))

        # ── 실험 조건: 실행 중에는 잠기고, 끝나면 수정 가능 ──
        tk.Label(panel, text="실험 조건 (실행 종료 후 수정 가능)", bg=PANEL_BG, fg=TEXT,
                 font=("맑은 고딕", 11, "bold")).pack(anchor="w")

        base = PRESETS[preset]["arrival_prob"]
        self.arrival_ns, s1 = self._add_slider(panel, "남북 교통량", 0.0, 1.0, 0.05, base)
        self.arrival_ew, s2 = self._add_slider(panel, "동서 교통량", 0.0, 1.0, 0.05, base)
        self.pass_rate, s3 = self._add_slider(panel, "통과 속도 (대/초)", 0.1, 3.0, 0.1, 0.5)
        self.hold_time, s4 = self._add_slider(
            panel, f"고정주기 유지시간 (초, {MIN_GREEN_TIME}~{MAX_GREEN_TIME})",
            MIN_GREEN_TIME, MAX_GREEN_TIME, 1, MIN_GREEN_TIME)
        self.duration, s5 = self._add_slider(panel, "실행 시간 (초)", 60, 1800, 60, 600)
        self.locked_sliders = [s1, s2, s3, s4, s5]

        # 프리셋 버튼 (실험 조건이므로 함께 잠근다)
        preset_row = tk.Frame(panel, bg=PANEL_BG)
        preset_row.pack(anchor="w", pady=(2, 4))
        self.preset_buttons = []
        for name, label in [("night", "심야"), ("normal", "일반"), ("rush_hour", "러시아워")]:
            b = tk.Button(preset_row, text=label, width=6, relief="flat",
                          bg="#3a3f4b", fg=TEXT, activebackground="#4a5060",
                          command=lambda n=name: self._apply_preset(n))
            b.pack(side=tk.LEFT, padx=2)
            self.preset_buttons.append(b)

        skew_btn = tk.Button(panel, text="남북 편중 (비대칭 교통량)", relief="flat", width=24,
                             bg="#4a4050", fg=TEXT, activebackground="#5a5060",
                             command=self._apply_skew)
        skew_btn.pack(anchor="w", pady=(0, 6), padx=2)
        self.preset_buttons.append(skew_btn)

        self.use_dqn = tk.BooleanVar(value=self.agent is not None)
        self.dqn_check = tk.Checkbutton(
            panel, text="DQN 제어 (끄면 고정주기 신호)", variable=self.use_dqn,
            bg=PANEL_BG, fg=TEXT, selectcolor="#3a3f4b", activebackground=PANEL_BG,
            activeforeground=TEXT, font=("맑은 고딕", 10))
        self.dqn_check.pack(anchor="w")

        # ── 실행 제어 ──
        btn_row = tk.Frame(panel, bg=PANEL_BG)
        btn_row.pack(anchor="w", pady=(8, 6))
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
        tk.Frame(panel, bg="#3a3f4b", height=1).pack(fill=tk.X, pady=(6, 6))
        tk.Label(panel, text="실행 기록", bg=PANEL_BG, fg=TEXT,
                 font=("맑은 고딕", 11, "bold")).pack(anchor="w")
        self.history_box = tk.Text(panel, height=11, width=44, bg="#1e222a", fg=TEXT,
                                   font=("Consolas", 9), relief="flat", state="disabled")
        self.history_box.pack(anchor="w", pady=(4, 0))
        self._refresh_history()

    def _add_slider(self, parent, label, lo, hi, step, initial):
        tk.Label(parent, text=label, bg=PANEL_BG, fg=MUTED,
                 font=("맑은 고딕", 9)).pack(anchor="w")
        var = tk.DoubleVar(value=initial)
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient=tk.HORIZONTAL,
                         variable=var, length=250, bg=PANEL_BG, fg=TEXT,
                         troughcolor="#1e222a", highlightthickness=0, activebackground=GREEN)
        scale.pack(anchor="w", pady=(0, 2))
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
        self.arrival_ns.set(PRESETS[name]["arrival_prob"])
        self.arrival_ew.set(PRESETS[name]["arrival_prob"])

    def _apply_skew(self):
        """한쪽 방향에만 교통량을 몰아준다 — 적응형 제어의 차이가 드러나는 상황."""
        self.arrival_ns.set(0.55)
        self.arrival_ew.set(0.10)

    # ── 실행 시작 / 종료 ───────────────────────────────────────────
    def _start_run(self):
        # 슬라이더 값을 이 시점에 한 번만 읽어 고정한다. 실행 중에는 바뀌지 않는다.
        self.active = {
            "arrival_ns": self.arrival_ns.get(),
            "arrival_ew": self.arrival_ew.get(),
            "pass_rate": self.pass_rate.get(),
            "hold_time": int(self.hold_time.get()),
            "duration": int(self.duration.get()),
            "use_dqn": bool(self.use_dqn.get() and self.agent is not None),
        }

        self.env.reset()
        self.env.arrival_prob_ns = self.active["arrival_ns"]
        self.env.arrival_prob_ew = self.active["arrival_ew"]
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

        self.history.append(RunResult(
            index=len(self.history) + 1,
            control=("DQN" if self.active["use_dqn"] else f"고정{self.active['hold_time']}초")
                    + ("*" if interrupted else ""),
            arrival_ns=self.active["arrival_ns"],
            arrival_ew=self.active["arrival_ew"],
            duration=self.step_count,
            avg_queue=self.queue_sum / self.step_count,
            throughput=self.env.total_passed / self.step_count * 60,
            passed=self.env.total_passed,
            switches=self.switch_count,
        ))
        self._refresh_history()

    def _refresh_history(self):
        self.history_box.config(state="normal")
        self.history_box.delete("1.0", tk.END)

        header = f"{'#':>2} {'제어':<9}{'남북':>5}{'동서':>5}{'평균대기':>8}{'대/분':>7}{'전환':>5}\n"
        self.history_box.insert(tk.END, header)
        self.history_box.insert(tk.END, "-" * 44 + "\n")

        if not self.history:
            self.history_box.insert(tk.END, "\n  아직 실행 기록이 없습니다.\n"
                                            "  설정을 정하고 '시작'을 누르세요.\n")
        for r in self.history[-9:]:
            self.history_box.insert(
                tk.END,
                f"{r.index:>2} {r.control:<9}{r.arrival_ns:>5.2f}{r.arrival_ew:>5.2f}"
                f"{r.avg_queue:>8.2f}{r.throughput:>7.1f}{r.switches:>5d}\n")

        if any(r.control.endswith("*") for r in self.history):
            self.history_box.insert(tk.END, "\n* 중간에 중단된 실행\n")

        self.history_box.config(state="disabled")

    # ── 시뮬레이션 한 스텝 ─────────────────────────────────────────
    def _select_action(self, state):
        if self.active.get("use_dqn"):
            return self.agent.select_action(state, greedy=True)
        # 고정주기 신호: 정해진 시간마다 기계적으로 전환
        return SWITCH if self.env.phase_time >= self.active["hold_time"] else STAY

    def _sim_step(self):
        state = (self.env.ns_queue, self.env.ew_queue, self.env.phase, self.env.phase_time)
        _, _, _, info = self.env.step(self._select_action(state))

        # 통과한 차량을 애니메이션 대상으로 등록
        movement = MOVEMENT_BY_PHASE[self.env.phase]
        for _ in range(info.passed):
            self.moving_cars.append(MovingCar(movement, color=CAR_COLORS[movement]))
        for _ in range(info.right_passed):
            self.moving_cars.append(
                MovingCar(self.env.rng.choice(list(CAR_COLORS)), color=RIGHT_TURN_COLOR))

        self.step_count += 1
        self.queue_sum += info.total_queue
        self.switch_count += info.switched

        if self.step_count >= self.active["duration"]:
            self._finish_run()

    def _advance_animation(self):
        for car in self.moving_cars:
            car.progress += 0.14
        self.moving_cars = [c for c in self.moving_cars if c.progress < 1.0]

    # ── 렌더링 ─────────────────────────────────────────────────────
    def _draw_roads(self):
        c = self.canvas
        c.create_rectangle(0, CY - ROAD_HALF, CANVAS_W, CY + ROAD_HALF, fill=ROAD, width=0)
        c.create_rectangle(CX - ROAD_HALF, 0, CX + ROAD_HALF, CANVAS_H, fill=ROAD, width=0)

        # 중앙선 (교차로 내부는 비움)
        for start, end in [(0, CX - ROAD_HALF), (CX + ROAD_HALF, CANVAS_W)]:
            c.create_line(start, CY, end, CY, fill=LINE, dash=(12, 10))
        for start, end in [(0, CY - ROAD_HALF), (CY + ROAD_HALF, CANVAS_H)]:
            c.create_line(CX, start, CX, end, fill=LINE, dash=(12, 10))

        # 정지선
        c.create_line(CX - ROAD_HALF, CY + ROAD_HALF, CX, CY + ROAD_HALF, fill=LINE, width=3)
        c.create_line(CX, CY - ROAD_HALF, CX + ROAD_HALF, CY - ROAD_HALF, fill=LINE, width=3)
        c.create_line(CX - ROAD_HALF, CY - ROAD_HALF, CX - ROAD_HALF, CY, fill=LINE, width=3)
        c.create_line(CX + ROAD_HALF, CY, CX + ROAD_HALF, CY + ROAD_HALF, fill=LINE, width=3)

    @staticmethod
    def _lane_pos(movement, back):
        """정지선에서 `back` 픽셀 뒤에 있는 차량의 (x, y, 가로, 세로)를 계산한다."""
        lane = LANE_OFFSET if movement.endswith("left") else LANE_OFFSET // 3
        if movement.startswith("ns"):
            # 남쪽에서 북쪽으로 진입 (화면 아래 -> 위)
            return CX - lane, CY + back, CAR_W, CAR_LEN
        # 동쪽에서 서쪽으로 진입 (화면 오른쪽 -> 왼쪽)
        return CX + back, CY - lane, CAR_LEN, CAR_W

    def _draw_queues(self):
        """대기 중인 차량을 정지선 뒤로 줄 세워 그린다."""
        for movement, count in self.env.queue_snapshot().items():
            color = CAR_COLORS[movement]
            for i in range(min(int(count), MAX_DRAWN_CARS)):
                x, y, w, h = self._lane_pos(movement, ROAD_HALF + 6 + i * (CAR_LEN + CAR_GAP))
                self._car(x, y, w, h, color)

            # 화면 밖으로 넘치는 대기 행렬은 숫자로 표기
            if count > MAX_DRAWN_CARS:
                far = ROAD_HALF + 10 + MAX_DRAWN_CARS * (CAR_LEN + CAR_GAP)
                x, y, _, _ = self._lane_pos(movement, far)
                self.canvas.create_text(x, y, text=f"+{int(count) - MAX_DRAWN_CARS}",
                                        fill=color, font=("Consolas", 11, "bold"))

    def _draw_moving_cars(self):
        span = CANVAS_W // 2 + 60
        for car in self.moving_cars:
            p = car.progress
            if car.movement == "ns_straight":
                self._car(CX - LANE_OFFSET // 3, CY + ROAD_HALF - p * span,
                          CAR_W, CAR_LEN, car.color)
            elif car.movement == "ns_left":
                # 좌회전: 위로 올라가다 서쪽으로 꺾임
                x = CX - LANE_OFFSET - max(0.0, p - 0.45) * span
                y = CY + ROAD_HALF - min(p, 0.45) * span
                turned = p > 0.45
                self._car(x, y, CAR_LEN if turned else CAR_W,
                          CAR_W if turned else CAR_LEN, car.color)
            elif car.movement == "ew_straight":
                self._car(CX + ROAD_HALF - p * span, CY - LANE_OFFSET // 3,
                          CAR_LEN, CAR_W, car.color)
            else:  # ew_left
                x = CX + ROAD_HALF - min(p, 0.45) * span
                y = CY - LANE_OFFSET + max(0.0, p - 0.45) * span
                turned = p > 0.45
                self._car(x, y, CAR_W if turned else CAR_LEN,
                          CAR_LEN if turned else CAR_W, car.color)

    def _car(self, x, y, w, h, color):
        self.canvas.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                                     fill=color, outline="#11141a", width=1)

    def _draw_signals(self):
        """각 접근로 정지선 옆에 신호등을 그린다."""
        green = self.env.green_movement()
        specs = [
            ("ns_straight", CX - LANE_OFFSET // 3 + 26, CY + ROAD_HALF + 20),
            ("ns_left", CX - LANE_OFFSET - 26, CY + ROAD_HALF + 20),
            ("ew_straight", CX + ROAD_HALF + 20, CY - LANE_OFFSET // 3 + 26),
            ("ew_left", CX + ROAD_HALF + 20, CY - LANE_OFFSET - 26),
        ]
        for movement, x, y in specs:
            on = movement == green
            # 최소 유지시간을 못 채워 전환이 막힌 상태는 노란색으로 표시
            if on and self.env.phase_time < MIN_GREEN_TIME:
                color = YELLOW
            else:
                color = GREEN if on else RED
            self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=color, outline="#11141a")

    def _draw_hud(self):
        c = self.canvas
        c.create_text(16, 18, anchor="w", fill=TEXT, font=("맑은 고딕", 13, "bold"),
                      text=f"현재 신호: {self.env.phase_name()}  ({self.env.phase_time}초 유지)")

        if self.finished:
            label, color = "대기 중 — 설정을 수정하고 시작하세요", YELLOW
        else:
            mode = "DQN 제어" if self.active["use_dqn"] else f"고정주기 {self.active['hold_time']}초"
            label, color = f"실행 중 · {mode}", GREEN
        c.create_text(16, 42, anchor="w", fill=color, font=("맑은 고딕", 11), text=label)

        # 진행률 바
        if not self.finished:
            ratio = min(self.step_count / self.active["duration"], 1.0)
            c.create_rectangle(16, 58, 316, 66, outline="#3a3f4b")
            c.create_rectangle(16, 58, 16 + 300 * ratio, 66, fill=GREEN, width=0)

        # 범례
        legend = [("남북 직진", CAR_COLORS["ns_straight"]), ("남북 좌회전", CAR_COLORS["ns_left"]),
                  ("동서 직진", CAR_COLORS["ew_straight"]), ("동서 좌회전", CAR_COLORS["ew_left"])]
        for i, (name, color) in enumerate(legend):
            y = CANVAS_H - 74 + i * 18
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
                  f"통과 차량    {self.env.total_passed:5d} 대\n"
                  f"분당 처리량  {throughput:7.1f} 대/분\n"
                  f"현재 대기열  {self.env.total_queue:5d} 대  "
                  f"(남북 {self.env.ns_queue}, 동서 {self.env.ew_queue})\n"
                  f"평균 대기열  {avg_queue:7.2f} 대\n"
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
