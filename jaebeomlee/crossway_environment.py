"""단일 교차로 신호 제어 환경 (팀 공용 모듈) — ★수정본★

⚠️ 저장소 루트에 있는 확장자 없는 구버전 `crossway_environment`는 더 이상 쓰지 말 것.
   구버전은 보상 함수가 상쇄되어 학습이 안 되고, 방향별 교통량을 나눌 수 없으며,
   주기 상한(120초)도 구현되어 있지 않다. 코드를 고칠 때는 반드시 이 폴더의 파일을 고칠 것.

설정값(교통량, 신호 시간, 보상 등)은 전부 env_config.py에 있다. 실험 조건을 바꿀 때는
이 파일이 아니라 env_config.py를 보면 된다.

인터페이스 (Gym 스타일):
    env = CrosswayEnvironment(preset="normal", seed=0)
    state = env.reset()
    state, reward, done, info = env.step(action)

state (state_mode="movement", 기본, 10개 값):
    (북직진, 북좌회전, 남직진, 남좌회전, 동직진, 동좌회전, 서직진, 서좌회전, 페이즈, 유지시간)
    앞의 8개는 각 이동류에 줄 서 있는 차량 수. 순서는 env_config.MOVEMENTS와 같다.
state (state_mode="simple", 4개 값):
    (남북 대기 합계, 동서 대기 합계, 페이즈, 유지시간)

action:
    0 = STAY   (현재 페이즈 유지)
    1 = SWITCH (다음 페이즈로 전환. 최소 초록시간 전이면 무시됨)

신경망에 넣을 때는 env.normalize_state(state)로 0~1 근처 값으로 바꿔서 쓴다.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, replace

from env_config import (
    APPROACHES,
    MOVEMENTS,
    PRESETS,
    EnvConfig,
    make_config,
    movement_name,
)

STAY, SWITCH = 0, 1
N_ACTIONS = 2

# 신경망 입력 정규화 기준. 대기열을 이 값으로 나눈다.
QUEUE_NORMALIZER = 20.0


@dataclass
class StepInfo:
    """step() 한 번의 결과 상세. 성능 지표 집계와 시각화에 사용."""

    switched: bool                     # 이번 step에 신호가 전환됐는지
    passed: int                        # 이번 step에 초록불로 통과한 차량 수
    right_passed: int                  # 이번 step에 우회전으로 바로 통과한 차량 수
    total_passed: int                  # 에피소드 누적 통과 차량 수 (우회전 포함)
    total_queue: int                   # 현재 총 대기 차량 수
    ns_queue: int                      # 남북 방향 대기 합계
    ew_queue: int                      # 동서 방향 대기 합계
    queues: dict = field(default_factory=dict)        # 이동류별 대기 차량 수
    departures: list = field(default_factory=list)    # 이번 step에 통과한 차 [(이동류, 대기시간초)]
    right_turns: list = field(default_factory=list)   # 이번 step에 우회전한 차의 진입 방향


class CrosswayEnvironment:
    """단일 4방향 교차로 신호 제어 환경 (4방향 × 직진·좌회전 = 8개 대기열)."""

    def __init__(self, config: EnvConfig | None = None, preset: str | None = None,
                 seed: int | None = None, **overrides):
        """
        config  : EnvConfig를 직접 넘길 때
        preset  : "night" / "normal" / "rush_hour" / "ns_heavy" 중 하나
        overrides : 설정 일부만 바꿀 때. 예) min_green=10, reward_mode="delta"
        """
        if config is None:
            config = make_config(preset or "normal", **overrides)
        elif overrides:
            config = replace(config, **overrides)
        config.validate()

        self.config = config
        self.preset = preset or "custom"
        self.rng = random.Random(seed)

        # 아래 세 값은 실행 중에도 바꿀 수 있다 (시각화에서 슬라이더로 조절).
        self.arrival_per_hour = dict(config.arrival_per_hour)
        self.pass_rate = config.pass_rate
        self.switch_penalty = config.switch_penalty

        self.reset()

    # ── 기본 정보 ───────────────────────────────────────────────────
    @property
    def n_phases(self) -> int:
        return len(self.config.phases)

    @property
    def state_dim(self) -> int:
        return len(MOVEMENTS) + 2 if self.config.state_mode == "movement" else 4

    @property
    def min_green(self) -> int:
        return self.config.min_green

    @property
    def max_green(self) -> int:
        return self.config.max_green

    def phase_name(self, phase: int | None = None) -> str:
        return self.config.phases[self.phase if phase is None else phase][0]

    def green_movements(self, phase: int | None = None) -> tuple:
        return self.config.phases[self.phase if phase is None else phase][1]

    # ── 대기열 조회 ─────────────────────────────────────────────────
    def queue_snapshot(self) -> dict:
        return {m: len(q) for m, q in self.queues.items()}

    @property
    def total_queue(self) -> int:
        return sum(len(q) for q in self.queues.values())

    @property
    def ns_queue(self) -> int:
        return sum(len(self.queues[m]) for m in MOVEMENTS if m[0] in "NS")

    @property
    def ew_queue(self) -> int:
        return sum(len(self.queues[m]) for m in MOVEMENTS if m[0] in "EW")

    @property
    def avg_wait(self) -> float:
        """지금까지 초록불로 통과한 차량들의 평균 대기시간(초). 우회전 차량은 제외."""
        return self.total_wait / self.n_departed if self.n_departed else 0.0

    # ── 에피소드 ────────────────────────────────────────────────────
    def reset(self):
        # 각 대기열에는 차가 도착한 시각(초)을 넣어둔다. 통과할 때 '출발 - 도착'으로 대기시간을 잰다.
        self.queues = {m: deque() for m in MOVEMENTS}
        # 통과율이 초당 1대 미만이어도 차가 정수 단위로 빠져나가도록 이동류별로 쌓아두는 값.
        self._credit = {m: 0.0 for m in MOVEMENTS}

        self.phase = 0
        self.phase_time = 0
        self.cycle_time = 0

        self.total_passed = 0
        self.total_wait = 0.0
        self.n_departed = 0
        return self.get_state()

    def get_state(self):
        if self.config.state_mode == "simple":
            return (self.ns_queue, self.ew_queue, self.phase, self.phase_time)
        return tuple(len(self.queues[m]) for m in MOVEMENTS) + (self.phase, self.phase_time)

    def normalize_state(self, state) -> list:
        """신경망 입력용으로 값의 크기를 0~1 근처로 맞춘다."""
        *queues, phase, phase_time = state
        return ([q / QUEUE_NORMALIZER for q in queues]
                + [phase / max(1, self.n_phases - 1), phase_time / self.max_green])

    def _spawn_vehicles(self) -> list:
        """방향별로 차량 도착을 굴린다. 반환값은 우회전으로 바로 통과한 차의 진입 방향 목록."""
        ratio = self.config.turn_ratio
        right_turns = []
        for approach in APPROACHES:
            prob = min(self.arrival_per_hour[approach] / 3600.0, 1.0)
            if self.rng.random() >= prob:
                continue

            r = self.rng.random()
            if r < ratio["straight"]:
                self.queues[f"{approach}_straight"].append(self.cycle_time)
            elif r < ratio["straight"] + ratio["left"]:
                self.queues[f"{approach}_left"].append(self.cycle_time)
            else:
                right_turns.append(approach)  # 우회전: 줄 서지 않고 바로 통과
        return right_turns

    def _discharge_vehicles(self) -> list:
        """초록불인 이동류마다 차로 하나씩 차를 내보낸다. 반환값은 [(이동류, 대기시간)]."""
        departures = []
        for movement in self.green_movements():
            self._credit[movement] += self.pass_rate
            capacity = int(self._credit[movement])
            self._credit[movement] -= capacity

            queue = self.queues[movement]
            for _ in range(min(capacity, len(queue))):
                wait = self.cycle_time - queue.popleft()
                departures.append((movement, wait))
                self.total_wait += wait
                self.n_departed += 1
        return departures

    def step(self, action: int):
        if action not in (STAY, SWITCH):
            raise ValueError(f"알 수 없는 action: {action!r} (0=stay, 1=switch만 허용)")

        prev_total_queue = self.total_queue

        switched = False
        can_switch = self.phase_time >= self.min_green
        # 주기 상한을 지키기 위해 한 페이즈를 max_green보다 오래 붙잡을 수 없다.
        must_switch = self.phase_time >= self.max_green
        if (action == SWITCH and can_switch) or must_switch:
            # 빨간불이 된 이동류에 남아 있던 통과 여유분은 버린다.
            for movement in self.green_movements():
                self._credit[movement] = 0.0
            self.phase = (self.phase + 1) % self.n_phases
            self.phase_time = 0
            switched = True
        else:
            self.phase_time += 1

        right_turns = self._spawn_vehicles()
        departures = self._discharge_vehicles()
        self.total_passed += len(departures) + len(right_turns)
        self.cycle_time += 1

        if self.config.reward_mode == "queue":
            reward = (-self.total_queue * self.config.queue_reward_scale
                      - self.switch_penalty * switched)
        else:  # "delta" — 연구계획서 원안
            reward = -(self.total_queue - prev_total_queue) - self.switch_penalty * switched

        done = (self.config.episode_seconds is not None
                and self.cycle_time >= self.config.episode_seconds)
        info = StepInfo(
            switched=switched,
            passed=len(departures),
            right_passed=len(right_turns),
            total_passed=self.total_passed,
            total_queue=self.total_queue,
            ns_queue=self.ns_queue,
            ew_queue=self.ew_queue,
            queues=self.queue_snapshot(),
            departures=departures,
            right_turns=right_turns,
        )
        return self.get_state(), reward, done, info

    def __repr__(self):
        queues = ", ".join(f"{movement_name(m)} {n}" for m, n in self.queue_snapshot().items() if n)
        return (f"CrosswayEnvironment(preset={self.preset!r}, 신호={self.phase_name()!r}, "
                f"t={self.cycle_time}, 통과={self.total_passed}, "
                f"평균대기={self.avg_wait:.1f}초, 대기열=[{queues}])")


if __name__ == "__main__":
    # 랜덤 정책으로 한 episode 굴려보고 환경이 정상 동작하는지 확인.
    for preset in PRESETS:
        env = CrosswayEnvironment(preset=preset, seed=0)
        total_reward, done = 0.0, False
        while not done:
            _, reward, done, info = env.step(env.rng.choice([STAY, SWITCH]))
            total_reward += reward
        print(f"[{preset}] reward {total_reward:8.1f} | {env}")
