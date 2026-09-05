"""단일 교차로 신호 제어 환경 (팀 공용 모듈) — ★수정본★

⚠️ 저장소에 있는 확장자 없는 구버전 `crossway_environment`는 더 이상 쓰지 말 것.
   이 파일이 최신 수정본이며, 구버전에는 아래 문제들이 있었다:
     - 보상 함수가 상쇄되어 학습이 되지 않음 (자세한 내용은 README 4절)
     - 남북/동서 교통량을 분리할 수 없어 적응형 제어의 이점이 드러나지 않음
     - 페이즈 강제 전환(주기 상한 120초)이 구현되지 않음
   Python은 `import crossway_environment` 시 .py가 붙은 이 파일만 읽으므로
   실행에는 영향이 없지만, 코드를 수정할 때는 반드시 이 파일을 고칠 것.

Q-learning, DQN 등 어떤 에이전트든 아래 Gym 스타일 인터페이스만 따르면 이 환경으로
학습할 수 있다: `env.reset()` -> state, `env.step(action)` -> (state, reward, done, info)

state  = (남북 대기차량수, 동서 대기차량수, 현재 페이즈 인덱스, 현재 페이즈 유지시간)
action = 0(stay, 현재 페이즈 유지) / 1(switch, 다음 페이즈로 전환. 최소 유지시간 미만이면 무시됨)
reward = reward_mode에 따라 두 가지 (아래 참고)

보상 함수는 두 가지 모드를 제공한다:

- "queue" (기본): `-(현재 총 대기차량수) * QUEUE_REWARD_SCALE - switch_penalty * (전환 여부)`
  매 스텝의 정체 '수준'을 벌하므로, 누적 보상을 최대화하는 것이 곧 평균 대기열을 줄이는 것과 같다.

- "delta" (연구계획서 원안): `-(총 대기차량 증가분) - switch_penalty * (전환 여부)`
  주의: 증가분을 한 에피소드 동안 모두 더하면 중간 항이 상쇄되어 `-(마지막 대기열)`만 남는다.
  즉 중간에 얼마나 정체됐는지가 보상에 반영되지 않아, 에이전트가 '전환을 최대한 안 하는' 정책으로
  수렴한다. 두 모드의 비교 자체가 실험 결과로 쓸 만하므로 원안도 남겨둔다.

페이즈는 남북 직진 -> 동서 직진 -> 남북 좌회전 -> 동서 좌회전 순으로 순환하며,
각 페이즈는 해당 이동류(직진/좌회전)의 대기열만 통과시킨다. 우회전은 신호와 무관하게
항상 통과 가능하다고 가정해 대기열 없이 즉시 통과 처리한다.

`arrival_prob`, `pass_rate`, `switch_penalty`는 시뮬레이션 도중에도 그대로 바꿔 쓸 수 있다
(시각화 도구에서 슬라이더로 실시간 조절하는 용도).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PHASES = ["남북 직진", "동서 직진", "남북 좌회전", "동서 좌회전"]
PHASE_NS_STRAIGHT, PHASE_EW_STRAIGHT, PHASE_NS_LEFT, PHASE_EW_LEFT = range(4)

# 각 페이즈에서 초록불이 켜지는 이동류(= 대기열 속성명)
MOVEMENT_BY_PHASE = ["ns_straight", "ew_straight", "ns_left", "ew_left"]
MOVEMENTS = ["ns_straight", "ns_left", "ew_straight", "ew_left"]

MIN_GREEN_TIME = 15          # 최소 초록불 유지 시간(초)
MAX_CYCLE_TIME = 120         # 4페이즈를 한 바퀴 도는 전체 신호 주기의 상한(초)
# 주기 상한을 지키려면 한 페이즈가 붙잡을 수 있는 시간도 상한이 필요하다.
# 이 시간을 넘기면 action과 무관하게 강제로 전환된다.
MAX_GREEN_TIME = MAX_CYCLE_TIME // len(PHASES)   # 30초
EPISODE_SECONDS = 600        # 한 에피소드 길이(초). 신호 주기보다 충분히 길어야 한다.
PASS_RATE_PER_SEC = 0.5      # 초록불일 때 초당 통과 가능 대수

# 방향별 이동류 비율 (직진 / 좌회전 / 우회전)
STRAIGHT_RATIO = 0.34
LEFT_RATIO = 0.33
RIGHT_RATIO = 0.33  # 우회전은 신호 무관 상시 통과 -> 대기열에 넣지 않음

STAY, SWITCH = 0, 1
N_ACTIONS = 2
STATE_DIM = 4

REWARD_MODES = ("queue", "delta")
# 대기열(수십 대)을 그대로 보상에 쓰면 값이 너무 커서 DQN 학습이 불안정해지므로 축소한다.
QUEUE_REWARD_SCALE = 0.1

# 교통량 프리셋: 방향별(남북/동서) 차량 도착 확률(초당)
PRESETS = {
    "normal": {"arrival_prob": 0.30},
    "rush_hour": {"arrival_prob": 0.55},
    "night": {"arrival_prob": 0.10},
}


@dataclass
class StepInfo:
    """step() 한 번의 결과 상세. 성능 지표 집계와 시각화에 사용."""

    switched: bool                  # 이번 step에 신호가 전환됐는지
    passed: int                     # 이번 step에 초록불로 통과한 차량 수
    right_passed: int               # 이번 step에 우회전으로 상시 통과한 차량 수
    total_passed: int               # 에피소드 누적 통과 차량 수 (우회전 포함)
    total_queue: int                # 현재 총 대기 차량 수
    ns_queue: int
    ew_queue: int
    queues: dict = field(default_factory=dict)  # 이동류별 대기열 (렌더링용)


class CrosswayEnvironment:
    """단일 4방향 교차로 신호 제어 환경."""

    def __init__(self, preset: str = "normal", switch_penalty: float = 2.0,
                 reward_mode: str = "queue", episode_seconds: int | None = EPISODE_SECONDS,
                 seed: int | None = None):
        if preset not in PRESETS:
            raise ValueError(f"알 수 없는 preset: {preset!r} (사용 가능: {list(PRESETS)})")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"알 수 없는 reward_mode: {reward_mode!r} (사용 가능: {REWARD_MODES})")

        self.preset = preset
        self.reward_mode = reward_mode
        # 방향별 도착 확률을 따로 두면 비대칭 교통량(간선 vs 이면도로)을 재현할 수 있다.
        # 대칭 환경에서는 '바로 전환'이 최적이라 적응형 제어가 고정주기를 이길 여지가 없다.
        self.arrival_prob_ns = PRESETS[preset]["arrival_prob"]
        self.arrival_prob_ew = PRESETS[preset]["arrival_prob"]
        self.pass_rate = PASS_RATE_PER_SEC
        self.switch_penalty = switch_penalty
        # None이면 에피소드가 끝나지 않는다 (시각화용 무한 실행 모드).
        self.episode_seconds = episode_seconds
        self.rng = random.Random(seed)

        self.reset()

    def reset(self):
        # 이동류별 대기열 (신호가 관장하는 4개)
        self.ns_straight = 0
        self.ns_left = 0
        self.ew_straight = 0
        self.ew_left = 0

        self.phase = PHASE_NS_STRAIGHT
        self.phase_time = 0
        self.cycle_time = 0
        self.total_passed = 0

        # 통과율이 초당 1대 미만이어도 차량이 정수 단위로 빠져나가도록 누적하는 크레딧.
        self._discharge_credit = 0.0
        return self._get_state()

    @property
    def arrival_prob(self) -> float:
        """양방향 평균 도착 확률. 대입하면 남북/동서를 한꺼번에 같은 값으로 맞춘다."""
        return (self.arrival_prob_ns + self.arrival_prob_ew) / 2

    @arrival_prob.setter
    def arrival_prob(self, value: float):
        self.arrival_prob_ns = value
        self.arrival_prob_ew = value

    @property
    def ns_queue(self) -> int:
        return self.ns_straight + self.ns_left

    @property
    def ew_queue(self) -> int:
        return self.ew_straight + self.ew_left

    @property
    def total_queue(self) -> int:
        return self.ns_queue + self.ew_queue

    def queue_snapshot(self) -> dict:
        return {m: getattr(self, m) for m in MOVEMENTS}

    def _get_state(self):
        return (self.ns_queue, self.ew_queue, self.phase, self.phase_time)

    def _spawn_vehicles(self) -> int:
        """방향별로 차량 도착을 굴린다. 반환값은 우회전으로 즉시 통과한 대수."""
        right_passed = 0
        for direction, prob in (("ns", self.arrival_prob_ns), ("ew", self.arrival_prob_ew)):
            if self.rng.random() >= prob:
                continue

            r = self.rng.random()
            if r < STRAIGHT_RATIO:
                movement = "straight"
            elif r < STRAIGHT_RATIO + LEFT_RATIO:
                movement = "left"
            else:
                right_passed += 1  # 우회전: 대기 없이 그대로 통과
                continue

            setattr(self, f"{direction}_{movement}", getattr(self, f"{direction}_{movement}") + 1)

        return right_passed

    def _discharge_vehicles(self) -> int:
        """현재 초록불인 이동류의 대기열만 통과시킨다. 반환값은 통과 대수."""
        self._discharge_credit += self.pass_rate
        capacity = int(self._discharge_credit)
        if capacity == 0:
            return 0
        self._discharge_credit -= capacity

        movement = MOVEMENT_BY_PHASE[self.phase]
        waiting = getattr(self, movement)
        passed = min(waiting, capacity)
        setattr(self, movement, waiting - passed)
        return passed

    def step(self, action: int):
        if action not in (STAY, SWITCH):
            raise ValueError(f"알 수 없는 action: {action!r} (0=stay, 1=switch만 허용)")

        prev_total_queue = self.total_queue

        switched = False
        can_switch = self.phase_time >= MIN_GREEN_TIME
        # 주기 상한(120초)을 지키기 위해 한 페이즈를 30초 넘게 붙잡을 수는 없다.
        must_switch = self.phase_time >= MAX_GREEN_TIME
        if (action == SWITCH and can_switch) or must_switch:
            self.phase = (self.phase + 1) % len(PHASES)
            self.phase_time = 0
            switched = True
        else:
            self.phase_time += 1

        right_passed = self._spawn_vehicles()
        passed = self._discharge_vehicles()
        self.total_passed += passed + right_passed
        self.cycle_time += 1

        if self.reward_mode == "queue":
            reward = -self.total_queue * QUEUE_REWARD_SCALE - self.switch_penalty * switched
        else:  # "delta" — 연구계획서 원안
            reward = -(self.total_queue - prev_total_queue) - self.switch_penalty * switched

        done = self.episode_seconds is not None and self.cycle_time >= self.episode_seconds
        info = StepInfo(
            switched=switched,
            passed=passed,
            right_passed=right_passed,
            total_passed=self.total_passed,
            total_queue=self.total_queue,
            ns_queue=self.ns_queue,
            ew_queue=self.ew_queue,
            queues=self.queue_snapshot(),
        )
        return self._get_state(), reward, done, info

    def phase_name(self) -> str:
        return PHASES[self.phase]

    def green_movement(self) -> str:
        return MOVEMENT_BY_PHASE[self.phase]

    def __repr__(self):
        return (f"CrosswayEnvironment(preset={self.preset!r}, phase={self.phase_name()!r}, "
                f"ns_queue={self.ns_queue}, ew_queue={self.ew_queue}, "
                f"passed={self.total_passed}, t={self.cycle_time})")


if __name__ == "__main__":
    # 랜덤 정책으로 한 episode 굴려보고 환경이 정상 동작하는지 확인.
    env = CrosswayEnvironment(preset="rush_hour", seed=0)
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, info = env.step(env.rng.choice([STAY, SWITCH]))
        total_reward += reward

    print(env)
    print(f"랜덤 정책 episode 총 reward: {total_reward:.1f}, 통과 차량: {env.total_passed}대")
