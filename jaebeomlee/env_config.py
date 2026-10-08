"""교차로 환경 설정 — 팀원이 바꿔볼 값은 전부 이 파일에 있다.

환경의 동작 로직(crossway_environment.py)은 건드리지 않고, 여기 값만 바꿔서 실험한다.

사용 예:
    from crossway_environment import CrosswayEnvironment
    from env_config import make_config

    env = CrosswayEnvironment(preset="rush_hour", seed=0)                 # 프리셋 그대로
    env = CrosswayEnvironment(preset="normal", min_green=10, seed=0)      # 일부만 바꾸기
    env = CrosswayEnvironment(config=make_config("normal",                # 방향별 교통량 직접 지정
                              arrival_per_hour={"N": 900, "S": 900, "E": 300, "W": 300}))

방향 표기: N/S/E/W는 차가 '들어오는 쪽'이다. 예) "N_straight" = 북쪽에서 들어와 직진(남쪽으로 감).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ── 방향과 이동류 ────────────────────────────────────────────────────
APPROACHES = ("N", "S", "E", "W")
APPROACH_NAMES = {"N": "북쪽", "S": "남쪽", "E": "동쪽", "W": "서쪽"}

# 신호를 받는 이동류는 직진·좌회전 두 가지. 우회전은 신호와 상관없이 항상 통과한다고 본다.
SIGNALED_TURNS = ("straight", "left")
TURN_NAMES = {"straight": "직진", "left": "좌회전", "right": "우회전"}

# 4방향 × 2이동류 = 8개 대기열. 상태 벡터도 이 순서를 따른다.
MOVEMENTS = tuple(f"{a}_{t}" for a in APPROACHES for t in SIGNALED_TURNS)


def movement_name(movement: str) -> str:
    """'N_left' -> '북쪽 좌회전'"""
    approach, turn = movement.split("_")
    return f"{APPROACH_NAMES[approach]} {TURN_NAMES[turn]}"


# ── 신호 페이즈 ──────────────────────────────────────────────────────
# (페이즈 이름, 이 페이즈에서 초록불을 받는 이동류들). 위에서부터 순서대로 돈다.
# 순서를 바꾸거나 페이즈를 추가해도 된다. 예) 방향별 단독 좌회전 페이즈를 넣고 싶으면
# ("북쪽 좌회전", ("N_left",)) 같은 항목을 추가하면 된다.
# 단, 모든 이동류가 적어도 한 페이즈에는 들어가 있어야 한다 (안 그러면 그 차는 영원히 못 감).
DEFAULT_PHASES = (
    ("남북 직진", ("N_straight", "S_straight")),
    ("동서 직진", ("E_straight", "W_straight")),
    ("남북 좌회전", ("N_left", "S_left")),
    ("동서 좌회전", ("E_left", "W_left")),
)

# ── 교통량 프리셋 (방향별 시간당 진입 대수) ──────────────────────────
# 차로 하나의 처리 능력은 약 1,800대/시간(초당 0.5대)이고, 한 이동류는 신호 주기의 1/4 정도만
# 초록불을 받으므로 이동류당 처리 능력은 약 450대/시간이다. 이를 기준으로 혼잡도를 맞췄다.
#   night     : 한산함
#   normal    : 처리 능력의 절반 정도
#   rush_hour : 처리 능력에 거의 꽉 참 (적응형 제어의 차이가 드러나는 구간)
#   ns_heavy  : 남북이 간선도로, 동서가 이면도로인 비대칭 교차로
PRESETS = {
    "night": {"N": 200, "S": 200, "E": 200, "W": 200},
    "normal": {"N": 600, "S": 600, "E": 600, "W": 600},
    "rush_hour": {"N": 1200, "S": 1200, "E": 1200, "W": 1200},
    "ns_heavy": {"N": 1100, "S": 1100, "E": 350, "W": 350},
}


@dataclass
class EnvConfig:
    """환경의 모든 조절 가능한 값. 각 항목 옆 설명을 보고 바꾸면 된다."""

    # 방향별 시간당 진입 대수 (대/시간). 우회전 차량도 포함한 총량.
    arrival_per_hour: dict = field(default_factory=lambda: dict(PRESETS["normal"]))

    # 진입 차량의 이동 비율. 합이 1이어야 한다. (연구계획서 기본값 34/33/33)
    # 참고: 항저우 실측 데이터에서는 직진 85% / 좌회전 15% 정도였다.
    turn_ratio: dict = field(
        default_factory=lambda: {"straight": 0.34, "left": 0.33, "right": 0.33})

    # 초록불일 때 차로 하나에서 초당 빠져나가는 대수.
    pass_rate: float = 0.5

    # 신호 페이즈 구성 (위 DEFAULT_PHASES 설명 참고).
    phases: tuple = DEFAULT_PHASES

    # 최소 초록불 시간(초). 이보다 짧으면 switch를 해도 무시된다.
    min_green: int = 15

    # 모든 페이즈를 한 바퀴 도는 주기의 상한(초). 페이즈 하나가 붙잡을 수 있는 최대 시간은
    # max_cycle / 페이즈 수 이고, 이를 넘기면 action과 상관없이 강제로 다음 페이즈로 넘어간다.
    max_cycle: int = 120

    # 한 에피소드 길이(초). None이면 끝나지 않는다(시각화용).
    episode_seconds: int | None = 600

    # 보상 방식. "queue"(기본): 대기 차량 수준을 벌함 / "delta": 연구계획서 원안(증가분을 벌함).
    # delta는 에피소드 동안 더하면 상쇄되어 학습이 안 된다는 문제가 있다 (README 참고).
    reward_mode: str = "queue"
    queue_reward_scale: float = 0.1   # queue 보상에서 대기 차량 수에 곱하는 값
    switch_penalty: float = 0.5       # 신호를 바꿀 때마다 빼는 점수

    # 에이전트에게 주는 상태의 형태.
    #   "movement"(기본): 8개 이동류 대기열 + 페이즈 + 유지시간 = 10개 값
    #   "simple"        : 남북 합계 + 동서 합계 + 페이즈 + 유지시간 = 4개 값 (예전 방식)
    # 두 방식을 비교하면 '상태 정보량이 성능에 미치는 영향'을 실험할 수 있다.
    state_mode: str = "movement"

    @property
    def max_green(self) -> int:
        return self.max_cycle // len(self.phases)

    def validate(self):
        """설정을 잘못 바꿨을 때 실행 초반에 바로 알려준다."""
        missing = set(APPROACHES) - set(self.arrival_per_hour)
        if missing:
            raise ValueError(f"arrival_per_hour에 방향이 빠졌습니다: {sorted(missing)}")
        if abs(sum(self.turn_ratio.values()) - 1.0) > 1e-6:
            raise ValueError(f"turn_ratio의 합이 1이 아닙니다: {self.turn_ratio}")
        covered = {m for _, ms in self.phases for m in ms}
        unknown = covered - set(MOVEMENTS)
        if unknown:
            raise ValueError(f"phases에 없는 이동류 이름이 있습니다: {sorted(unknown)} "
                             f"(사용 가능: {MOVEMENTS})")
        uncovered = set(MOVEMENTS) - covered
        if uncovered:
            raise ValueError(f"어느 페이즈에도 들어가지 않은 이동류가 있습니다: {sorted(uncovered)}")
        if self.max_green < self.min_green:
            raise ValueError(f"max_cycle({self.max_cycle})/페이즈 수({len(self.phases)}) = "
                             f"{self.max_green}초가 min_green({self.min_green}초)보다 짧습니다.")
        if self.reward_mode not in ("queue", "delta"):
            raise ValueError(f"reward_mode는 'queue' 또는 'delta'만 가능합니다: {self.reward_mode!r}")
        if self.state_mode not in ("movement", "simple"):
            raise ValueError(f"state_mode는 'movement' 또는 'simple'만 가능합니다: {self.state_mode!r}")


def skewed_demand(arrival_per_hour: dict, skew: float) -> dict:
    """남북과 동서 교통량 비율을 바꾼다. 총량은 유지된다.

    skew=1.0이면 그대로, 1.6이면 남북 ×1.6 / 동서 ×0.4, 0.5면 남북 ×0.5 / 동서 ×1.5.
    """
    return {a: v * (skew if a in "NS" else 2.0 - skew) for a, v in arrival_per_hour.items()}


def make_config(preset: str = "normal", **overrides) -> EnvConfig:
    """프리셋 교통량으로 설정을 만들고, 원하는 항목만 덮어쓴다."""
    if preset not in PRESETS:
        raise ValueError(f"알 수 없는 preset: {preset!r} (사용 가능: {list(PRESETS)})")
    config = replace(EnvConfig(arrival_per_hour=dict(PRESETS[preset])), **overrides)
    config.validate()
    return config
