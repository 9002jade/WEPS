import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ============================================================
# 1. 실험 재현을 위한 난수 설정
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. 미로 환경
# ============================================================

ROWS = 4
COLS = 4

START = (0, 0)
GOAL = (3, 3)

# 벽
WALLS = {
    (1, 1),
    (2, 1)
}

# 행동
# 0 = 위
# 1 = 아래
# 2 = 왼쪽
# 3 = 오른쪽

ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

ACTION_NAMES = [
    "위",
    "아래",
    "왼쪽",
    "오른쪽"
]


# ============================================================
# 3. DQN 하이퍼파라미터
# ============================================================

GAMMA = 0.95

LEARNING_RATE = 0.001

INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

EPISODES = 1000

MAX_STEPS = 50

BATCH_SIZE = 64

MEMORY_SIZE = 5000

TARGET_UPDATE = 20


# ============================================================
# 4. 상태를 신경망 입력으로 변환
# ============================================================

def state_to_tensor(state):

    """
    현재 위치를 16차원 벡터로 변환한다.

    4 x 4 미로이므로 총 16개의 위치가 존재한다.

    예:
    (0,0) → 1번째 위치가 1
    (1,2) → 7번째 위치가 1
    """

    tensor = torch.zeros(
        ROWS * COLS,
        dtype=torch.float32
    )

    r, c = state

    index = r * COLS + c

    tensor[index] = 1.0

    return tensor


# ============================================================
# 5. 환경의 움직임
# ============================================================

def environment_step(state, action):

    r, c = state

    dr, dc = ACTIONS[action]

    next_state = (
        r + dr,
        c + dc
    )

    # --------------------------------------------------------
    # 미로 밖으로 나가는 경우
    # --------------------------------------------------------

    if (
        next_state[0] < 0
        or next_state[0] >= ROWS
        or next_state[1] < 0
        or next_state[1] >= COLS
    ):

        return state, -1.0, False


    # --------------------------------------------------------
    # 벽에 부딪히는 경우
    # --------------------------------------------------------

    if next_state in WALLS:

        return state, -1.0, False


    # --------------------------------------------------------
    # 목표에 도착
    # --------------------------------------------------------

    if next_state == GOAL:

        return next_state, 10.0, True


    # --------------------------------------------------------
    # 일반적인 이동
    # --------------------------------------------------------

    return next_state, -0.1, False


# ============================================================
# 6. DQN 신경망
# ============================================================

class DQN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # 입력: 현재 상태 16개
            nn.Linear(16, 64),

            nn.ReLU(),

            # 은닉층
            nn.Linear(64, 64),

            nn.ReLU(),

            # 출력:
            # 위 / 아래 / 왼쪽 / 오른쪽
            nn.Linear(64, 4)
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 7. 경험 재생 메모리
# ============================================================

memory = deque(
    maxlen=MEMORY_SIZE
)


# ============================================================
# 8. 행동 선택
# ============================================================

def choose_action(
    state,
    epsilon,
    model
):

    # --------------------------------------------------------
    # 탐험
    # --------------------------------------------------------

    if random.random() < epsilon:

        return random.randint(0, 3)


    # --------------------------------------------------------
    # 활용
    # --------------------------------------------------------

    state_tensor = (
        state_to_tensor(state)
        .unsqueeze(0)
    )

    with torch.no_grad():

        q_values = model(
            state_tensor
        )

    return torch.argmax(
        q_values
    ).item()


# ============================================================
# 9. DQN 학습 함수
# ============================================================

def train_dqn(
    policy_network,
    target_network,
    optimizer
):

    # 충분한 경험이 쌓이지 않았다면 학습하지 않음

    if len(memory) < BATCH_SIZE:

        return None


    # --------------------------------------------------------
    # 경험을 무작위로 추출
    # --------------------------------------------------------

    batch = random.sample(
        memory,
        BATCH_SIZE
    )


    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []


    for experience in batch:

        (
            state,
            action,
            reward,
            next_state,
            done
        ) = experience


        states.append(
            state_to_tensor(state)
        )

        actions.append(action)

        rewards.append(reward)

        next_states.append(
            state_to_tensor(next_state)
        )

        dones.append(done)


    states = torch.stack(states)

    next_states = torch.stack(
        next_states
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32
    )


    # ========================================================
    # 현재 상태에서 선택한 행동의 Q값
    # ========================================================

    current_q_values = (
        policy_network(states)
        .gather(
            1,
            actions.unsqueeze(1)
        )
        .squeeze(1)
    )


    # ========================================================
    # 다음 상태에서 가장 좋은 Q값
    # ========================================================

    with torch.no_grad():

        next_q_values = (
            target_network(next_states)
        )

        max_next_q = (
            next_q_values
            .max(dim=1)[0]
        )


        # Q-learning 목표값
        target_q_values = (
            rewards
            + GAMMA
            * max_next_q
            * (1 - dones)
        )


    # ========================================================
    # 현재 Q값과 목표 Q값의 차이
    # ========================================================

    loss_function = nn.SmoothL1Loss()

    loss = loss_function(
        current_q_values,
        target_q_values
    )


    # ========================================================
    # 신경망 업데이트
    # ========================================================

    optimizer.zero_grad()

    loss.backward()

    # 너무 큰 업데이트 방지
    torch.nn.utils.clip_grad_norm_(
        policy_network.parameters(),
        1.0
    )

    optimizer.step()


    return loss.item()


# ============================================================
# 10. 신경망 생성
# ============================================================

policy_network = DQN()

target_network = DQN()


# 처음에는 두 네트워크를 동일하게 설정

target_network.load_state_dict(
    policy_network.state_dict()
)


# ============================================================
# 11. Optimizer
# ============================================================

optimizer = optim.Adam(
    policy_network.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 12. 학습 시작
# ============================================================

print("=" * 60)
print("DQN 학습 시작")
print("=" * 60)

epsilon = INITIAL_EPSILON


for episode in range(
    1,
    EPISODES + 1
):

    # 시작점으로 초기화

    state = START

    steps = 0

    total_reward = 0

    done = False


    # --------------------------------------------------------
    # 하나의 Episode
    # --------------------------------------------------------

    while (
        not done
        and steps < MAX_STEPS
    ):

        # 행동 선택

        action = choose_action(
            state,
            epsilon,
            policy_network
        )


        # 환경에서 행동

        next_state, reward, done = (
            environment_step(
                state,
                action
            )
        )


        # ----------------------------------------------------
        # 경험 저장
        # ----------------------------------------------------

        memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


        # ----------------------------------------------------
        # 신경망 학습
        # ----------------------------------------------------

        loss = train_dqn(
            policy_network,
            target_network,
            optimizer
        )


        state = next_state

        total_reward += reward

        steps += 1


    # --------------------------------------------------------
    # 탐험률 감소
    # --------------------------------------------------------

    epsilon = max(
        MIN_EPSILON,
        epsilon * EPSILON_DECAY
    )


    # --------------------------------------------------------
    # Target Network 업데이트
    # --------------------------------------------------------

    if episode % TARGET_UPDATE == 0:

        target_network.load_state_dict(
            policy_network.state_dict()
        )


    # --------------------------------------------------------
    # 학습 과정 출력
    # --------------------------------------------------------

    if episode in [
        1,
        10,
        50,
        100,
        200,
        500,
        1000
    ]:

        print(
            f"Episode {episode:4d} | "
            f"목표 도달: {done} | "
            f"이동 횟수: {steps:2d} | "
            f"Reward: {total_reward:6.2f} | "
            f"ε = {epsilon:.3f}"
        )


# ============================================================
# 13. 학습 후 평가
# ============================================================

print()
print("=" * 60)
print("학습된 DQN 평가")
print("=" * 60)


state = START

path = [state]

total_reward = 0


for _ in range(MAX_STEPS):

    if state == GOAL:

        break


    # --------------------------------------------------------
    # 평가에서는 탐험하지 않는다.
    # 가장 높은 Q값을 가진 행동만 선택
    # --------------------------------------------------------

    state_tensor = (
        state_to_tensor(state)
        .unsqueeze(0)
    )


    with torch.no_grad():

        q_values = (
            policy_network(
                state_tensor
            )
        )


    action = torch.argmax(
        q_values
    ).item()


    next_state, reward, done = (
        environment_step(
            state,
            action
        )
    )


    path.append(next_state)

    total_reward += reward

    state = next_state


    if done:

        break


# ============================================================
# 14. 최적 경로 출력
# ============================================================

print()

print("학습된 경로:")

for i, state in enumerate(path):

    print(
        f"{i + 1:2d}. {state}"
    )


print()

print(
    f"총 이동 횟수: {len(path) - 1}"
)

print(
    f"총 Reward: {total_reward:.2f}"
)


# ============================================================
# 15. 최종 미로 표시
# ============================================================

print()
print("=" * 60)
print("최종 결과")
print("=" * 60)


path_set = set(path)


for r in range(ROWS):

    row = ""

    for c in range(COLS):

        state = (r, c)


        if state == START:

            row += " S "


        elif state == GOAL:

            row += " G "


        elif state in WALLS:

            row += " ■ "


        elif state in path_set:

            row += " ● "


        else:

            row += " · "


    print(row)
