import random

# ============================================================
# 1. 환경 설정
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
# 2. Q-learning 설정
# ============================================================

ALPHA = 0.1       # 학습률
GAMMA = 0.9       # 할인율

EPSILON = 0.3     # 탐험 확률
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

EPISODES = 500


# ============================================================
# 3. Q-table
# ============================================================

Q = {}

for r in range(ROWS):
    for c in range(COLS):

        state = (r, c)

        if state not in WALLS:

            Q[state] = [0.0, 0.0, 0.0, 0.0]


# ============================================================
# 4. 다음 상태 계산
# ============================================================

def move(state, action):

    r, c = state

    dr, dc = ACTIONS[action]

    nr = r + dr
    nc = c + dc

    next_state = (nr, nc)

    # 격자 밖
    if nr < 0 or nr >= ROWS:
        return state, -2

    if nc < 0 or nc >= COLS:
        return state, -2

    # 벽
    if next_state in WALLS:
        return state, -2

    # 목표
    if next_state == GOAL:
        return next_state, 10

    # 일반 이동
    return next_state, -1


# ============================================================
# 5. 행동 선택
# ============================================================

def choose_action(state, epsilon):

    # 탐험
    if random.random() < epsilon:
        return random.randint(0, 3)

    # 활용
    max_q = max(Q[state])

    best_actions = [
        i for i, q in enumerate(Q[state])
        if q == max_q
    ]

    return random.choice(best_actions)


# ============================================================
# 6. 학습
# ============================================================

print("=" * 60)
print("Q-learning 학습 시작")
print("=" * 60)

epsilon = EPSILON

for episode in range(1, EPISODES + 1):

    state = START
    steps = 0

    while state != GOAL and steps < 100:

        # 행동 선택
        action = choose_action(
            state,
            epsilon
        )

        # 행동 결과
        next_state, reward = move(
            state,
            action
        )

        # 현재 Q값
        old_q = Q[state][action]

        # 다음 상태의 최대 Q값
        if next_state == GOAL:
            next_max_q = 0
        else:
            next_max_q = max(
                Q[next_state]
            )

        # Q-learning 업데이트
        new_q = (
            old_q
            + ALPHA
            * (
                reward
                + GAMMA * next_max_q
                - old_q
            )
        )

        Q[state][action] = new_q

        state = next_state

        steps += 1

    # 탐험률 감소
    epsilon = max(
        MIN_EPSILON,
        epsilon * EPSILON_DECAY
    )

    # 학습 과정 출력
    if episode in [1, 10, 50, 100, 200, 500]:

        print(
            f"Episode {episode:3d} | "
            f"목표 도달: {state == GOAL} | "
            f"이동 횟수: {steps:2d} | "
            f"ε = {epsilon:.3f}"
        )


# ============================================================
# 7. 학습된 Q-table 확인
# ============================================================

print("\n")
print("=" * 60)
print("학습된 Q-table")
print("=" * 60)

for r in range(ROWS):

    for c in range(COLS):

        state = (r, c)

        if state in WALLS:

            print("  WALL  ", end="")

        else:

            values = Q[state]

            print(
                f"{max(values):7.2f}",
                end=" "
            )

    print()


# ============================================================
# 8. 학습된 최적 경로 확인
# ============================================================

print("\n")
print("=" * 60)
print("학습된 최적 경로")
print("=" * 60)

state = START
path = [state]

for _ in range(20):

    if state == GOAL:
        break

    action = Q[state].index(
        max(Q[state])
    )

    next_state, reward = move(
        state,
        action
    )

    path.append(next_state)

    state = next_state


for i, state in enumerate(path):

    print(
        f"{i + 1:2d}. {state}"
    )


# ============================================================
# 9. 격자에 최적 경로 표시
# ============================================================

print("\n")
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
