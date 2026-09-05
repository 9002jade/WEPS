# 강화학습 기반 교차로 신호 제어 시뮬레이션

2026 WEPS 연구 — "강화학습을 활용한 신호등 최적화 방안 탐구"
조장 이재범(20621) · 박제환(20411) · 한창우(20426)

4방향 교차로 환경에서 **Q-learning**과 **DQN**으로 신호를 제어하고 성능을 비교한다.

---

## 1. 처음 한 번만: 설치

Python 3.10 이상이 필요하다. ([python.org](https://www.python.org/downloads/)에서 설치,
설치 화면에서 **"Add Python to PATH"** 체크할 것)

저장소를 받고 이 폴더로 이동한다:

```bash
git clone https://github.com/9002jade/WEPS.git
cd WEPS/jaebeomlee
pip install -r requirements.txt
```

> 아래 모든 명령은 `WEPS/jaebeomlee` 폴더 안에서 실행한다.

> tkinter(시각화용)는 Python에 기본 포함이라 따로 설치하지 않아도 된다.

## 2. 실행 방법

### 시각화 — 차가 움직이는 걸 눈으로 보기

```bash
python visualize.py
```

- 좌측: 교차로. 차량이 색깔별로 대기줄에 서고 초록불에 통과한다.
- 우측: 실험 조건 설정 + 실행 기록.
- **실행 중에는 설정이 잠긴다.** 끝나면 수정하고 "재실행"을 누른다.
  (실행 도중 조건을 바꾸면 평균 지표가 여러 조건의 혼합이 되어 비교가 불가능해지기 때문)
- `DQN 제어` 체크를 껐다 켜며 같은 조건에서 고정주기 신호와 비교할 수 있다.

학습된 모델(`results/dqn_normal.pt`)이 이미 들어있어 바로 실행된다.
없다면 아래 학습을 먼저 돌린다.

### 학습

```bash
python train_dqn.py                      # DQN, normal 교통량
python train_dqn.py --preset rush_hour   # 러시아워
python q_learning_agent.py               # Q-learning
```

### 성능 평가 (60회 반복 평균 — 발표용 수치는 이걸 쓸 것)

```bash
python evaluate.py
```

DQN과 고정주기 신호를 대칭/비대칭 교통량 두 시나리오에서 비교한다.

---

## 3. 파일 구조

> ⚠️ 저장소에 확장자 없는 구버전 `crossway_environment` 파일이 남아 있다.
> **`crossway_environment.py`(수정본)가 최신**이며, 환경을 고칠 때는 반드시 이쪽을 고칠 것.
> 구버전은 아래 4절의 문제들이 수정되기 전 버전이다.

| 파일 | 역할 |
|---|---|
| `crossway_environment.py` | 교차로 시뮬레이션 환경 (팀 공용, **수정본**). Gym 스타일 `reset()` / `step(action)` |
| `q_learning_agent.py` | Q-table 기반 에이전트 |
| `dqn_agent.py` | PyTorch DQN (Experience Replay + Target Network) |
| `train_dqn.py` | DQN 학습, 모델을 `results/`에 저장 |
| `evaluate.py` | 성능 지표 측정 및 기준선 비교 |
| `visualize.py` | tkinter 실시간 시각화 |
| `results/` | 학습된 모델(.pt), 학습 이력·평가 결과(.json) |

새 알고리즘을 추가하려면 환경만 가져다 쓰면 된다:

```python
from crossway_environment import CrosswayEnvironment, STAY, SWITCH

env = CrosswayEnvironment(preset="normal", seed=0)
state = env.reset()
next_state, reward, done, info = env.step(STAY)
```

---

## 4. 스펙에서 바뀐 점 (중요 — 보고서에 반영 필요)

구현하며 연구계획서 원안의 문제 세 가지를 발견해 수정했다.

**① 보상 함수** — 원안 `reward = -(대기열 증가분)`은 한 에피소드 동안 더하면
중간 항이 상쇄되어 `-(마지막 대기열)`만 남는다. 중간에 얼마나 막혔는지가
보상에 반영되지 않아, DQN이 "전환을 최대한 안 하기"만 학습해 평균 대기열이
오히려 8.4대 → 11.6대로 **악화**됐다.
→ 대기열의 *수준*을 벌하는 `reward_mode="queue"`를 기본값으로 바꿨다.
원안은 `reward_mode="delta"`로 남아있으니 **두 보상 설계 비교 자체를 실험 결과로 쓸 것.**

**② 할인율(gamma)** — 0.9는 1초 스텝에서 약 10초 앞만 본다. 신호 한 주기가
60~120초라 에이전트가 자기 결정의 결과를 볼 수 없었다. → 0.99로 변경.

**③ "전체 신호 주기 최대 120초"의 해석** — 이건 4페이즈를 한 바퀴 도는 시간의
상한이지 에피소드 길이가 아니다. → 페이즈당 최대 30초 강제 전환(`MAX_GREEN_TIME`)으로
구현하고, 에피소드 길이는 `EPISODE_SECONDS=600`으로 분리했다.

**④ 방향별 교통량 분리** — 남북/동서 교통량이 같으면 "바로 전환"이 최적이라
적응형 제어가 이길 여지가 구조적으로 없다(실제로 DQN이 고정주기와 완전히 동일한
정책을 학습했다). 실제 교차로에서 적응형 신호가 쓰이는 이유는 방향별 교통량이
다르기 때문이므로 `arrival_prob_ns` / `arrival_prob_ew`를 분리했다.

## 5. 현재 성능 (evaluate.py, 60 에피소드 평균)

| 시나리오 | DQN 평균대기열 | 최고 기준선 | 결과 |
|---|---|---|---|
| 대칭 교통량 | 12.87대 | 11.63대 (고정 15초) | 10.6% 악화 |
| 비대칭 (남북 편중) | **26.80대** | 27.51대 (고정 20초) | **2.6% 개선** |

대칭 환경에서 DQN이 지는 건 정상이다 — 적응할 여지가 없는 조건에서는
신경망 근사 오차만 손해로 남는다.

## 6. 다음 과제

- **행동 공간 확장 검토**: 현재는 "다음 페이즈로만" 전환 가능해서 한산한 방향도
  최소 15초는 반드시 서줘야 한다. "어느 페이즈로 갈지 선택"으로 바꾸면 빈 페이즈를
  건너뛸 수 있어 적응 이득이 훨씬 커진다. (스펙 3.4 변경이라 팀 논의 필요)
- Q-learning vs DQN 정식 비교 실험 (실험 1~3)
- 다중 교차로 환경 (실험 3)
- 공공데이터 연동 `data_loader.py` (실험 4)
