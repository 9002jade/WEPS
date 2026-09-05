"""DQN 학습 스크립트. 학습된 모델을 results/ 에 저장한다.

사용법:
    python train_dqn.py                      # normal 프리셋, 400 episode
    python train_dqn.py --preset rush_hour   # 러시아워 환경으로 학습
    python train_dqn.py --episodes 800
"""

import argparse
import json
from pathlib import Path

import numpy as np

from crossway_environment import CrosswayEnvironment
from dqn_agent import DQNAgent

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 0


def train(preset="normal", episodes=150, reward_mode="queue", switch_penalty=0.5,
          randomize_demand=True, verbose_every=10):
    env = CrosswayEnvironment(preset=preset, reward_mode=reward_mode,
                              switch_penalty=switch_penalty, seed=SEED)
    # 학습 구간의 60% 지점에서 탐험이 끝나고 나머지는 수렴에 쓰이도록 맞춘다.
    agent = DQNAgent(seed=SEED, epsilon_decay_episodes=max(1, int(episodes * 0.6)))
    base_prob = env.arrival_prob

    history = {"reward": [], "avg_queue": [], "passed": []}

    for ep in range(episodes):
        state = env.reset()

        if randomize_demand:
            # 매 에피소드 방향별 교통량을 다르게 준다. 그래야 에이전트가 특정 비율을
            # 외우는 대신 '대기열을 보고 판단하는' 정책을 배운다.
            skew = env.rng.uniform(0.2, 1.8)
            env.arrival_prob_ns = min(base_prob * skew, 1.0)
            env.arrival_prob_ew = min(base_prob * (2.0 - skew), 1.0)
        total_reward = 0.0
        queue_samples = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            queue_samples.append(info.total_queue)

        agent.decay_epsilon(ep)
        history["reward"].append(total_reward)
        history["avg_queue"].append(float(np.mean(queue_samples)))
        history["passed"].append(env.total_passed)

        if verbose_every and (ep + 1) % verbose_every == 0:
            n = verbose_every
            print(f"episode {ep + 1:4d}/{episodes}  "
                  f"reward={np.mean(history['reward'][-n:]):7.2f}  "
                  f"평균대기={np.mean(history['avg_queue'][-n:]):5.2f}대  "
                  f"통과={np.mean(history['passed'][-n:]):5.1f}대  "
                  f"eps={agent.epsilon:.3f}")

    return agent, history


def main():
    parser = argparse.ArgumentParser(description="교차로 신호 제어 DQN 학습")
    parser.add_argument("--preset", default="normal", choices=["normal", "rush_hour", "night"])
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--reward-mode", default="queue", choices=["queue", "delta"])
    parser.add_argument("--switch-penalty", type=float, default=0.5)
    parser.add_argument("--no-randomize-demand", action="store_true",
                        help="방향별 교통량 무작위화를 끄고 대칭 교통량으로만 학습")
    args = parser.parse_args()

    print(f"=== DQN 학습 시작 (preset={args.preset}, episodes={args.episodes}, "
          f"reward={args.reward_mode}, penalty={args.switch_penalty}) ===")
    agent, history = train(preset=args.preset, episodes=args.episodes,
                           reward_mode=args.reward_mode, switch_penalty=args.switch_penalty,
                           randomize_demand=not args.no_randomize_demand)

    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = RESULTS_DIR / f"dqn_{args.preset}.pt"
    agent.save(model_path)
    (RESULTS_DIR / f"history_dqn_{args.preset}.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    last = 50
    print(f"\n마지막 {last} episode 평균 reward: {np.mean(history['reward'][-last:]):.2f}")
    print(f"마지막 {last} episode 평균 대기열: {np.mean(history['avg_queue'][-last:]):.2f}대")
    print(f"모델 저장 완료 -> {model_path}")
    print(f"\n시각화 실행:  python visualize.py --preset {args.preset}")


if __name__ == "__main__":
    main()
