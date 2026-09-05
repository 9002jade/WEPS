"""Q-table 기반 Q-learning 에이전트.

state는 연속적인 대기차량수를 그대로 쓰면 Q-table이 무한히 커지므로,
bucket 단위로 이산화(discretize)해서 Q-table의 인덱스로 사용한다.
"""

import random
from collections import defaultdict

import numpy as np

QUEUE_BUCKETS = [0, 2, 5, 10, 20]  # 대기차량수 구간 경계
# 페이즈 유지시간은 MAX_GREEN_TIME(30초)에서 강제 전환되므로 그 안에서 잘게 나눈다.
# 최소 초록시간 15초 전후를 구분하는 게 핵심 (그 전에는 전환 자체가 불가능).
TIME_BUCKETS = [5, 10, 14, 17, 20, 25]
N_ACTIONS = 2  # 0: stay, 1: switch


def _bucketize(value, boundaries):
    for i, b in enumerate(boundaries):
        if value <= b:
            return i
    return len(boundaries)


class QLearningAgent:
    # gamma는 DQN과 동일하게 맞춘다 (신호 주기를 볼 수 있는 시야 확보 + 공정한 비교).
    def __init__(self, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_episodes=200, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = random.Random(seed)
        # Q-table: dict[state_key] -> np.array(N_ACTIONS)
        self.q_table = defaultdict(lambda: np.zeros(N_ACTIONS))

    def discretize(self, state):
        ns_q, ew_q, phase, phase_time = state
        return (
            _bucketize(ns_q, QUEUE_BUCKETS),
            _bucketize(ew_q, QUEUE_BUCKETS),
            phase,
            _bucketize(phase_time, TIME_BUCKETS),
        )

    def select_action(self, state, greedy=False):
        key = self.discretize(state)
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done):
        key = self.discretize(state)
        next_key = self.discretize(next_state)

        best_next_q = 0.0 if done else np.max(self.q_table[next_key])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[key][action]
        self.q_table[key][action] += self.alpha * td_error

    def decay_epsilon(self, episode):
        # 선형 decay: epsilon_start -> epsilon_end 를 episode에 걸쳐 진행
        frac = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


def train(env, episodes=300, verbose_every=20):
    agent = QLearningAgent()
    episode_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon(ep)
        episode_rewards.append(total_reward)

        if verbose_every and (ep + 1) % verbose_every == 0:
            avg = np.mean(episode_rewards[-verbose_every:])
            print(f"[Q-learning] episode {ep + 1}/{episodes}  avg_reward={avg:.2f}  "
                  f"epsilon={agent.epsilon:.3f}  q_table_size={len(agent.q_table)}")

    return agent, episode_rewards


if __name__ == "__main__":
    from crossway_environment import CrosswayEnvironment

    env = CrosswayEnvironment(preset="normal", seed=0)
    trained_agent, rewards = train(env, episodes=300)
    print(f"\n최종 100 episode 평균 reward: {np.mean(rewards[-100:]):.2f}")
