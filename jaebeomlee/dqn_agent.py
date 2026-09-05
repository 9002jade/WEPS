"""PyTorch 기반 DQN(Deep Q-Network) 에이전트.

Q-learning과 달리 state를 이산화하지 않고 연속값 벡터 그대로 신경망에 입력한다.
Experience Replay + Target Network를 사용해 학습을 안정화한다.
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

STATE_DIM = 4  # ns_queue, ew_queue, phase, phase_time
N_ACTIONS = 2  # 0: stay, 1: switch


class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000, seed=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def _normalize(state):
    ns_q, ew_q, phase, phase_time = state
    return [ns_q / 20.0, ew_q / 20.0, phase / 3.0, phase_time / 120.0]


class DQNAgent:
    # gamma=0.9는 1초 스텝에서 약 10초 앞만 내다본다. 신호 한 주기가 60~120초이므로
    # 그 정도 시야가 확보되도록 0.99(≈100초)를 쓴다.
    def __init__(self, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_episodes=200, batch_size=64, buffer_size=10_000,
                 target_update_every=100, seed=None):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.rng = random.Random(seed)

        self.policy_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        self.train_steps = 0

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)
        with torch.no_grad():
            state_t = torch.tensor([_normalize(state)], dtype=torch.float32)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(_normalize(state), action, reward, _normalize(next_state), done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.tensor(states)
        actions_t = torch.tensor(actions).unsqueeze(1)
        rewards_t = torch.tensor(rewards)
        next_states_t = torch.tensor(next_states)
        dones_t = torch.tensor(dones)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            td_target = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self, episode):
        frac = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(Path(path), map_location="cpu")
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.epsilon = 0.0  # 불러온 모델은 탐험 없이 greedy하게 사용
        return self


def train(env, episodes=300, verbose_every=20):
    agent = DQNAgent()
    episode_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        agent.decay_epsilon(ep)
        episode_rewards.append(total_reward)

        if verbose_every and (ep + 1) % verbose_every == 0:
            avg = np.mean(episode_rewards[-verbose_every:])
            print(f"[DQN] episode {ep + 1}/{episodes}  avg_reward={avg:.2f}  "
                  f"epsilon={agent.epsilon:.3f}  buffer_size={len(agent.replay_buffer)}")

    return agent, episode_rewards


if __name__ == "__main__":
    from crossway_environment import CrosswayEnvironment

    env = CrosswayEnvironment(preset="normal", seed=0)
    trained_agent, rewards = train(env, episodes=300)
    print(f"\n최종 100 episode 평균 reward: {np.mean(rewards[-100:]):.2f}")
