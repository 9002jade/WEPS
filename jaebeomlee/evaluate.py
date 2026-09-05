"""학습된 정책의 성능 지표를 측정하고 기준선(고정주기 신호)과 비교한다.

측정 지표 (연구계획서 6절):
  1. 평균 대기열 길이
  2. 평균 차량 대기 시간 (Little's Law: 평균 대기열 / 분당 처리량)
  3. 차량 통과량 (분당)
  4. 신호 전환 횟수

사용법:
    python evaluate.py                        # normal 프리셋
    python evaluate.py --preset rush_hour
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crossway_environment import MIN_GREEN_TIME, STAY, SWITCH, CrosswayEnvironment

RESULTS_DIR = Path(__file__).parent / "results"
EVAL_EPISODES = 60
EVAL_SEED_BASE = 10_000  # 학습에 쓰지 않은 seed 대역으로 평가


def fixed_time_policy(hold_seconds):
    """고정주기 신호: hold_seconds마다 기계적으로 다음 페이즈로 전환."""
    def policy(env, state):
        return SWITCH if env.phase_time >= hold_seconds else STAY
    return policy


def dqn_policy(agent):
    def policy(env, state):
        return agent.select_action(state, greedy=True)
    return policy


def evaluate(policy, preset="normal", episodes=EVAL_EPISODES, skew=1.0):
    """skew=1.0이면 남북/동서 교통량이 같고, 클수록 남북에 교통량이 몰린다."""
    avg_queues, throughputs, switches, wait_times = [], [], [], []

    for ep in range(episodes):
        env = CrosswayEnvironment(preset=preset, seed=EVAL_SEED_BASE + ep)
        base = env.arrival_prob
        env.arrival_prob_ns = min(base * skew, 1.0)
        env.arrival_prob_ew = min(base * (2.0 - skew), 1.0)
        state = env.reset()
        queue_samples, n_switch = [], 0
        done = False

        while not done:
            state, _, done, info = env.step(policy(env, state))
            queue_samples.append(info.total_queue)
            n_switch += info.switched

        avg_queue = float(np.mean(queue_samples))
        per_min = env.total_passed / env.cycle_time * 60

        avg_queues.append(avg_queue)
        throughputs.append(per_min)
        switches.append(n_switch)
        # Little's Law: 평균 대기시간 = 평균 대기열 / 처리율
        wait_times.append(avg_queue / (per_min / 60) if per_min > 0 else float("inf"))

    return {
        "평균_대기열": float(np.mean(avg_queues)),
        "평균_대기시간_초": float(np.mean(wait_times)),
        "분당_통과량": float(np.mean(throughputs)),
        "신호_전환횟수": float(np.mean(switches)),
    }


def main():
    parser = argparse.ArgumentParser(description="교차로 신호 제어 성능 평가")
    parser.add_argument("--preset", default="normal", choices=["normal", "rush_hour", "night"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    policies = {
        "고정주기 15초": fixed_time_policy(15),
        "고정주기 20초": fixed_time_policy(20),
        "고정주기 30초": fixed_time_policy(30),
    }

    model_path = Path(args.model) if args.model else RESULTS_DIR / f"dqn_{args.preset}.pt"
    if model_path.exists():
        from dqn_agent import DQNAgent
        policies["DQN"] = dqn_policy(DQNAgent().load(model_path))
    else:
        print(f"[경고] 모델 없음: {model_path} — 기준선만 평가합니다.")

    # skew=1.0은 대칭, 1.6은 남북에 교통량이 몰린 비대칭 상황.
    # 적응형 제어의 이점은 비대칭 상황에서만 드러난다.
    scenarios = [("대칭 교통량", 1.0), ("비대칭 교통량 (남북 편중)", 1.6)]
    results = {}

    for scenario_name, skew in scenarios:
        print(f"\n=== {scenario_name} (preset={args.preset}, {EVAL_EPISODES} episode 평균) ===")
        header = f"{'정책':<14}{'평균대기열':>11}{'평균대기(초)':>13}{'통과(대/분)':>12}{'전환':>7}"
        print(header)
        print("-" * len(header))

        scenario_results = {}
        for name, policy in policies.items():
            m = evaluate(policy, preset=args.preset, skew=skew)
            scenario_results[name] = m
            print(f"{name:<14}{m['평균_대기열']:11.2f}{m['평균_대기시간_초']:13.1f}"
                  f"{m['분당_통과량']:12.1f}{m['신호_전환횟수']:7.1f}")

        if "DQN" in scenario_results:
            baselines = [(r["평균_대기열"], n) for n, r in scenario_results.items() if n != "DQN"]
            best_q, best_name = min(baselines)
            gain = (best_q - scenario_results["DQN"]["평균_대기열"]) / best_q * 100
            verdict = "개선" if gain > 0 else "악화"
            print(f"→ DQN vs 최고 기준선({best_name}): 평균 대기열 {abs(gain):.1f}% {verdict}")

        results[scenario_name] = scenario_results

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"evaluation_{args.preset}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"결과 저장 -> {out}")


if __name__ == "__main__":
    main()
