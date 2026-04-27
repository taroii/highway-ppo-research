"""
DQN with a fixed uniform action grid (no state-conditioning, no splits).

Usage:
    python src/highway/run_uniform.py
    python src/highway/run_uniform.py --n_actions 32 --seed 43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.highway.action_manager import make_grid
from src.highway.dqn import DQN, EpsGreedy
from src.highway.env import make_racetrack_env


def main(seed: int = 42, n_actions: int = 16, total_timesteps: int = 150_000,
         output: str | None = None) -> dict:
    if output is None:
        output = f"checkpoints/highway/uniform_n{n_actions}_seed{seed}.pt"

    env = make_racetrack_env()
    action_dim = int(np.prod(env.action_space.shape))

    grid = make_grid("uniform", da=action_dim, uniform_n=n_actions)
    agent = DQN(
        env=env,
        grid=grid,
        selection_policy=EpsGreedy(eps_start=1.0, eps_end=0.05,
                                   decay_steps=int(0.4 * total_timesteps)),
        hidden_dim=256,
        gamma=0.9,
        tau=0.01,
        lr=5e-4,
        batch_size=128,
        buffer_capacity=100_000,
        learning_starts=2000,
        target_update_freq=2,
        split_check_freq=10**9,   # uniform never splits
        split_delay=10**9,
        seed=seed,
    )

    print(f"Uniform DQN on racetrack-v0  N={n_actions}  seed={seed}  "
          f"steps={total_timesteps}")
    rewards = agent.learn(total_timesteps=total_timesteps, print_every=5_000)
    agent.save(output, rewards)

    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        ep = 0.0
        done = truncated = False
        while not (done or truncated):
            obs, r, done, truncated, _ = env.step(agent.predict(obs, deterministic=True))
            ep += r
        eval_rewards.append(ep)
    eval_mean, eval_std = float(np.mean(eval_rewards)), float(np.std(eval_rewards))
    print(f"Eval(20): mean={eval_mean:.2f}  std={eval_std:.2f}")
    env.close()
    return {"output": output, "eval_mean": eval_mean, "eval_std": eval_std,
            "n_actions": n_actions, "episode_rewards": rewards}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_actions", type=int, default=16,
                   help="Total actions in the fixed grid (per axis if da>1).")
    p.add_argument("--total_timesteps", type=int, default=150_000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    main(seed=args.seed, n_actions=args.n_actions,
         total_timesteps=args.total_timesteps, output=args.output)
