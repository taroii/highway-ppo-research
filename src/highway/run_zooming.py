"""
DQN with an adaptive zooming action grid.

Initial resolution is ``init_depth`` levels deep (2^init_depth cubes for
1-D actions), and the grid can refine up to ``max_depth`` (cap at
2^max_depth cubes).  Setting ``max_depth == init_depth`` disables
splits (effectively uniform at that depth).

Usage:
    python src/highway/run_zooming.py
    python src/highway/run_zooming.py --max_depth 5 --seed 43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.highway.action_manager import make_grid
from src.highway.dqn import DQN, UCB
from src.highway.env import make_racetrack_env


def main(seed: int = 42, init_depth: int = 3, max_depth: int = 4,
         total_timesteps: int = 150_000, output: str | None = None) -> dict:
    if output is None:
        output = f"checkpoints/highway/zooming_d{max_depth}_seed{seed}.pt"

    env = make_racetrack_env()
    action_dim = int(np.prod(env.action_space.shape))

    grid = make_grid("zooming", da=action_dim,
                     init_depth=init_depth, max_depth=max_depth)
    agent = DQN(
        env=env,
        grid=grid,
        selection_policy=UCB(c_start=0.3, c_end=0.03,
                             decay_steps=int(0.4 * total_timesteps)),
        hidden_dim=256,
        gamma=0.9,
        tau=0.01,
        lr=5e-4,
        batch_size=128,
        buffer_capacity=100_000,
        learning_starts=2000,
        target_update_freq=2,
        split_check_freq=2000,
        split_delay=int(0.2 * total_timesteps),
        seed=seed,
    )

    print(f"Zooming DQN on racetrack-v0  init_depth={init_depth}  "
          f"max_depth={max_depth}  seed={seed}  steps={total_timesteps}")
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
    print(f"Final cubes: {grid.n_actions}  total_splits: {agent.total_splits}")
    env.close()
    return {"output": output, "eval_mean": eval_mean, "eval_std": eval_std,
            "n_actions_final": grid.n_actions, "total_splits": agent.total_splits,
            "episode_rewards": rewards}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_depth", type=int, default=3,
                   help="Pre-split depth: starts with 2^init_depth cubes.")
    p.add_argument("--max_depth", type=int, default=4,
                   help="Maximum split depth: cubes capped at 2^max_depth total.")
    p.add_argument("--total_timesteps", type=int, default=150_000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    main(seed=args.seed, init_depth=args.init_depth, max_depth=args.max_depth,
         total_timesteps=args.total_timesteps, output=args.output)
