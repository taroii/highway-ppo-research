"""
DQN with an adaptive zooming action grid.

Per-axis 1-D zooming trees + branching Q-net (factored grid).  For
racetrack (``da == 1``) this is equivalent to the joint formulation —
one tree, one Q-head — but uses the same agent core as DMCS so the
two task families share code.  See ``src/highway/zooming_factored.py``
for the budget-aware split logic.

The action budget is ``total_budget = n_actions * da`` (matched against
``run_uniform.py`` at the same ``n_actions``).  Starting resolution is
``2 ** init_depth`` bins per axis, then splits refine.

Usage:
    python src/highway/run_zooming.py
    python src/highway/run_zooming.py --n_actions 32 --seed 43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.highway.dqn import UCB
from src.highway.dqn_factored import BranchingDQN
from src.highway.env import make_racetrack_env
from src.highway.zooming_factored import FactoredActionZooming


def main(seed: int = 42, init_depth: int = 3, n_actions: int = 16,
         total_timesteps: int = 150_000, output: str | None = None) -> dict:
    if output is None:
        output = f"checkpoints/highway/zooming_n{n_actions}_seed{seed}.pt"

    env = make_racetrack_env()
    action_dim = int(np.prod(env.action_space.shape))
    total_budget = n_actions * action_dim

    grid = FactoredActionZooming(
        da=action_dim,
        init_depth=init_depth,
        total_budget=total_budget,
    )
    agent = BranchingDQN(
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

    print(f"Zooming DQN on racetrack-v0  da={action_dim}  "
          f"init_depth={init_depth} (start={grid.total_cells} cells)  "
          f"n={n_actions} -> total_budget={total_budget}  "
          f"seed={seed}  steps={total_timesteps}")
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
    print(f"Final n_per_axis: {grid.n_per_axis()}  "
          f"total_cells: {grid.total_cells}/{total_budget}  "
          f"total_splits: {agent.total_splits}")
    env.close()
    return {"output": output, "eval_mean": eval_mean, "eval_std": eval_std,
            "n_per_axis_final": grid.n_per_axis(),
            "total_cells_final": grid.total_cells,
            "total_budget": total_budget,
            "total_splits": agent.total_splits,
            "episode_rewards": rewards}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_depth", type=int, default=3,
                   help="Starting bins per axis = 2^init_depth.  Default 3 "
                        "(8 bins per axis) preserves the racetrack experimental "
                        "setup; lower values give the algorithm more split "
                        "decisions at the cost of coarser early performance.")
    p.add_argument("--n_actions", type=int, default=16,
                   help="Matched-budget partner for run_uniform.py: "
                        "total_budget = n_actions * da.  Default 16.")
    p.add_argument("--total_timesteps", type=int, default=150_000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    main(seed=args.seed, init_depth=args.init_depth, n_actions=args.n_actions,
         total_timesteps=args.total_timesteps, output=args.output)
