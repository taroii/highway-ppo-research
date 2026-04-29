"""
DQN with a factored adaptive zooming grid on DMCS tasks.

Per-axis 1-D zooming trees + branching Q-net (Tavakoli, Pardo, Kormushev,
"Action Branching Architectures for Deep Reinforcement Learning",
AAAI 2018).  A single global cell budget caps the *sum* of bins across
axes, so the algorithm can spend more cells on important axes (the
ones the policy visits often) and less on quiet ones.

For ``da == 1`` (cartpole-swingup), the factored grid reduces to a
single 1-D zooming tree — equivalent to the joint formulation but with
a corrected multi-split warm-start.

Matched-budget contract with ``run_uniform.py``:
    pass the same ``--n_actions`` to both.  Zooming sets
    ``total_budget = n_actions * da``; uniform pins each axis to
    ``n_actions`` bins.  Both arms therefore have the same total cell
    count; only the placement differs (adaptive vs fixed even grid).

Usage:
    python src/dmcs/run_zooming.py --task walker-walk
    python src/dmcs/run_zooming.py --task cheetah-run --n_actions 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.dmcs.env import make_dmcs_env
from src.highway.dqn import UCB
from src.highway.dqn_factored import BranchingDQN
from src.highway.zooming_factored import FactoredActionZooming


def main(task: str = "walker-walk", seed: int = 42,
         init_depth: int = 1, n_actions: int = 16,
         total_timesteps: int = 300_000, output: str | None = None) -> dict:
    if output is None:
        output = f"checkpoints/dmcs/{task}/zooming_n{n_actions}_seed{seed}.pt"

    env = make_dmcs_env(task)
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
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        buffer_capacity=1_000_000,
        learning_starts=10_000,
        target_update_freq=2,
        split_check_freq=2000,
        split_delay=int(0.2 * total_timesteps),
        seed=seed,
    )

    print(f"Zooming DQN on dm_control/{task}  da={action_dim}  "
          f"init_depth={init_depth} (start={grid.total_cells} cells)  "
          f"n={n_actions} -> total_budget={total_budget}  "
          f"seed={seed}  steps={total_timesteps}")
    rewards = agent.learn(total_timesteps=total_timesteps, print_every=10_000)
    agent.save(output, rewards)

    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        ep = 0.0
        done = truncated = False
        while not (done or truncated):
            obs, r, done, truncated, _ = env.step(agent.predict(obs, deterministic=True))
            ep += r
        eval_rewards.append(ep)
    eval_mean, eval_std = float(np.mean(eval_rewards)), float(np.std(eval_rewards))
    print(f"Eval(10): mean={eval_mean:.2f}  std={eval_std:.2f}")
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
    p.add_argument("--task", type=str, default="walker-walk",
                   help="DMCS task slug; cartpole-swingup also valid (da=1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_depth", type=int, default=1,
                   help="Starting bins per axis = 2^init_depth.  Default 1 "
                        "(2 bins per axis: 'go left vs go right') leaves the "
                        "algorithm maximum room for adaptive splits.  "
                        "Set higher for a warmer start at the cost of "
                        "fewer split decisions.")
    p.add_argument("--n_actions", type=int, default=16,
                   help="Matched-budget partner for run_uniform.py: "
                        "total_budget = n_actions * da.  Default 16.")
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    main(task=args.task, seed=args.seed, init_depth=args.init_depth,
         n_actions=args.n_actions, total_timesteps=args.total_timesteps,
         output=args.output)
