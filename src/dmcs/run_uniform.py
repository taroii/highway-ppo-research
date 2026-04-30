"""
DQN with a factored uniform action grid on DMCS tasks.

Each axis has ``n_actions`` bins independently; total cells = ``n * da``
(additive in ``da``, sidesteps the ``n^da`` Cartesian blowup).  For
``da == 1`` (cartpole-swingup) this is equivalent to a single 1-D grid
of ``n`` cells.

Apples-to-apples partner for src/dmcs/run_zooming.py: same
``BranchingDQN`` core, same Q-net topology, same TD target -- only the
grid is non-adaptive (fixed bins, no splits).

Matched-budget contract with ``run_zooming.py``:
    pass the same ``--n_actions`` to both.  Zooming uses it as
    ``total_budget = n_actions * da`` (a global cap on total cells
    across axes); uniform uses it as the per-axis bin count (also
    ``n_actions * da`` total cells, fixed).  So both arms have the
    same total cell count; the only difference is whether those
    cells are placed adaptively (zooming) or on a fixed even grid
    (uniform).

Usage:
    python src/dmcs/run_uniform.py --task walker-walk
    python src/dmcs/run_uniform.py --task cheetah-run --n_actions 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.dmcs.env import make_dmcs_env
from src.highway.dqn import EpsGreedy
from src.highway.dqn_factored import BranchingDQN
from src.highway.uniform_grid_factored import FactoredUniformActionGrid


def main(task: str = "walker-walk", seed: int = 42, n_actions: int = 16,
         total_timesteps: int = 300_000, output: str | None = None) -> dict:
    if output is None:
        output = f"checkpoints/dmcs/{task}/uniform_n{n_actions}_seed{seed}.pt"

    env = make_dmcs_env(task)
    action_dim = int(np.prod(env.action_space.shape))

    grid = FactoredUniformActionGrid(da=action_dim, n=n_actions)
    agent = BranchingDQN(
        env=env,
        grid=grid,
        selection_policy=EpsGreedy(eps_start=1.0, eps_end=0.05,
                                   decay_steps=int(0.4 * total_timesteps)),
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        buffer_capacity=1_000_000,
        learning_starts=10_000,
        target_update_freq=2,
        split_check_freq=10**9,   # uniform never splits
        split_delay=10**9,
        seed=seed,
    )

    print(f"Uniform DQN on dm_control/{task}  da={action_dim}  "
          f"n={n_actions} (per axis -> {n_actions * action_dim} total cells)  "
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
    env.close()
    return {"output": output, "eval_mean": eval_mean, "eval_std": eval_std,
            "n_actions": n_actions, "n_per_axis": grid.n_per_axis(),
            "episode_rewards": rewards}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="walker-walk",
                   help="DMCS task slug.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_actions", type=int, default=16,
                   help="Bins per action axis.  Total cells = n_actions * da.  "
                        "Match the same arg on run_zooming.py for the budget A/B.")
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    main(task=args.task, seed=args.seed, n_actions=args.n_actions,
         total_timesteps=args.total_timesteps, output=args.output)
