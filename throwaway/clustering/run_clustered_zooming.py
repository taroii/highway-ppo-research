"""
Arm 5 -- DQN with per-cluster adaptive zooming grids.

The proposed method: SAC feature clustering + per-cluster zooming +
UCB action selection.  Each cluster refines its action resolution
independently based on its own play counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.baseline._cluster_bootstrap import build_clusterer
from src.baseline.action_manager import ClusteredActionManager
from src.baseline.dqn import DQN, UCB
from src.baseline.env import make_racetrack_env


TOTAL_TIMESTEPS = 150_000
SEED = 42
K_CLUSTERS = 4
UCB_C_START = 0.3
UCB_C_END = 0.03
UCB_DECAY_STEPS = 60_000
SAC_CHECKPOINT = "checkpoints/baseline/sac.pt"
CHECKPOINT = "checkpoints/baseline/clustered_zooming.pt"


if __name__ == "__main__":
    env = make_racetrack_env()
    action_dim = int(np.prod(env.action_space.shape))

    cluster_fn, _ = build_clusterer(SAC_CHECKPOINT, env, k=K_CLUSTERS, seed=SEED)

    action_manager = ClusteredActionManager(
        k=K_CLUSTERS, da=action_dim, mode="zooming",
    )
    agent = DQN(
        env=env,
        action_manager=action_manager,
        cluster_fn=cluster_fn,
        selection_policy=UCB(c_start=UCB_C_START, c_end=UCB_C_END,
                              decay_steps=UCB_DECAY_STEPS),
        hidden_dim=256,
        gamma=0.9,
        tau=0.01,
        lr=5e-4,
        batch_size=128,
        buffer_capacity=100_000,
        learning_starts=2000,
        target_update_freq=2,
        split_check_freq=2000,
        split_delay=30_000,
        seed=SEED,
    )

    print(f"Clustered Zooming DQN on racetrack-v0 (K={K_CLUSTERS})")
    rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=5_000)
    agent.save(CHECKPOINT, rewards)

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
    print(f"Eval(20): mean={np.mean(eval_rewards):.2f}  std={np.std(eval_rewards):.2f}")
    print(f"Final per-cluster cubes: "
          f"{[action_manager.grids[c].n_actions for c in range(K_CLUSTERS)]}")
    print(f"Total splits: {agent.total_splits}")
    env.close()
