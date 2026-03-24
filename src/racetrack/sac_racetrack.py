"""
SAC baseline for the HighwayEnv racetrack-v0 task.

Reuses the SAC agent from sac.py with a racetrack-specific environment
factory.  The racetrack is a continuous lane-keeping / obstacle-avoidance
task on a curved track with steering-only control.

Key differences from the highway SAC baseline:
  - Uses the built-in racetrack reward (lane centering, collision penalty,
    action penalty, on-road multiplier) instead of CustomRewardWrapper.
  - Default OccupancyGrid observation (2, 12, 12) = 288-dim flat.
  - Hyperparameters tuned for longer episodes (~300 steps).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highway"))

from sac import SAC


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_racetrack_env():
    """Create racetrack-v0 with continuous steering (no acceleration)."""
    env = gym.make(
        "racetrack-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
        },
    )
    return env  # built-in reward — no wrapper needed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 500_000
    env = make_racetrack_env()

    agent = SAC(
        env,
        hidden_dim=256,
        gamma=0.9,
        tau=0.005,
        actor_lr=5e-4,
        critic_lr=5e-4,
        alpha_lr=1e-4,
        batch_size=256,
        buffer_capacity=300_000,
        learning_starts=5000,
        actor_update_freq=2,
        critic_target_update_freq=2,
        init_temperature=0.1,
        seed=42,
    )

    print("Starting SAC training on racetrack-v0...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/sac_racetrack.pt", episode_rewards)

    # Evaluate
    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        eval_rewards.append(total_reward)

    print(f"Eval over 20 episodes: mean={np.mean(eval_rewards):.2f}, std={np.std(eval_rewards):.2f}")
    env.close()
