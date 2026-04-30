"""
Zooming PPO for the HighwayEnv racetrack-v0 task.

Reuses the ZoomingPPO agent from zooming_ppo.py with a racetrack-specific
environment factory.  Adaptive action discretization of the 1D continuous
steering space on a curved track.

Key differences from the highway Zooming PPO baseline:
  - Uses the built-in racetrack reward (lane centering, collision penalty,
    action penalty, on-road multiplier) instead of CustomRewardWrapper.
  - Default OccupancyGrid observation (2, 12, 12) = 288-dim flat.
  - Hyperparameters tuned for longer episodes (~1500 steps).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highway"))

import numpy as np
import gymnasium as gym

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

from zooming_ppo import ZoomingPPO


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
    return env  # built-in reward -- no wrapper needed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 50_000
    env = make_racetrack_env()

    agent = ZoomingPPO(
        env,
        hidden=[256, 256],
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        lr=3e-4,
        gamma=0.9,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        seed=42,
    )

    print("Starting Zooming PPO training on racetrack-v0...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/zooming_ppo_racetrack.pt", episode_rewards)

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
    print(f"Final action count: {agent.zooming.n_actions}")

    env.close()
