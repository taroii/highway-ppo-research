"""
Bisimulation SAC for the HighwayEnv racetrack-v0 task.

Reuses the BisimSAC agent from continuous_bisim.py with a racetrack-specific
environment factory.  The racetrack is a continuous lane-keeping /
obstacle-avoidance task on a curved track with steering-only control.

Key differences from the highway bisimulation SAC:
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

from continuous_bisim import BisimSAC


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

    agent = BisimSAC(
        env,
        feature_dim=25,
        hidden_dim=256,
        gamma=0.99, # 0.9 to make more myopic
        tau=0.005,
        encoder_tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        encoder_lr=1e-3,
        alpha_lr=1e-3,
        transition_lr=1e-3,
        batch_size=256,
        buffer_capacity=300_000, # paper uses 100k, this gives us more initial exploration 
        learning_starts=5000, # paper uses 1k, also gives us more time to explore.
        actor_update_freq=2,
        critic_target_update_freq=2,
        bisim_coef=0.5,
        init_temperature=0.01,
        seed=42,
    )

    print("Starting Bisimulation SAC training on racetrack-v0...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/bisim_sac_racetrack.pt", episode_rewards)

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
