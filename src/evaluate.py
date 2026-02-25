"""
Evaluate a trained PPO or ZoomingPPO agent: run episodes, record videos
for the best and worst episodes.

Usage:
    python src/evaluate.py checkpoints/ppo.pt --type ppo
    python src/evaluate.py checkpoints/zooming_ppo.pt --type zooming
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo import PPO, make_highway_env
from zooming_ppo import ZoomingPPO, make_highway_env_continuous


def run_episode(agent, env, agent_type: str):
    """Run one episode, return total reward."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = truncated = False
    while not (done or truncated):
        if agent_type == "ppo":
            action = agent.predict(obs, deterministic=True)
        else:
            action = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward


def make_env(agent_type: str, render_mode=None):
    """Create the right env type for the agent."""
    if agent_type == "ppo":
        config = {
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": 5,
            },
        }
    else:
        config = {
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
        }

    kwargs = {"config": config}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make("highway-fast-v0", **kwargs)


def record_episode(agent, agent_type: str, video_folder: str, seed: int):
    """Run one episode with video recording."""
    env = make_env(agent_type, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = truncated = False
    while not (done or truncated):
        action = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    env.close()
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent and record videos")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--type", choices=["ppo", "zooming"], required=True,
                        help="Agent type")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    agent_type = args.type

    # Load agent
    env = make_env(agent_type)
    if agent_type == "ppo":
        agent = PPO.load(args.checkpoint, env)
    else:
        agent = ZoomingPPO.load(args.checkpoint, env)

    print(f"Loaded {agent_type} agent from {args.checkpoint}")

    # Run evaluation episodes with different seeds
    results = []
    for i in range(args.episodes):
        seed = 1000 + i
        eval_env = make_env(agent_type)
        eval_env.reset(seed=seed)
        reward = run_episode(agent, eval_env, agent_type)
        results.append((i, seed, reward))
        eval_env.close()
        print(f"  Episode {i+1:>2d}  seed={seed}  reward={reward:.2f}")

    rewards = [r for _, _, r in results]
    print(f"\nSummary: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}, "
          f"min={np.min(rewards):.2f}, max={np.max(rewards):.2f}")

    # Find best and worst
    best = max(results, key=lambda x: x[2])
    worst = min(results, key=lambda x: x[2])

    print(f"\nBest episode:  #{best[0]+1} (seed={best[1]}, reward={best[2]:.2f})")
    print(f"Worst episode: #{worst[0]+1} (seed={worst[1]}, reward={worst[2]:.2f})")

    # Record videos for best and worst
    video_dir = Path("videos")
    for label, (_, seed, reward) in [("best", best), ("worst", worst)]:
        folder = str(video_dir / f"{agent_type}_{label}")
        print(f"\nRecording {label} episode (seed={seed}) to {folder}/")
        recorded_reward = record_episode(agent, agent_type, folder, seed)
        print(f"  Recorded reward: {recorded_reward:.2f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
