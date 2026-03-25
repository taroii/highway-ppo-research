"""
Evaluate a trained agent and record videos for the best and worst episodes.

Supported agent types:
    ppo             - discrete PPO (src/highway/ppo.py)
    sac             - continuous SAC (src/highway/sac.py)
    bisim           - continuous BisimSAC (src/highway/continuous_bisim.py)
    zooming         - Zooming PPO (src/highway/zooming_ppo.py)
    bisim_zooming   - Bisim-guided Clustered Zooming SAC (src/highway/bisim_zooming.py)
    racetrack_ppo   - discrete PPO on racetrack (src/racetrack/ppo_racetrack.py)
    racetrack_sac   - continuous SAC on racetrack (src/racetrack/sac_racetrack.py)
    racetrack_zooming - Zooming PPO on racetrack (src/racetrack/zooming_ppo_racetrack.py)
    racetrack_bisim - BisimSAC on racetrack (src/racetrack/bisim_racetrack.py)

Usage:
    python src/visualize/evaluate.py checkpoints/ppo.pt --type ppo
    python src/visualize/evaluate.py checkpoints/sac.pt --type sac
    python src/visualize/evaluate.py checkpoints/continuous_bisim.pt --type bisim
    python src/visualize/evaluate.py checkpoints/zooming_ppo.pt --type zooming
    python src/visualize/evaluate.py checkpoints/bisim_zooming.pt --type bisim_zooming
    python src/visualize/evaluate.py checkpoints/ppo_racetrack.pt --type racetrack_ppo
    python src/visualize/evaluate.py checkpoints/sac_racetrack.pt --type racetrack_sac
    python src/visualize/evaluate.py checkpoints/zooming_ppo_racetrack.pt --type racetrack_zooming
    python src/visualize/evaluate.py checkpoints/bisim_sac_racetrack.pt --type racetrack_bisim
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highway"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "racetrack"))

from ppo import PPO, make_highway_env, CustomRewardWrapper
from sac import SAC
from zooming_ppo import ZoomingPPO, make_highway_env_continuous
from continuous_bisim import BisimSAC
from bisim_zooming import ClusteredZoomingSAC
from ppo_racetrack import make_racetrack_env as make_racetrack_env_discrete
from sac_racetrack import make_racetrack_env as make_racetrack_env_continuous
from zooming_ppo_racetrack import make_racetrack_env as make_racetrack_env_zooming


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


RACETRACK_TYPES = {"racetrack_ppo", "racetrack_sac", "racetrack_zooming", "racetrack_bisim"}


def make_env(agent_type: str, render_mode=None):
    """Create the right env type for the agent."""
    if agent_type in RACETRACK_TYPES:
        return _make_racetrack_env(agent_type, render_mode)
    return _make_highway_env(agent_type, render_mode)


def _make_highway_env(agent_type: str, render_mode=None):
    if agent_type == "ppo":
        config = {
            "action": {
                "type": "DiscreteAction",
                "longitudinal": False,
                "lateral": True,
                "actions_per_axis": 5,
            },
        }
    else:  # sac, bisim, zooming, or bisim_zooming
        config = {
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
        }

    kwargs = {"config": config}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return CustomRewardWrapper(gym.make("highway-fast-v0", **kwargs))


def _make_racetrack_env(agent_type: str, render_mode=None):
    if agent_type == "racetrack_ppo":
        config = {
            "action": {
                "type": "DiscreteAction",
                "longitudinal": False,
                "lateral": True,
                "actions_per_axis": 5,
            },
        }
    else:  # racetrack_sac, racetrack_zooming, racetrack_bisim
        config = {
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
        }

    kwargs = {"config": config}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make("racetrack-v0", **kwargs)


def record_episode(agent, agent_type: str, video_folder: str, seed: int):
    """Run one episode with video recording."""
    env = make_env(agent_type, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    obs, _ = env.reset(seed=seed)
    # Ensure the underlying highway_env renders offscreen (no pygame window).
    # gymnasium may intercept render_mode via wrappers, leaving the base env
    # with None, and highway_env needs offscreen_rendering=True to avoid
    # opening a display window.
    inner = env.unwrapped
    if inner.render_mode is None:
        inner.render_mode = "rgb_array"
    inner.config["offscreen_rendering"] = True
    inner.render()
    inner.set_record_video_wrapper(env)
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
    parser.add_argument("--type", choices=[
                            "ppo", "sac", "bisim", "zooming", "bisim_zooming",
                            "racetrack_ppo", "racetrack_sac", "racetrack_zooming",
                            "racetrack_bisim",
                        ],
                        required=True, help="Agent type")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    agent_type = args.type

    # Load agent
    env = make_env(agent_type)
    if agent_type in ("ppo", "racetrack_ppo"):
        agent = PPO.load(args.checkpoint, env)
    elif agent_type in ("sac", "racetrack_sac"):
        agent = SAC.load(args.checkpoint, env)
    elif agent_type in ("bisim", "racetrack_bisim"):
        agent = BisimSAC.load(args.checkpoint, env)
    elif agent_type in ("zooming", "racetrack_zooming"):
        agent = ZoomingPPO.load(args.checkpoint, env)
    else:
        agent = ClusteredZoomingSAC.load(args.checkpoint, env)

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
