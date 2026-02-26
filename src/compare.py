"""
Compare training curves from checkpoints (PPO, ZoomingPPO, ContextualZoomingPPO).

Usage:
    python src/compare.py checkpoints/ppo.pt checkpoints/zooming_ppo.pt
    python src/compare.py checkpoints/ppo.pt checkpoints/zooming_ppo.pt checkpoints/contextual_zooming_ppo.pt
    python src/compare.py checkpoints/ppo.pt checkpoints/zooming_ppo.pt --labels "PPO" "Zooming"
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
DEFAULT_LABELS = ["PPO (discrete)", "Zooming PPO (adaptive)", "Contextual Zooming PPO"]


def rolling_mean(data, window=50):
    """Compute rolling mean with given window size."""
    if len(data) < window:
        window = max(1, len(data))
    cumsum = np.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    # First `window` elements get a growing-window average
    result = np.empty(len(data))
    for i in range(len(data)):
        w = min(i + 1, window)
        result[i] = cumsum[i] / w
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare training curves from checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files to compare")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window")
    parser.add_argument("--output", default="plots/comparison.png", help="Output plot path")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each curve (defaults to built-in names)")
    args = parser.parse_args()

    labels = args.labels or DEFAULT_LABELS[:len(args.checkpoints)]
    if len(labels) < len(args.checkpoints):
        labels += [Path(cp).stem for cp in args.checkpoints[len(labels):]]

    # Load and summarize
    all_rewards = []
    for cp, label in zip(args.checkpoints, labels):
        data = torch.load(cp, weights_only=False)
        rewards = np.array(data["episode_rewards"])
        all_rewards.append(rewards)
        print(f"{label}: {len(rewards)} episodes, "
              f"final mean(last 50)={np.mean(rewards[-50:]):.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (rewards, label) in enumerate(zip(all_rewards, labels)):
        color = COLORS[i % len(COLORS)]
        smooth = rolling_mean(rewards, args.window)
        #ax.plot(rewards, alpha=0.08, color=color)
        ax.plot(smooth, label=label, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
