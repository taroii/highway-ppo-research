"""
Compare training curves from two checkpoints (PPO vs ZoomingPPO).

Usage:
    python src/compare.py checkpoints/ppo.pt checkpoints/zooming_ppo.pt
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Compare training curves from two checkpoints")
    parser.add_argument("checkpoint1", help="First checkpoint .pt file")
    parser.add_argument("checkpoint2", help="Second checkpoint .pt file")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window")
    parser.add_argument("--output", default="plots/comparison.png", help="Output plot path")
    parser.add_argument("--labels", nargs=2, default=["PPO (discrete)", "Zooming PPO (adaptive)"],
                        help="Labels for the two curves")
    args = parser.parse_args()

    # Load episode rewards from checkpoints
    data1 = torch.load(args.checkpoint1, weights_only=False)
    data2 = torch.load(args.checkpoint2, weights_only=False)

    rewards1 = np.array(data1["episode_rewards"])
    rewards2 = np.array(data2["episode_rewards"])

    print(f"{args.labels[0]}: {len(rewards1)} episodes, "
          f"final mean(last 50)={np.mean(rewards1[-50:]):.2f}")
    print(f"{args.labels[1]}: {len(rewards2)} episodes, "
          f"final mean(last 50)={np.mean(rewards2[-50:]):.2f}")

    # Compute rolling averages
    smooth1 = rolling_mean(rewards1, args.window)
    smooth2 = rolling_mean(rewards2, args.window)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards1, alpha=0.15, color="tab:blue")
    ax.plot(smooth1, label=args.labels[0], color="tab:blue", linewidth=2)

    ax.plot(rewards2, alpha=0.15, color="tab:orange")
    ax.plot(smooth2, label=args.labels[1], color="tab:orange", linewidth=2)

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
