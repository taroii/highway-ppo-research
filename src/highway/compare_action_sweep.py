"""
Plot the action-budget sweep: final reward vs action budget N for
uniform and zooming, aggregated across seeds.

The hypothesis the sweep tests is that **zooming's advantage over
uniform grows with N** because uniform pays the full N-output Q-learning
cost from step 1 while zooming bootstraps from a coarser grid (init=8
cubes) and only refines where UCB-driven plays concentrate.

Caveat — at fixed training budget, large N may look bad here purely
because a bigger bandit needs more samples; the companion
``compare_timestep_sweep.py`` separates that confound.

The right shape for that claim is a 2-line plot of *final reward* vs
*action budget*, not stacked reward curves.  This script auto-discovers
checkpoints under ``checkpoints/highway/action_sweep/`` named:

    uniform_n{N}_seed{S}.pt
    zooming_d{D}_seed{S}.pt   (where N == 2**D)

Usage:
    python src/highway/compare_action_sweep.py
    python src/highway/compare_action_sweep.py --window 100
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch


UNIFORM_RE = re.compile(r"uniform_n(\d+)_seed(\d+)\.pt$")
ZOOMING_RE = re.compile(r"zooming_d(\d+)_seed(\d+)\.pt$")


def _load_finals(path: Path, window: int) -> float:
    data = torch.load(path, weights_only=False)
    rewards = np.asarray(data.get("episode_rewards", []), dtype=np.float32)
    if len(rewards) == 0:
        return float("nan")
    w = min(window, len(rewards))
    return float(rewards[-w:].mean())


def _discover(checkpoints_dir: Path, window: int) -> Dict[str, Dict[int, List[float]]]:
    """Return finals[arm][N] = list of per-seed final-mean rewards."""
    finals: Dict[str, Dict[int, List[float]]] = {
        "uniform": defaultdict(list),
        "zooming": defaultdict(list),
    }
    for p in sorted(checkpoints_dir.glob("*.pt")):
        m = UNIFORM_RE.search(p.name)
        if m:
            n = int(m.group(1))
            finals["uniform"][n].append(_load_finals(p, window))
            continue
        m = ZOOMING_RE.search(p.name)
        if m:
            d = int(m.group(1))
            n = 2 ** d
            finals["zooming"][n].append(_load_finals(p, window))
    return finals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=Path,
                   default=Path("checkpoints/highway/action_sweep"))
    p.add_argument("--window", type=int, default=50,
                   help="Average over the last `window` episodes per run.")
    p.add_argument("--output", type=Path,
                   default=Path("plots/highway/action_sweep.png"))
    p.add_argument("--title", type=str,
                   default="Action-budget sweep — racetrack-v0")
    args = p.parse_args()

    finals = _discover(args.checkpoints_dir, args.window)
    if not finals["uniform"] and not finals["zooming"]:
        print(f"No checkpoints found under {args.checkpoints_dir}. "
              f"Run src/highway/run_action_sweep.py --run first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    summary: List[str] = []
    for arm, color, marker in [("uniform", "tab:blue", "o"),
                               ("zooming", "tab:orange", "s")]:
        ns = sorted(finals[arm].keys())
        if not ns:
            continue
        means = np.array([np.nanmean(finals[arm][n]) for n in ns])
        stderrs = np.array([np.nanstd(finals[arm][n], ddof=0)
                            / np.sqrt(max(1, len(finals[arm][n]))) for n in ns])
        seed_counts = [len(finals[arm][n]) for n in ns]
        ax.errorbar(ns, means, yerr=stderrs,
                    color=color, marker=marker, linewidth=2, capsize=4,
                    label=f"{arm} (seeds: {min(seed_counts)}–{max(seed_counts)})")
        for n, m, s, k in zip(ns, means, stderrs, seed_counts):
            summary.append(f"  {arm:<8}  N={n:>3}  seeds={k}  "
                           f"final-{args.window} mean={m:>7.2f} ± {s:>5.2f}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({n for arm in finals for n in finals[arm]}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Action budget N (log scale)")
    ax.set_ylabel(f"Final reward (mean of last {args.window} episodes)")
    ax.set_title(args.title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print("\nFinal reward by arm × N (mean ± stderr across seeds):")
    for line in summary:
        print(line)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
