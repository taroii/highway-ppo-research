"""
Aggregate comparison via the ``rliable`` package (Agarwal et al. 2021):
interquartile mean with stratified-bootstrap 95% confidence intervals, a
performance profile, and probability of improvement -- aggregated across
tasks at a matched action budget. Produces the paper's aggregate /
robust-statistics figure (Sec. 4 / appendix).

Requires ``rliable`` (``pip install rliable``); its ``arch`` dependency
currently needs numpy < 2 (see README pin).

Score matrices are assembled at a fixed budget ``--n_actions`` (default
32, shared by the DMCS and highway sweeps) and normalized per task to
[0, 1] so tasks with different return scales can be pooled.

Usage:
    python src/highway/plot_rliable.py --n_actions 32 \
        --dir cartpole=checkpoints/dmcs/cartpole-swingup/efficiency \
        --dir walker=checkpoints/dmcs/walker-walk/efficiency \
        --dir cheetah=checkpoints/dmcs/cheetah-run/efficiency \
        --dir racetrack=checkpoints/highway/efficiency \
        --output plots/aggregate_rliable.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.highway.rliable_utils import (
    build_score_matrices,
    iqm_interval,
    performance_profiles,
    prob_improvement,
)

ARMS = [("uniform", "tab:blue"), ("zooming", "tab:orange"), ("sac", "tab:green")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", action="append", dest="dirs", required=True,
                   help="label=path to a task's efficiency checkpoint dir (repeat)")
    p.add_argument("--n_actions", type=int, default=32)
    p.add_argument("--normalize", choices=["minmax", "sac"], default="minmax")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=50000,
                   help="bootstrap resamples (rliable default 50000)")
    args = p.parse_args()

    task_dirs = {}
    for spec in args.dirs:
        label, path = spec.split("=", 1)
        task_dirs[label] = Path(path)

    mats, tasks, seeds = build_score_matrices(task_dirs, args.n_actions, args.normalize)
    if not mats:
        print("No common seeds across the given task dirs / budget. "
              "Check --n_actions and that the checkpoints exist.")
        return
    present = [(a, c) for a, c in ARMS if a in mats and np.isfinite(mats[a]).any()]
    print(f"Aggregating N={args.n_actions} over tasks={tasks}, "
          f"{len(seeds)} common seeds, normalize={args.normalize}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # ---- Panel 1: aggregate IQM with 95% CI ----
    ax = axes[0]
    intervals = iqm_interval({a: mats[a] for a, _ in present}, reps=args.reps)
    for i, (arm, color) in enumerate(present):
        pt, lo, hi = intervals[arm]
        ax.errorbar([pt], [i], xerr=[[pt - lo], [hi - pt]], fmt="o",
                    color=color, capsize=5, markersize=8)
        ax.text(pt, i + 0.18, f"{pt:.3f}", ha="center", color=color, fontsize=9)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([a for a, _ in present])
    ax.set_xlabel("Normalized score (IQM, 95% CI)")
    ax.set_title("Aggregate performance")
    ax.grid(True, axis="x", alpha=0.3)

    # ---- Panel 2: performance profiles ----
    ax = axes[1]
    taus = np.linspace(0.0, 1.0, 41)
    sd, cis = performance_profiles({a: mats[a] for a, _ in present}, taus,
                                   reps=min(args.reps, 2000))
    for arm, color in present:
        ax.plot(taus, sd[arm], color=color, label=arm, linewidth=2)
        ax.fill_between(taus, cis[arm][0], cis[arm][1], color=color, alpha=0.15)
    ax.set_xlabel(r"Normalized score threshold $\tau$")
    ax.set_ylabel(r"Fraction of runs with score $> \tau$")
    ax.set_title("Performance profiles")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    # ---- Panel 3: probability of improvement ----
    ax = axes[2]
    pairs = []
    if "zooming" in mats and "uniform" in mats:
        pairs.append(("ZoomQ > Uniform", "zooming", "uniform"))
    if "zooming" in mats and "sac" in mats:
        pairs.append(("ZoomQ > SAC", "zooming", "sac"))
    if "uniform" in mats and "sac" in mats:
        pairs.append(("Uniform > SAC", "uniform", "sac"))
    for i, (label, x, y) in enumerate(pairs):
        pt, lo, hi = prob_improvement(mats[x], mats[y], reps=args.reps)
        ax.errorbar([pt], [i], xerr=[[pt - lo], [hi - pt]], fmt="o",
                    color="tab:purple", capsize=5, markersize=8)
        ax.text(pt, i + 0.18, f"{pt:.2f}", ha="center", fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([lab for lab, _, _ in pairs])
    ax.set_xlabel("P(row > column)")
    ax.set_title("Probability of improvement")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Aggregate across {len(tasks)} tasks at N={args.n_actions} "
                 f"({len(seeds)} seeds)", y=1.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
