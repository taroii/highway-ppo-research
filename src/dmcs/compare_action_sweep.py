"""
Plot the DMCS action-budget sweep: final reward vs action budget N for
uniform and zooming, aggregated across seeds.

The hypothesis the sweep tests is that **zooming's advantage over
uniform grows with N** because uniform pays the full ``n * da``-output
Q-learning cost from step 1 while zooming bootstraps from a coarser
grid (init=2 bins per axis at ``init_depth=1``) and only refines where
UCB-driven plays concentrate.

Caveat — at fixed training budget, large N may look bad here purely
because a bigger bandit needs more samples; the timestep-sweep phase
of ``run_dmcs_pipeline.sh`` separates that confound.

Auto-discovers checkpoints under ``checkpoints/dmcs/<task>/action_sweep/``
named:

    uniform_n{N}_seed{S}.pt
    zooming_n{N}_seed{S}.pt

Usage:
    python src/dmcs/compare_action_sweep.py
    python src/dmcs/compare_action_sweep.py --task walker-walk --window 100
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
ZOOMING_RE = re.compile(r"zooming_n(\d+)_seed(\d+)\.pt$")


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
            n = int(m.group(1))
            finals["zooming"][n].append(_load_finals(p, window))
    return finals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cartpole-swingup",
                   help="DMCS task slug (used for default paths and title).")
    p.add_argument("--checkpoints-dir", type=Path, default=None,
                   help="Defaults to checkpoints/dmcs/<task>/action_sweep/.")
    p.add_argument("--window", type=int, default=50,
                   help="Average over the last `window` episodes per run.")
    p.add_argument("--output", type=Path, default=None,
                   help="Defaults to plots/dmcs/<task>_action_sweep.png.")
    p.add_argument("--title", type=str, default=None,
                   help="Defaults to 'Action-budget sweep — dm_control/<task>'.")
    args = p.parse_args()

    ckpt_dir = args.checkpoints_dir or Path(f"checkpoints/dmcs/{args.task}/action_sweep")
    out_path = args.output or Path(f"plots/dmcs/{args.task}_action_sweep.png")
    title = args.title or f"Action-budget sweep — dm_control/{args.task}"

    finals = _discover(ckpt_dir, args.window)
    if not finals["uniform"] and not finals["zooming"]:
        print(f"No checkpoints found under {ckpt_dir}. "
              f"Run src/dmcs/run_action_sweep.py --run first.")
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
    ax.set_xlabel("Action budget N (per axis; log scale)")
    ax.set_ylabel(f"Final reward (mean of last {args.window} episodes)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("\nFinal reward by arm × N (mean ± stderr across seeds):")
    for line in summary:
        print(line)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
