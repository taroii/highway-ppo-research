"""
Robustness / auto-tuning plot (adjudicates the #2 contribution).

The auto-tuning claim -- "with zooming you don't have to pick N" -- only
survives review if zooming is *less* sensitive to its own knobs (budget
ceiling, buffer_period, init_depth) than uniform is to N.  This figure
puts the two spreads side by side:

  - uniform: the set of per-configuration mean-eval rewards across N;
  - zooming: the set across ALL its knob settings (budget-N reuses the
    efficiency sweep; buffer_period / init_depth variants add to it).

If uniform's spread across N is wide and zooming's spread across its
knobs is narrow, the claim holds.  If not, demote it to a secondary
observation.  The printed summary reports range, std, and coefficient
of variation for each arm so the verdict is quantitative, not eyeballed.

Discovery (pool one or more dirs via repeated --dir):
    uniform_n{N}_seed{S}.pt          -> grouped by N
    zooming_{cfg}_seed{S}.pt         -> grouped by {cfg} (the token between
                                        'zooming_' and '_seed'), e.g.
                                        n32, bp8, bp8_id2, budget48

Usage:
    python src/highway/plot_robustness.py \
        --dir checkpoints/dmcs/walker-walk/efficiency \
        --dir checkpoints/dmcs/walker-walk/robustness \
        --output plots/dmcs/walker-walk_robustness.png \
        --title "Robustness -- dm_control/walker-walk"
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

from src.highway.eval_utils import final_eval

UNIFORM_RE = re.compile(r"uniform_n(\d+)_seed\d+\.pt$")
ZOOMING_RE = re.compile(r"zooming_(.+?)_seed\d+\.pt$")


def _collect(dirs) -> Dict[str, Dict[str, List[float]]]:
    groups = {"uniform": defaultdict(list), "zooming": defaultdict(list)}
    seen = set()
    for d in dirs:
        for p in sorted(Path(d).glob("*.pt")):
            if p.name in seen:
                continue
            seen.add(p.name)
            mu = UNIFORM_RE.search(p.name)
            if mu:
                groups["uniform"][f"N={mu.group(1)}"].append(final_eval(p))
                continue
            mz = ZOOMING_RE.search(p.name)
            if mz:
                groups["zooming"][mz.group(1)].append(final_eval(p))
    return groups


def _group_means(cfg_to_vals: Dict[str, List[float]]):
    """One mean-over-seeds value per configuration."""
    labels, means = [], []
    for label, vals in sorted(cfg_to_vals.items()):
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size:
            labels.append(label)
            means.append(float(a.mean()))
    return labels, np.asarray(means, dtype=np.float64)


def _spread_stats(m: np.ndarray) -> str:
    if m.size < 2:
        return f"n_configs={m.size} (need >=2 for spread)"
    rng = m.max() - m.min()
    cv = m.std(ddof=0) / (abs(m.mean()) + 1e-9)
    return (f"n_configs={m.size}  range={rng:8.2f}  std={m.std(ddof=0):8.2f}  "
            f"CV={cv:5.3f}  [min={m.min():.1f}, max={m.max():.1f}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", action="append", dest="dirs", required=True,
                   help="checkpoint dir to pool (repeat for several)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="Robustness / auto-tuning")
    args = p.parse_args()

    groups = _collect(args.dirs)
    u_labels, u_means = _group_means(groups["uniform"])
    z_labels, z_means = _group_means(groups["zooming"])
    if u_means.size == 0 and z_means.size == 0:
        print(f"No checkpoints found under {args.dirs}.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for x, means, labels, color, name in [
        (1, u_means, u_labels, "tab:blue", "uniform (across N)"),
        (2, z_means, z_labels, "tab:orange", "zooming (across knobs)"),
    ]:
        if means.size == 0:
            continue
        jitter = (np.arange(means.size) - (means.size - 1) / 2) * 0.03
        ax.scatter(np.full(means.size, x) + jitter, means, color=color,
                   s=45, zorder=3, label=name)
        ax.plot([x - 0.18, x + 0.18], [means.mean()] * 2, color=color, lw=2)
        ax.vlines(x, means.min(), means.max(), color=color, alpha=0.4, lw=8)
        for lab, mv, jx in zip(labels, means, jitter):
            ax.annotate(lab, (x + jitter.max() + 0.06, mv), fontsize=6,
                        va="center", color=color, alpha=0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["uniform\n(vary N)", "zooming\n(vary knobs)"])
    ax.set_xlim(0.5, 2.6)
    ax.set_ylabel("Deterministic-eval reward (mean over seeds)")
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")

    print("\nRobustness summary (spread of per-config mean-eval):")
    print(f"  uniform (vary N):      {_spread_stats(u_means)}")
    print(f"  zooming (vary knobs):  {_spread_stats(z_means)}")
    if u_means.size >= 2 and z_means.size >= 2:
        u_rng = u_means.max() - u_means.min()
        z_rng = z_means.max() - z_means.min()
        verdict = ("auto-tuning SUPPORTED (zooming spread < uniform spread)"
                   if z_rng < u_rng else
                   "auto-tuning NOT supported (zooming spread >= uniform spread)")
        print(f"  -> range ratio zoom/uniform = {z_rng / (u_rng + 1e-9):.2f}  =>  {verdict}")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
