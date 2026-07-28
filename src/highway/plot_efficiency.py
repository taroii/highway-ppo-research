"""
Efficiency + ceiling plot. Produces the paper's efficiency figure
(Sec. 4, "Efficiency: deterministic-evaluation return versus action
budget N") and the per-(arm, N) IQM values behind the summary table.

Deterministic-eval reward vs action budget N for both discretized arms,
with SAC drawn as a flat reference band (it has no N).  Reads three ways
in one figure:

  - matched at each N (uniform vs zooming point-for-point);
  - the N at which zooming reaches uniform's best plateau (efficiency);
  - each arm's best-over-N envelope vs the SAC ceiling.

This merges the old ``action_sweep`` and ``architectures`` phases: rather
than picking one N and doing a three-bar comparison, SAC is overlaid on
the full performance-vs-N curve.

Family-agnostic: point ``--checkpoints-dir`` at a directory containing

    uniform_n{N}_seed{S}.pt
    zooming_n{N}_seed{S}.pt
    sac_seed{S}.pt            (optional -> reference band)

Reward is read via ``eval_utils.final_eval`` (deterministic-eval curve;
falls back to the online tail only for legacy checkpoints).

Usage:
    python src/highway/plot_efficiency.py \
        --checkpoints-dir checkpoints/dmcs/walker-walk/efficiency \
        --output plots/dmcs/walker-walk_efficiency.png \
        --title "Efficiency -- dm_control/walker-walk"
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

from src.highway.eval_utils import final_eval, load_total_steps, load_train_seconds
from src.highway.rliable_utils import iqm_ci

UNIFORM_RE = re.compile(r"uniform_n(\d+)_seed(\d+)\.pt$")
ZOOMING_RE = re.compile(r"zooming_n(\d+)_seed(\d+)\.pt$")
SAC_RE = re.compile(r"sac_seed(\d+)\.pt$")


def _by_n(paths, regex) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = defaultdict(list)
    for p in paths:
        m = regex.search(p.name)
        if m:
            out[int(m.group(1))].append(final_eval(p))
    return out


def _arm_of(name: str) -> str:
    for a in ("uniform", "zooming", "sac"):
        if name.startswith(a):
            return a
    return "?"


def _compute_summary(paths) -> List[str]:
    """Per-arm mean training wall-clock and throughput, from the timing
    fields in the checkpoints (blank for legacy checkpoints without them)."""
    agg: Dict[str, List] = defaultdict(list)
    for p in paths:
        ts, st = load_train_seconds(p), load_total_steps(p)
        if ts:
            agg[_arm_of(p.name)].append((ts, st))
    out = []
    for arm in ("uniform", "zooming", "sac"):
        rows = agg.get(arm, [])
        if not rows:
            continue
        secs = np.array([r[0] for r in rows], dtype=float)
        sps = np.array([r[1] / r[0] for r in rows if r[1]], dtype=float)
        thr = f"  ({sps.mean():.0f} steps/s)" if sps.size else ""
        out.append(f"  {arm:<8} {secs.mean() / 60:6.1f} min/run "
                   f"(n={len(rows)}){thr}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="Efficiency + ceiling")
    args = p.parse_args()

    paths = sorted(args.checkpoints_dir.glob("*.pt"))
    if not paths:
        print(f"No checkpoints under {args.checkpoints_dir}.")
        return

    uni = _by_n(paths, UNIFORM_RE)
    zoo = _by_n(paths, ZOOMING_RE)
    sac_vals = [final_eval(p) for p in paths if SAC_RE.search(p.name)]

    fig, ax = plt.subplots(figsize=(8, 5))
    summary: List[str] = []

    for arm, data, color, marker in [("uniform", uni, "tab:blue", "o"),
                                     ("zooming", zoo, "tab:orange", "s")]:
        ns = sorted(data.keys())
        if not ns:
            continue
        pts, los, his, seeds = [], [], [], []
        for n in ns:
            m, lo, hi = iqm_ci(data[n])
            pts.append(m); los.append(lo); his.append(hi); seeds.append(len(data[n]))
        pts, los, his = np.array(pts), np.array(los), np.array(his)
        yerr = np.vstack([pts - los, his - pts])   # asymmetric bootstrap CI
        ax.errorbar(ns, pts, yerr=yerr, color=color, marker=marker,
                    linewidth=2, capsize=4,
                    label=f"{arm} (IQM, {min(seeds)}-{max(seeds)} seeds)")
        for n, m, lo, hi, k in zip(ns, pts, los, his, seeds):
            summary.append(f"  {arm:<8} N={n:>3} seeds={k}  "
                           f"IQM={m:>8.2f}  95% CI [{lo:>7.1f}, {hi:>7.1f}]")

    if sac_vals:
        m, lo, hi = iqm_ci(sac_vals)
        if np.isfinite(m):
            n_sac = len([v for v in sac_vals if np.isfinite(v)])
            ax.axhline(m, color="tab:green", linestyle="--", linewidth=2,
                       label=f"SAC (IQM, {n_sac} seeds)")
            ax.axhspan(lo, hi, color="tab:green", alpha=0.12)
            summary.append(f"  {'sac':<8} (no N)     seeds={n_sac}  "
                           f"IQM={m:>8.2f}  95% CI [{lo:>7.1f}, {hi:>7.1f}]")

    all_ns = sorted({n for d in (uni, zoo) for n in d})
    if all_ns:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_ns)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Action budget N (per axis; log scale)")
    ax.set_ylabel("Deterministic-eval reward (IQM)")
    ax.set_title(args.title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print("\nEfficiency summary (deterministic eval; IQM + 95% bootstrap CI across seeds):")
    for line in summary:
        print(line)
    compute = _compute_summary(paths)
    if compute:
        print("\nTraining wall-clock (excludes periodic eval):")
        for line in compute:
            print(line)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
