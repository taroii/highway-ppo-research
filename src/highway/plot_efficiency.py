"""
Efficiency + ceiling plot (the headline figure of the new framing).

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

from src.highway.eval_utils import final_eval

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


def _mean_stderr(vals: List[float]):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=0) / np.sqrt(a.size))


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
        means = np.array([_mean_stderr(data[n])[0] for n in ns])
        errs = np.array([_mean_stderr(data[n])[1] for n in ns])
        seeds = [len(data[n]) for n in ns]
        ax.errorbar(ns, means, yerr=errs, color=color, marker=marker,
                    linewidth=2, capsize=4,
                    label=f"{arm} (seeds {min(seeds)}-{max(seeds)})")
        for n, m, e, k in zip(ns, means, errs, seeds):
            summary.append(f"  {arm:<8} N={n:>3} seeds={k} eval={m:>8.2f} +/- {e:>5.2f}")

    if sac_vals:
        s_mean, s_err = _mean_stderr(sac_vals)
        if np.isfinite(s_mean):
            ax.axhline(s_mean, color="tab:green", linestyle="--", linewidth=2,
                       label=f"SAC ceiling (seeds {len([v for v in sac_vals if np.isfinite(v)])})")
            ax.axhspan(s_mean - s_err, s_mean + s_err, color="tab:green", alpha=0.12)
            summary.append(f"  {'sac':<8} (no N)      eval={s_mean:>8.2f} +/- {s_err:>5.2f}")

    all_ns = sorted({n for d in (uni, zoo) for n in d})
    if all_ns:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_ns)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Action budget N (per axis; log scale)")
    ax.set_ylabel("Deterministic-eval reward")
    ax.set_title(args.title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print("\nEfficiency summary (deterministic eval, mean +/- stderr across seeds):")
    for line in summary:
        print(line)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
