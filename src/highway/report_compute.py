"""
Report training wall-clock and throughput per arm from the timing fields
written into each checkpoint (``train_seconds`` = training compute,
excluding the periodic deterministic-eval episodes; plus throughput in
env-steps/second). Prints a text table and an optional LaTeX row block.

Usage:
    python src/highway/report_compute.py --checkpoints-dir checkpoints/dmcs/walker-walk/efficiency
    python src/highway/report_compute.py --checkpoints-dir checkpoints/highway/efficiency --latex
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.highway.eval_utils import load_total_steps, load_train_seconds

LABEL_RE = re.compile(r"(uniform|zooming)_n(\d+)_seed\d+\.pt$")
SAC_RE = re.compile(r"sac_seed\d+\.pt$")


def _key(name: str):
    m = LABEL_RE.search(name)
    if m:
        return (m.group(1), int(m.group(2)))
    if SAC_RE.search(name):
        return ("sac", None)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=Path, required=True)
    p.add_argument("--latex", action="store_true", help="also emit LaTeX rows")
    args = p.parse_args()

    groups = defaultdict(list)   # (arm, N) -> list of (seconds, steps)
    for pth in sorted(args.checkpoints_dir.glob("*.pt")):
        k = _key(pth.name)
        if k is None:
            continue
        ts = load_train_seconds(pth)
        if ts:
            groups[k].append((ts, load_total_steps(pth)))

    if not groups:
        print(f"No timing fields found under {args.checkpoints_dir}. "
              f"(Checkpoints must be written after compute tracking was added.)")
        return

    def sortk(item):
        (arm, n), _ = item
        return ({"uniform": 0, "zooming": 1, "sac": 2}[arm], n if n else 0)

    print(f"\nTraining wall-clock under {args.checkpoints_dir}")
    print(f"  {'arm':<8} {'N':>4}  {'runs':>4}  {'min/run':>9}  {'steps/s':>9}")
    rows = []
    for (arm, n), vals in sorted(groups.items(), key=sortk):
        secs = np.array([v[0] for v in vals], dtype=float)
        sps = np.array([v[1] / v[0] for v in vals if v[1]], dtype=float)
        mins = secs.mean() / 60.0
        sps_m = sps.mean() if sps.size else float("nan")
        nstr = str(n) if n is not None else "-"
        print(f"  {arm:<8} {nstr:>4}  {len(vals):>4}  {mins:>9.1f}  {sps_m:>9.0f}")
        rows.append((arm, nstr, len(vals), mins, sps_m))

    if args.latex:
        print("\n% LaTeX rows: arm & N & runs & min/run & steps/s")
        for arm, nstr, k, mins, sps_m in rows:
            print(f"  {arm} & {nstr} & {k} & {mins:.1f} & {sps_m:.0f} \\\\")


if __name__ == "__main__":
    main()
