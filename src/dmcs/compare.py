from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch


SEED_RE = re.compile(r"_seed(\d+)\.pt$")
LABEL_N_RE = re.compile(r"_n(\d+)(?:_|$)")


def _group_key(stem: str) -> str:
    """`uniform_n16_seed42` -> `uniform_n16`."""
    return re.sub(r"_seed\d+$", "", stem)


def _label_n(label: str) -> Optional[int]:
    """`uniform_n16` -> 16; `sac` -> None (no action-budget token)."""
    m = LABEL_N_RE.search(label)
    return int(m.group(1)) if m else None


def discover(checkpoints_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(checkpoints_dir.glob("*.pt")):
        if not SEED_RE.search(p.name):
            continue
        groups[_group_key(p.stem)].append(p)
    return groups


def filter_by_n(groups: Dict[str, List[Path]],
                n_actions: Optional[int]) -> Dict[str, List[Path]]:
    """Drop groups whose `_n<N>` token disagrees with ``n_actions``.

    Groups without an `_n<N>` token (e.g. ``sac``) are always kept.
    """
    if n_actions is None:
        return groups
    return {label: paths for label, paths in groups.items()
            if _label_n(label) in (None, n_actions)}


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    window = min(window, len(x))
    cumsum = np.cumsum(x)
    out = np.empty(len(x))
    for i in range(len(x)):
        w = min(i + 1, window)
        lo = max(0, i - w + 1)
        out[i] = (cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0)) / w
    return out


def load_rewards(path: Path) -> np.ndarray:
    data = torch.load(path, weights_only=False)
    return np.asarray(data.get("episode_rewards", []), dtype=np.float32)


def aggregate(group_paths: List[Path], window: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (mean smooth curve, stderr curve, min episode count)."""
    smooths = [rolling_mean(load_rewards(p), window) for p in group_paths]
    smooths = [s for s in smooths if len(s) > 0]
    if not smooths:
        return np.array([]), np.array([]), 0
    n_min = min(len(s) for s in smooths)
    stack = np.stack([s[:n_min] for s in smooths])     # (seeds, episodes)
    mean = stack.mean(axis=0)
    stderr = stack.std(axis=0, ddof=0) / np.sqrt(max(1, len(smooths)))
    return mean, stderr, n_min


PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def assign_colors(labels: List[str]) -> Dict[str, str]:
    """Assign a unique color per label by position in the sorted list."""
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(sorted(labels))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cartpole-swingup",
                   help="DMCS task slug (used for default paths and title).")
    p.add_argument("--checkpoints-dir", type=Path, default=None,
                   help="Defaults to checkpoints/dmcs/{task}/.")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--n_actions", type=int, default=None,
                   help="If set, only plot groups whose `_n<N>` token equals "
                        "this value. Groups without an `_n<N>` token (e.g. sac) "
                        "are always kept.")
    p.add_argument("--output", type=Path, default=None,
                   help="Defaults to plots/dmcs/{task}.png.")
    p.add_argument("--title", type=str, default=None,
                   help="Defaults to 'Architectures -- dm_control/{task}'.")
    args = p.parse_args()

    ckpt_dir = args.checkpoints_dir or Path(f"checkpoints/dmcs/{args.task}")
    out_path = args.output or Path(f"plots/dmcs/{args.task}.png")
    title = args.title or f"Architectures -- dm_control/{args.task}"

    groups = discover(ckpt_dir)
    if not groups:
        print(f"No checkpoints found under {ckpt_dir}. "
              f"Run src/dmcs/run_*.py first.")
        return

    if args.n_actions is not None:
        before = set(groups)
        groups = filter_by_n(groups, args.n_actions)
        dropped = sorted(before - set(groups))
        if dropped:
            print(f"Filtering to n_actions={args.n_actions}; "
                  f"dropped {len(dropped)} group(s): {', '.join(dropped)}")
        if not groups:
            print(f"No groups remain after filtering to n_actions={args.n_actions}.")
            return

    fig, ax = plt.subplots(figsize=(10, 5))
    summary: List[str] = []
    colors = assign_colors(list(groups.keys()))

    for label, paths in sorted(groups.items()):
        mean, stderr, n_min = aggregate(paths, args.window)
        if n_min == 0:
            continue
        color = colors[label]
        x = np.arange(n_min)
        ax.plot(x, mean, label=f"{label} (seeds={len(paths)})",
                color=color, linewidth=2)
        if len(paths) >= 2:
            ax.fill_between(x, mean - stderr, mean + stderr,
                            color=color, alpha=0.18)

        finals = []
        for path in paths:
            r = load_rewards(path)
            finals.append(np.mean(r[-50:]) if len(r) >= 50 else np.mean(r) if len(r) else np.nan)
        finals = np.asarray(finals, dtype=np.float32)
        summary.append(
            f"  {label:<28}  seeds={len(paths)}  "
            f"final-50 mean={np.nanmean(finals):>7.2f}  "
            f"stddev_across_seeds={np.nanstd(finals):>6.2f}"
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode reward (rolling mean, window={args.window})")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("\nSummary (per group, aggregated across seeds):")
    for line in summary:
        print(line)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
