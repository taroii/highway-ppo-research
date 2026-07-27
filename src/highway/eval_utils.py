"""Deterministic-evaluation helpers shared across all arms.

Two responsibilities:

  1. ``evaluate_policy`` -- run greedy (no-exploration) episodes on a
     dedicated eval env during training, so ``learn()`` can log a
     deterministic-eval curve alongside the online trace.

  2. checkpoint I/O (``load_eval_curve`` / ``load_curve_preferring_eval``
     / ``final_eval``) -- the comparison plotters read the eval curve
     through these, falling back to the online ``episode_rewards`` trace
     only for legacy checkpoints written before the eval hook existed.

Why this matters: ``episode_rewards`` is recorded *with exploration
live* (UCB for zooming, eps-greedy for uniform), so it understates the
greedy policy -- e.g. highway uniform_n32 ends at ~340 online but ~959
under deterministic eval.  Every published comparison should therefore
be read off the eval curve, not the online trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


def evaluate_policy(
    act_fn: Callable[[np.ndarray], np.ndarray],
    env,
    n_episodes: int,
) -> Tuple[float, float]:
    """Run ``n_episodes`` greedy episodes on ``env`` and return
    (mean_return, std_return).

    ``act_fn`` maps a raw observation to a *deterministic* env action.
    ``env`` must be a dedicated eval env -- never the training env --
    so the training rollout state is left untouched.
    """
    returns: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep = 0.0
        done = truncated = False
        while not (done or truncated):
            obs, r, done, truncated, _ = env.step(act_fn(obs))
            ep += r
        returns.append(ep)
    return float(np.mean(returns)), float(np.std(returns))


# An eval curve is a list of (step, mean_return, std_return).
EvalPoint = Tuple[int, float, float]


def load_eval_curve(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (steps, means, stds) from a checkpoint's ``eval_curve``,
    or ``None`` if the checkpoint predates the eval hook."""
    data = torch.load(path, weights_only=False)
    curve = data.get("eval_curve")
    if not curve:
        return None
    arr = np.asarray(curve, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2]


def load_curve_preferring_eval(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (x, y, kind) with ``kind in {'eval', 'online'}``.

    Prefers the deterministic-eval curve (x = env step); falls back to
    the online per-episode trace (x = episode index) for legacy
    checkpoints.  Callers should surface ``kind`` in labels/titles.
    """
    ev = load_eval_curve(path)
    if ev is not None:
        steps, means, _ = ev
        return steps, means, "eval"
    data = torch.load(path, weights_only=False)
    online = np.asarray(data.get("episode_rewards", []), dtype=np.float32)
    return np.arange(len(online)), online, "online"


def load_train_seconds(path: Path) -> Optional[float]:
    """Wall-clock training seconds (excluding periodic eval) recorded in the
    checkpoint, or None for checkpoints written before compute tracking."""
    data = torch.load(path, weights_only=False)
    v = data.get("train_seconds")
    return float(v) if v else None


def load_total_steps(path: Path) -> Optional[int]:
    """Env-step count for the run, read off the last eval-curve point
    (which is logged at the final step)."""
    data = torch.load(path, weights_only=False)
    curve = data.get("eval_curve")
    return int(curve[-1][0]) if curve else None


def final_eval(path: Path, k: int = 3) -> float:
    """Scalar summary reward for the action/timestep sweeps: mean of the
    last ``k`` eval points.  Falls back to the last-50 online-episode mean
    for legacy checkpoints."""
    ev = load_eval_curve(path)
    if ev is not None:
        means = ev[1]
        kk = min(k, len(means))
        return float(np.mean(means[-kk:])) if kk else float("nan")
    data = torch.load(path, weights_only=False)
    online = np.asarray(data.get("episode_rewards", []), dtype=np.float32)
    if len(online) == 0:
        return float("nan")
    w = min(50, len(online))
    return float(online[-w:].mean())
