"""
Aggregate statistics via the ``rliable`` package (Agarwal et al. 2021,
"Deep Reinforcement Learning at the Edge of the Statistical Precipice"):
interquartile mean with stratified-bootstrap confidence intervals,
performance profiles, and probability of improvement.

This module is a thin adapter around ``rliable`` plus the checkpoint ->
score-matrix plumbing that ``rliable`` does not provide. It requires
``rliable`` (``pip install rliable``); see the repo README for the exact
version pin, since ``rliable``'s ``arch`` dependency currently needs
numpy < 2.

Score-matrix convention (as in rliable): a method's scores are an array
of shape ``(num_runs, num_tasks)`` -- one row per seed, one column per
task -- of *normalized* returns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.highway.eval_utils import final_eval


def _require_rliable():
    try:
        from rliable import library as rly
        from rliable import metrics
        return rly, metrics
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "This figure needs rliable: `pip install rliable`. Its `arch` "
            "dependency currently requires numpy < 2 (e.g. "
            "`pip install rliable \"arch==7.2.0\" \"pandas<2.2\"`)."
        ) from e


# ---------------------------------------------------------------------------
# rliable estimators (IQM + stratified-bootstrap CIs, profiles, POI)
# ---------------------------------------------------------------------------

def iqm_interval(scores: Dict[str, np.ndarray], reps: int = 50000
                 ) -> Dict[str, Tuple[float, float, float]]:
    """{name: (IQM, ci_low, ci_high)} for each ``(num_runs, num_tasks)``
    score matrix, via rliable's stratified bootstrap."""
    rly, metrics = _require_rliable()
    agg = lambda m: np.array([metrics.aggregate_iqm(m)])
    pt, ci = rly.get_interval_estimates(
        {k: np.asarray(v, dtype=float) for k, v in scores.items()}, agg, reps=reps)
    return {k: (float(pt[k][0]), float(ci[k][0, 0]), float(ci[k][1, 0]))
            for k in scores}


def iqm_ci(x: Sequence[float], reps: int = 5000) -> Tuple[float, float, float]:
    """(IQM, ci_low, ci_high) for a 1-D sample of per-seed scores. Treated
    as a single-task ``(num_runs, 1)`` matrix for rliable."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return iqm_interval({"_": a.reshape(-1, 1)}, reps=reps)["_"]


def performance_profiles(scores: Dict[str, np.ndarray], taus: np.ndarray,
                         reps: int = 2000):
    """(distributions, cis): fraction of runs with score > tau, per method,
    with stratified-bootstrap CI bands."""
    rly, _ = _require_rliable()
    return rly.create_performance_profile(
        {k: np.asarray(v, dtype=float) for k, v in scores.items()},
        np.asarray(taus, dtype=float), reps=reps)


def prob_improvement(x: np.ndarray, y: np.ndarray, reps: int = 50000
                     ) -> Tuple[float, float, float]:
    """P(x > y): averaged over tasks, with stratified-bootstrap CI."""
    rly, metrics = _require_rliable()
    pt, ci = rly.get_interval_estimates(
        {"pair": (np.asarray(x, dtype=float), np.asarray(y, dtype=float))},
        metrics.probability_of_improvement, reps=reps)
    return float(pt["pair"]), float(ci["pair"][0, 0]), float(ci["pair"][1, 0])


# ---------------------------------------------------------------------------
# Checkpoint -> score-matrix assembly (not part of rliable)
# ---------------------------------------------------------------------------

def _finals_by_seed(paths: List[Path], regex: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for p in paths:
        m = re.search(regex, p.name)
        if m:
            out[int(m.group(1))] = final_eval(p)
    return out


def build_score_matrices(
    task_dirs: Dict[str, Path],
    n_actions: int,
    normalize: str = "minmax",
) -> Tuple[Dict[str, np.ndarray], List[str], List[int]]:
    """Assemble ``{arm: (num_seeds, num_tasks)}`` normalized score matrices
    at a fixed action budget ``n_actions``.

    ``task_dirs`` maps a task name to its efficiency checkpoint directory
    (``uniform_n{N}_seed*``, ``zooming_n{N}_seed*``, ``sac_seed*``). Uses the
    seeds common to all arms/tasks so the matrices are rectangular.

    ``normalize``: 'minmax' rescales each task to [0,1] via the pooled min/max
    across all arms and seeds in that task; 'sac' divides by the task's mean
    SAC score.
    """
    arms = {"uniform": rf"uniform_n{n_actions}_seed(\d+)\.pt$",
            "zooming": rf"zooming_n{n_actions}_seed(\d+)\.pt$",
            "sac":     r"sac_seed(\d+)\.pt$"}
    tasks = list(task_dirs)
    per = {arm: {} for arm in arms}
    for task, d in task_dirs.items():
        paths = sorted(Path(d).glob("*.pt"))
        for arm, rgx in arms.items():
            per[arm][task] = _finals_by_seed(paths, rgx)

    seed_sets = [set(per[a][t]) for a in arms for t in tasks if per[a][t]]
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not common:
        return {}, tasks, []

    norm = {}
    for t in tasks:
        if normalize == "sac":
            ref = np.nanmean([per["sac"][t][s] for s in common
                              if s in per["sac"][t]]) or 1.0
            norm[t] = (0.0, ref)
        else:
            pooled = np.array([per[a][t][s] for a in arms for s in common
                               if s in per[a][t]], dtype=float)
            lo, hi = np.nanmin(pooled), np.nanmax(pooled)
            norm[t] = (lo, hi if hi > lo else lo + 1.0)

    mats: Dict[str, np.ndarray] = {}
    for arm in arms:
        M = np.empty((len(common), len(tasks)))
        for j, t in enumerate(tasks):
            lo, hi = norm[t]
            for i, s in enumerate(common):
                M[i, j] = (per[arm][t].get(s, np.nan) - lo) / (hi - lo)
        mats[arm] = M
    return mats, tasks, common
