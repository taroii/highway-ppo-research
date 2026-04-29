"""
Factored adaptive action-space zooming with a global cell budget.

Wraps ``da`` independent 1-D ``ActionZooming`` trees, one per action
axis.  Total cells are additive in ``da`` (sidesteps the joint
``2^da`` blowup), and a single ``total_budget`` caps the *sum* across
axes — so the algorithm can spend more cells on important axes (the
ones the policy visits often) and less on quiet ones, instead of
forcing each axis into the same per-axis ceiling.

When candidates exceed the remaining budget, splits go to the
highest-play-count bins first regardless of axis.  This generalizes
joint zooming's behavior (high-traffic regions get refined first) to
the factored setting; the matched-budget contract against
``FactoredUniformActionGrid`` is preserved by construction:

    total_budget = n_actions_uniform * da
    => factored zooming total cells <= factored uniform total cells.

Different API from joint ``ActionZooming``:
  - ``get_env_action`` and ``register_play`` take a per-axis index
    array of length ``da``.
  - ``play_counts_per_axis`` returns a list (one count vector per
    axis, since each axis has its own bin count).
  - ``try_split`` returns ``List[List[SplitInfo]]`` so consumers can
    rebuild only the affected axis's Q-head.

Consumed by ``BranchingDQN`` (src/highway/dqn_factored.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.highway.zooming import ActionZooming, CubeStats, SplitInfo


class FactoredActionZooming:
    """``da`` independent 1-D zooming trees with a global cell budget."""

    def __init__(
        self,
        da: int,
        init_depth: int = 1,
        total_budget: Optional[int] = None,
        per_axis_max_depth: int = 20,
    ):
        """
        Args:
            da: action dimensionality.
            init_depth: starting refinement depth per axis.  Each axis
                begins with ``2 ** init_depth`` bins.  Default ``1`` means
                "go left vs go right" on each axis at start, then grow
                via splits — leaves the algorithm maximum room to make
                adaptive decisions.
            total_budget: maximum total cells across all axes.  ``None``
                means unlimited (each axis grows independently up to
                ``per_axis_max_depth``).  For matched-budget A/B with
                a factored uniform grid of ``n`` per axis, set
                ``total_budget = n * da``.
            per_axis_max_depth: hard per-axis depth cap (safety net so
                no single axis can refine to absurd resolutions).
                Default ``20`` (1M+ bins) is effectively unlimited and
                lets the global budget be the active constraint.
        """
        self.da = da
        self.total_budget = total_budget
        self.axes: List[ActionZooming] = [
            ActionZooming(da=1, init_depth=init_depth,
                          max_depth=per_axis_max_depth)
            for _ in range(da)
        ]
        if total_budget is not None and self.total_cells > total_budget:
            raise ValueError(
                f"init_depth={init_depth} produces {self.total_cells} cells "
                f"on da={da} (= {2 ** init_depth} per axis), which exceeds "
                f"total_budget={total_budget}.  Lower init_depth or raise "
                f"total_budget."
            )

    # ------------------------------------------------------------------
    # Per-axis state queries
    # ------------------------------------------------------------------

    def n_per_axis(self) -> List[int]:
        return [ax.n_actions for ax in self.axes]

    @property
    def total_cells(self) -> int:
        return sum(self.n_per_axis())

    def get_env_action(self, idx_per_axis: Sequence[int]) -> np.ndarray:
        # float64 to match the joint ``ActionZooming.get_env_action`` and the
        # gym Box action_space dtype — float32 would compound precision drift
        # in mujoco over long training horizons and break equivalence to joint.
        return np.array(
            [self.axes[i].get_env_action(int(idx_per_axis[i]))[0]
             for i in range(self.da)]
        )

    def register_play(self, idx_per_axis: Sequence[int]) -> None:
        for i in range(self.da):
            self.axes[i].register_play(int(idx_per_axis[i]))

    def play_counts_per_axis(self) -> List[np.ndarray]:
        return [ax.play_counts() for ax in self.axes]

    # ------------------------------------------------------------------
    # Budget-aware split
    # ------------------------------------------------------------------

    def try_split(self) -> List[List[SplitInfo]]:
        """Apply pending splits across all axes, capped at the remaining
        global budget.  Highest-play-count bins win when candidates
        exceed budget.  Returns one ``List[SplitInfo]`` per axis (empty
        for axes with no splits applied this call).  ``new_indices`` are
        relative to the FINAL post-call ``active_cubes`` of that axis.
        """
        budget = (self.total_budget - self.total_cells
                  if self.total_budget is not None
                  else self.da * 10**9)
        results: List[List[SplitInfo]] = [[] for _ in range(self.da)]
        if budget <= 0:
            return results

        per_axis_cands: Dict[int, List[tuple]] = {}
        for axis_idx, ax in enumerate(self.axes):
            cands = []
            for i, (c, s) in enumerate(zip(ax.active_cubes, ax.stats)):
                if s.n_play >= ax.split_threshold(c) and c.s > ax.min_side + 1e-9:
                    cands.append((s.n_play, i))
            if cands:
                per_axis_cands[axis_idx] = cands

        if not per_axis_cands:
            return results

        total_cands = sum(len(c) for c in per_axis_cands.values())
        if total_cands <= budget:
            chosen: Dict[int, List[int]] = {
                ax_i: [idx for _, idx in cands]
                for ax_i, cands in per_axis_cands.items()
            }
        else:
            flat = [(n, ax_i, idx)
                    for ax_i, cands in per_axis_cands.items()
                    for (n, idx) in cands]
            flat.sort(key=lambda t: -t[0])
            chosen = {}
            for _, ax_i, idx in flat[:budget]:
                chosen.setdefault(ax_i, []).append(idx)

        for axis_idx, idxs in chosen.items():
            ax = self.axes[axis_idx]
            idxs_sorted = sorted(idxs)
            n_pre = len(ax.active_cubes)
            num_survivors = n_pre - len(idxs_sorted)
            child_offset = 0
            # Reverse so pops don't shift indices that haven't been processed yet.
            for old_idx in reversed(idxs_sorted):
                cube = ax.active_cubes.pop(old_idx)
                ax.stats.pop(old_idx)
                children = cube.split_children()
                n_children = len(children)  # always 2 for a 1-D axis
                new_indices = list(range(
                    num_survivors + child_offset,
                    num_survivors + child_offset + n_children,
                ))
                child_offset += n_children
                for child in children:
                    ax.active_cubes.append(child)
                    ax.stats.append(CubeStats())
                results[axis_idx].append(
                    SplitInfo(old_idx=old_idx, new_indices=new_indices)
                )
        return results
