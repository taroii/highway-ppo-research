"""
The factored, action-only cube partition of \\textsc{ZoomQ}, with its
buffering phase. This is the core of the new method.

Paper: "Adaptive Action Discretization for Continuous-Action RL" (ZoomQ).
  - Factored partition + budget: Sec. 3.1 "Factored partition" (the
    ``d_A`` independent per-axis trees ``P_k^{(i)}``, ``F_k^{(i)}`` and the
    budget ``N_tot`` = ``total_budget``).
  - Split rule: Sec. 3.2 "When to split", Eq. (3) (N_split).
  - Buffering / promotion / retirement: Sec. 3.3 "When to promote",
    Eq. (4) (N_min).
  - Full step-by-step control flow: Algorithm 1.
The buffering scheme follows Cao & Krishnamurthy, "Provably adaptive
reinforcement learning in metric spaces" (2021, corrected/erratum version).

Wraps ``da`` independent 1-D ``ActionZooming`` trees, one per action axis.
Total cells are additive in ``da`` (sidesteps the joint ``2^da`` blowup;
paper Sec. 3.1), and a single ``total_budget`` = ``N_tot`` caps the *sum*
of cells across axes (active + buffering).

Buffering phase (paper Sec. 3.3; Algorithm 1). When an active cube's
*play* count reaches ``N_split = (1/s)^2`` (Eq. (3) (N_split)) it splits into
two children of half the side:

  - children are created **buffering** (NOT selectable) and do **not**
    inherit the parent's value/count -- their Q-rows are re-initialized by
    the Q-net (paper Sec. 3.2: children "do not inherit the parent's
    estimate at creation");
  - the **parent stays active and selectable**, covering the region
    (paper Sec. 3.2, "The parent remains in P_k^{(i)}");
  - every ``buffer_period``-th play of the parent, that sample is
    **redirected** to a buffering child (Algorithm 1, redirection step;
    paper Sec. 3.3): the executed action is that child's centre and the
    transition is stored under the child's index, so the child accumulates
    its *own* evidence ``n_update``;
  - a child **graduates** (buffering -> active) once ``n_update >= N_min =
    N_split/4`` (Eq. (4) (N_min)); on graduation its play count is seeded to
    its accumulated ``n_update`` so its UCB bonus activates at a neighbour
    scale, not a spike;
  - once **all** children of a parent graduate, the parent is **retired**
    (paper Sec. 3.3), its Q-row removed.

Net effect on budget: a completed split is +2 children then -1 retired
parent = +1 cell (matches ``FactoredUniformActionGrid`` in steady state).

Adaptation note (paper Assumption 3 "Optimistic-Q conditions", Sec. 4): the analyzed
tabular idealization uses period ``H+1`` from its learning-rate rule
``alpha_t = (H+1)/(H+t)``. This deep implementation uses a neural Q-net +
replay, so ``buffer_period`` (the paper's ``B``) is a hyperparameter, not
pinned to the horizon.

Consumed by ``BranchingDQN`` (src/highway/dqn_factored.py).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.highway.zooming import (
    ActionZooming,
    CubeStats,
    n_min_threshold,
    n_split_threshold,
)


class FactoredActionZooming:
    """The ZoomQ action partition (paper Sec. 3.1 "Factored partition"):
    ``da`` independent 1-D cube trees, one per action axis, with a global
    cell budget ``total_budget`` (== the paper's ``N_tot``) and the
    buffering phase of Sec. 3.3."""

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
            init_depth: starting refinement depth per axis (``2 ** init_depth``
                active bins per axis at start).
            total_budget: max total cells (active + buffering) across all
                axes.  ``None`` means unlimited.  For matched-budget A/B
                with a factored uniform grid of ``n`` per axis, set
                ``total_budget = n * da``.
            per_axis_max_depth: hard per-axis depth cap.
        """
        self.da = da
        self.total_budget = total_budget
        self.axes: List[ActionZooming] = [
            ActionZooming(da=1, init_depth=init_depth,
                          max_depth=per_axis_max_depth)
            for _ in range(da)
        ]
        # Deterministic group-id counter for parent<->children lineage.
        # (No time/RNG source -- resume-safe.)
        self._next_group = 0
        if total_budget is not None and self.total_cells > total_budget:
            raise ValueError(
                f"init_depth={init_depth} produces {self.total_cells} cells "
                f"on da={da} (= {2 ** init_depth} per axis), which exceeds "
                f"total_budget={total_budget}.  Lower init_depth or raise "
                f"total_budget."
            )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def n_per_axis(self) -> List[int]:
        """Total cells per axis (active + buffering).  Sizes the Q-heads."""
        return [ax.n_actions for ax in self.axes]

    @property
    def total_cells(self) -> int:
        return sum(self.n_per_axis())

    @property
    def active_cells(self) -> int:
        """Selectable cells (excludes buffering).  Diagnostic only."""
        return sum(int((~self._buffering_mask(ax)).sum()) for ax in self.axes)

    def get_env_action(self, idx_per_axis: Sequence[int]) -> np.ndarray:
        # float64 to match the gym Box action_space dtype and avoid precision
        # drift over long mujoco horizons.
        return np.array(
            [self.axes[i].get_env_action(int(idx_per_axis[i]))[0]
             for i in range(self.da)]
        )

    def play_counts_per_axis(self) -> List[np.ndarray]:
        """Per-axis n_play vectors (UCB denominator)."""
        return [ax.play_counts() for ax in self.axes]

    @staticmethod
    def _buffering_mask(ax: ActionZooming) -> np.ndarray:
        return np.array([s.buffering for s in ax.stats], dtype=bool)

    def active_mask_per_axis(self) -> List[np.ndarray]:
        """Per-axis boolean mask, True where a cube is selectable (active,
        i.e. not buffering).  Buffering children are excluded from both the
        exploration argmax and the greedy/target argmax."""
        return [~self._buffering_mask(ax) for ax in self.axes]

    # ------------------------------------------------------------------
    # Per-step action + buffering service
    # ------------------------------------------------------------------

    def on_step(self, idx_per_axis: Sequence[int], buffer_period: int
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Register a play of the selected (active) cube on each axis and
        return ``(env_action, store_idx_per_axis)``.

        Implements the buffering *redirection* step of the paper (Sec. 3.3;
        Algorithm 1: "every B-th play of a parent with buffering children,
        redirect the sample to a child"). ``idx_per_axis`` are selected
        *active* indices (the caller masks buffering cubes out of selection).
        On every ``buffer_period`` (== the paper's ``B``)-th play of a parent
        that has buffering children, the play is redirected to the
        least-updated child: the returned action component is that child's
        centre and ``store_idx`` is the child's index, so the transition
        trains and credits the child (increments its ``n_update``).
        """
        env_action = np.empty(self.da, dtype=np.float64)
        store_idx = np.empty(self.da, dtype=np.int64)
        for i in range(self.da):
            ax = self.axes[i]
            idx = int(idx_per_axis[i])
            st = ax.stats[idx]
            st.n_play += 1
            served = False
            if st.child_group >= 0 and buffer_period > 0 \
                    and st.n_play % buffer_period == 0:
                children = [j for j, s in enumerate(ax.stats)
                            if s.buffering and s.parent_group == st.child_group]
                if children:
                    j = min(children, key=lambda k: ax.stats[k].n_update)
                    ax.stats[j].n_update += 1
                    store_idx[i] = j
                    env_action[i] = ax.get_env_action(j)[0]
                    served = True
            if not served:
                store_idx[i] = idx
                env_action[i] = ax.get_env_action(idx)[0]
        return env_action, store_idx

    # ------------------------------------------------------------------
    # Maintenance: graduate -> retire -> split
    # ------------------------------------------------------------------

    def maintain(self) -> Tuple[List[List[int]], List[int], List[np.ndarray]]:
        """Advance the buffering lifecycle one round (the maintenance block of
        Algorithm 1: promote, then retire, then split) and return the Q-head
        edits + replay-index remap the caller must apply:

            removals[i] -> ascending cube indices to DROP from axis i's head
                           (retired parents), applied first (compaction);
            appends[i]  -> number of freshly-created buffering child rows to
                           APPEND to axis i's head (arbitrary init), applied
                           after the removals;
            remaps[i]   -> int array (length = pre-maintain cube count on
                           axis i) mapping every old cube index to its new
                           index after the removals.  Survivors compact; a
                           retired parent maps to one of its (now-active)
                           children, since its stored transitions were played
                           in the parent region which the children now cover.
                           The caller applies this to the replay buffer's
                           stored action indices so they never go stale.

        Order within a call: graduate eligible children (flag flip, seed
        play count) -> retire parents whose children all graduated (remove)
        -> split eligible active cubes (append), budget permitting.
        """
        self._graduate()
        removals, remaps = self._retire()
        appends = self._split()
        return removals, appends, remaps

    def _graduate(self) -> None:
        # Promotion rule (paper Sec. 3.3, Eq. (4) (N_min)): a buffering child
        # graduates to active once its own update count reaches N_min; both
        # siblings' promotion drives parent retirement (in _retire).
        for ax in self.axes:
            for j, s in enumerate(ax.stats):
                if not s.buffering:
                    continue
                if s.n_update >= n_min_threshold(ax.active_cubes[j].s):
                    s.buffering = False
                    s.n_play = s.n_update          # seed UCB count (neighbour scale)
                    g = s.parent_group
                    # keep parent_group/child_group intact so _retire can map
                    # a retired parent's replay data onto its children
                    if g < 0:
                        continue
                    for p in ax.stats:
                        if p.child_group == g:
                            p.n_pending -= 1
                            if p.n_pending <= 0:
                                p.retire = True
                            break

    def _retire(self) -> Tuple[List[List[int]], List[np.ndarray]]:
        # Parent retirement (paper Sec. 3.3): once both children of a parent
        # have graduated, the parent's region is covered by them, so it is
        # removed from the active set. Also builds the replay-index remap
        # (a value-based/neural detail beyond the tabular paper): a retired
        # parent's stored transitions are relabelled onto a surviving child.
        removals: List[List[int]] = [[] for _ in range(self.da)]
        remaps: List[np.ndarray] = [np.empty(0, dtype=np.int64)
                                    for _ in range(self.da)]
        for i, ax in enumerate(self.axes):
            old_n = len(ax.stats)
            removed = sorted(j for j, s in enumerate(ax.stats) if s.retire)
            removed_set = set(removed)

            # survivor -> compacted new index (order preserved)
            remap = np.full(old_n, -1, dtype=np.int64)
            nxt = 0
            for k in range(old_n):
                if k in removed_set:
                    continue
                remap[k] = nxt
                nxt += 1
            # a retired parent's transitions map onto a surviving child cube
            # (its region is now covered by the children)
            for p in removed:
                g = ax.stats[p].child_group
                child = None
                for k, s in enumerate(ax.stats):
                    if (k not in removed_set and s.parent_group == g
                            and not s.buffering):
                        child = k
                        break
                remap[p] = remap[child] if child is not None else 0
            assert (remap >= 0).all(), "retire remap left a stale index"

            for j in reversed(removed):
                ax.active_cubes.pop(j)
                ax.stats.pop(j)
            removals[i] = removed
            remaps[i] = remap
        return removals, remaps

    def _split(self) -> List[int]:
        # Split rule (paper Sec. 3.2, Eq. (3) (N_split)): an active cube whose
        # play count reaches N_split spawns two buffering children of half
        # the side, subject to the global budget N_tot (paper Sec. 3.1).
        appends = [0 for _ in range(self.da)]
        budget = (self.total_budget - self.total_cells
                  if self.total_budget is not None
                  else self.da * 10**9)
        if budget < 2:
            return appends

        # Gather eligible active cubes across axes: no pending children,
        # not buffering, playable count over threshold, not at min side.
        cands: List[tuple] = []  # (n_play, axis_idx, cube_idx)
        for i, ax in enumerate(self.axes):
            for j, (c, s) in enumerate(zip(ax.active_cubes, ax.stats)):
                if (not s.buffering and s.child_group < 0 and not s.retire
                        and s.n_play >= n_split_threshold(c.s)
                        and c.s > ax.min_side + 1e-9):
                    cands.append((s.n_play, i, j))
        if not cands:
            return appends

        # Each split costs 2 cells; highest-play cubes win under the budget.
        cands.sort(key=lambda t: -t[0])
        max_splits = budget // 2
        for _, i, j in cands[:max_splits]:
            ax = self.axes[i]
            g = self._next_group
            self._next_group += 1
            parent = ax.stats[j]
            parent.child_group = g
            parent.n_pending = 0
            for child in ax.active_cubes[j].split_children():
                ax.active_cubes.append(child)
                ax.stats.append(CubeStats(buffering=True, parent_group=g))
                parent.n_pending += 1
                appends[i] += 1
        return appends
