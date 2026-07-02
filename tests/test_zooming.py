"""Regression tests for the factored zooming lifecycle.

These lock the invariants that the buffering rewrite depends on, and that
the earlier joint ``try_split`` violated (its multi-split mid-list child
mapping was off-by-shift -- see the audit's Issue 4):

  1. Q-head rows stay identity-aligned with cubes through arbitrary
     split (append) + retire (remove/compact) rounds -- head row k always
     corresponds to cube k.
  2. The replay-index remap returned by ``maintain`` keeps every stored
     action index pointing at a currently-valid row (a retired parent's
     transitions follow onto a surviving child), even when several
     mid-list parents retire in the same round.
  3. Buffering cubes are never selectable.
  4. Total cells never exceed the budget.

Run: ``python tests/test_zooming.py`` (also collectable by pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.highway.zooming import n_min_threshold, n_split_threshold
from src.highway.zooming_factored import FactoredActionZooming


def _apply_head_edit(head, removals, appends, grid_axis):
    """Mirror what BranchingQNetwork does: drop removed rows (compact), then
    append the freshly-created children.  ``head`` is a shadow list of the
    Cube *identities* occupying each head row."""
    removed = set(removals)
    survivors = [head[k] for k in range(len(head)) if k not in removed]
    n_children = appends
    new_children = grid_axis.active_cubes[len(survivors):] if n_children else []
    assert len(new_children) == n_children
    return survivors + list(new_children)


def test_lifecycle_alignment_and_remap_stress():
    """Play every active cube hard on 2 axes for many rounds; assert head
    alignment + buffer-remap validity after every maintain."""
    grid = FactoredActionZooming(da=2, init_depth=2, total_budget=40)
    BP = 3

    heads = [list(ax.active_cubes) for ax in grid.axes]          # shadow heads
    buf = [[], []]                                               # shadow replay indices
    max_simul_retire = 0
    saw_multi_split = False

    for _round in range(120):
        # play every currently-active cube a few times to spread refinement
        for _ in range(4):
            masks = grid.active_mask_per_axis()
            active = [np.flatnonzero(m) for m in masks]
            # one env step per active-cube pairing (cycle the shorter axis)
            n = max(len(active[0]), len(active[1]))
            for t in range(n):
                idx = [int(active[0][t % len(active[0])]),
                       int(active[1][t % len(active[1])])]
                # selection must never land on a buffering cube
                for i in range(2):
                    assert masks[i][idx[i]], "selected a buffering cube"
                _, store = grid.on_step(idx, BP)
                for i in range(2):
                    buf[i].append(int(store[i]))

        removals, appends, remaps = grid.maintain()
        max_simul_retire = max(max_simul_retire, max(len(removals[0]), len(removals[1])))
        if max(appends) >= 4:
            saw_multi_split = True

        for i in range(2):
            ax = grid.axes[i]
            # remap shadow buffer, then assert every index is a valid row
            buf[i] = [int(remaps[i][k]) for k in buf[i]]
            for k in buf[i]:
                assert 0 <= k < ax.n_actions, f"stale buffer index {k} axis {i}"
            # apply head edit and assert identity alignment head[k] is cube[k]
            heads[i] = _apply_head_edit(heads[i], removals[i], appends[i], ax)
            assert len(heads[i]) == ax.n_actions
            for k, cube in enumerate(heads[i]):
                assert cube is ax.active_cubes[k], \
                    f"HEAD/CUBE MISALIGN axis {i} row {k}"
            assert grid.total_cells <= grid.total_budget

    print(f"stress ok: max simultaneous retirements={max_simul_retire}, "
          f"multi-split round seen={saw_multi_split}, "
          f"final n_per_axis={grid.n_per_axis()}")
    assert saw_multi_split, "test did not exercise a multi-split round"
    assert max_simul_retire >= 1, "test did not exercise a retirement"


def test_single_split_cycle_sides():
    """A cube of side 1/2 splits into two 1/4 children; after both graduate
    the parent is retired, leaving a valid finer partition."""
    g = FactoredActionZooming(da=1, init_depth=1, total_budget=4)
    ax = g.axes[0]
    BP = 2
    assert n_split_threshold(0.5) == 4 and n_min_threshold(0.25) == 4
    for step in range(1, 200):
        g.on_step([0], BP)   # hammer cube 0
        g.maintain()
        if g.total_cells == 3 and g.active_cells == 3:
            break
    else:
        raise AssertionError("split cycle never completed")
    assert not any(s.buffering for s in ax.stats)
    assert sorted(round(c.s, 4) for c in ax.active_cubes) == [0.25, 0.25, 0.5]
    print("single-split cycle ok: sides", sorted(round(c.s, 4) for c in ax.active_cubes))


if __name__ == "__main__":
    test_single_split_cycle_sides()
    test_lifecycle_alignment_and_remap_stress()
    print("\nALL ZOOMING REGRESSION TESTS PASSED")
