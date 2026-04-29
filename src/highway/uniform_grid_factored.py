"""
Factored uniform action grid.

Wraps ``da`` independent 1-D ``UniformActionGrid`` instances, one per
action axis.  Total active cells = ``n * da`` (additive in ``da``)
instead of the joint version's ``n ** da`` Cartesian blowup.

Same per-axis API as ``FactoredActionZooming`` so it can drop into
``BranchingDQN`` as the no-split, fixed-grid arm:

  - ``n_per_axis()`` -> ``[n] * da``
  - ``get_env_action(idx_per_axis)`` -> joint action in ``[-1, 1]^da``
  - ``register_play(idx_per_axis)``
  - ``play_counts_per_axis()``
  - ``try_split()`` -> ``[[]] * da``  (no-op)

Matched-budget contract: pair with ``FactoredActionZooming(da, max_depth=k)``
where ``n == 2 ** k`` so both arms have the same per-axis bin count and
hence the same total cell count ``n * da``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.highway.uniform_grid import UniformActionGrid
from src.highway.zooming import SplitInfo


class FactoredUniformActionGrid:
    """One ``UniformActionGrid(da=1, n=n)`` per action axis."""

    def __init__(self, da: int, n: int):
        self.da = da
        self.n = n
        self.axes: List[UniformActionGrid] = [
            UniformActionGrid(da=1, n=n) for _ in range(da)
        ]

    def n_per_axis(self) -> List[int]:
        return [ax.n_actions for ax in self.axes]

    @property
    def total_cells(self) -> int:
        return sum(self.n_per_axis())

    def get_env_action(self, idx_per_axis: Sequence[int]) -> np.ndarray:
        # float64 to match the joint ``UniformActionGrid.get_env_action`` and
        # the gym Box action_space dtype.
        return np.array(
            [self.axes[i].get_env_action(int(idx_per_axis[i]))[0]
             for i in range(self.da)]
        )

    def register_play(self, idx_per_axis: Sequence[int]) -> None:
        for i in range(self.da):
            self.axes[i].register_play(int(idx_per_axis[i]))

    def play_counts_per_axis(self) -> List[np.ndarray]:
        return [ax.play_counts() for ax in self.axes]

    def try_split(self) -> List[List[SplitInfo]]:
        return [[] for _ in range(self.da)]
