"""
Adaptive action-space zooming.

Maintains a flat list of active cubes in [0, 1]^d_action.  Each cube is
one discrete action whose continuous value is the cube's center mapped
to [-1, 1]^d.  When a cube's play count reaches its split threshold
(ceil((1/s)^2)), it splits into 2^d children.  Callers observe splits
through `SplitInfo` so they can rebuild downstream structures (Q-heads,
policy heads) accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Cube:
    """Axis-aligned cube in [0, 1]^d."""
    lower: np.ndarray
    s: float
    d: int

    def center(self) -> np.ndarray:
        return self.lower + 0.5 * self.s

    def split_children(self) -> List["Cube"]:
        half = self.s / 2.0
        children: List[Cube] = []
        for mask in range(1 << self.d):
            offset = np.array([(mask >> i) & 1 for i in range(self.d)], dtype=float)
            children.append(Cube(lower=self.lower + offset * half, s=half, d=self.d))
        return children


@dataclass
class CubeStats:
    """Per-cube lifecycle state for the paper's buffering scheme.

    Fields mirror Algorithm 1 of Cao & Krishnamurthy (2021):
      - ``n_play``  == n~ (tilde): times this *active* cube was selected.
        Drives the split trigger (>= N_split) and is the UCB denominator.
      - ``n_update`` == n: samples *credited* to this cube.  For a buffering
        child it is the number of redirected parent-plays it has received;
        it graduates once ``n_update >= N_min(child)``.
      - ``buffering``: True while the cube is a child that has not yet
        accumulated enough of its own evidence to be selectable.
      - ``parent_group``/``child_group``/``n_pending``: lineage so a parent
        can be retired once *all* its children graduate (its domain is then
        covered by the children, so it is no longer selectable).
    """
    n_play: int = 0
    n_update: int = 0
    buffering: bool = False
    parent_group: int = -1   # buffering child -> its parent's group id
    child_group: int = -1    # active parent -> group id of its buffering children (-1 if none)
    n_pending: int = 0       # active parent -> children not yet graduated
    retire: bool = False     # active parent -> flagged for removal this maintain()


def n_split_threshold(s: float) -> int:
    """N_split(B) = (d_max / r(B))^2 with d_max = 1 (cube in [0,1]^d).
    A cube splits once its *play* count reaches this."""
    return math.ceil((1.0 / s) ** 2)


def n_min_threshold(s: float) -> int:
    """N_min(B) = N_split(B) / 4.  A buffering child graduates once its
    *update* (redirected-sample) count reaches this."""
    return math.ceil((1.0 / s) ** 2 / 4.0)


@dataclass
class SplitInfo:
    """One split event: which local cube index was removed and what children replaced it."""
    old_idx: int
    new_indices: List[int]


class ActionZooming:
    """Manages a flat list of active cubes for one (optional) cluster."""

    def __init__(self, da: int = 1, init_depth: int = 3, max_depth: int = 4):
        """Pre-split the root ``init_depth`` times so every arm starts at
        resolution (1/2)^init_depth.  ``init_depth=3`` with ``da=1`` yields
        8 cubes of side 1/8 -- the same resolution as UniformActionGrid(n=8),
        so zooming and uniform share an initial operating point and
        zooming can only *refine* from there.  ``max_depth`` caps further
        refinement -- cubes at side length (1/2)^max_depth or finer won't
        be split even if they hit the play-count threshold."""
        self.da = da
        self.max_depth = max_depth
        self.min_side = (0.5) ** max_depth
        cubes: List[Cube] = [Cube(lower=np.zeros(da), s=1.0, d=da)]
        for _ in range(init_depth):
            split: List[Cube] = []
            for c in cubes:
                split.extend(c.split_children())
            cubes = split
        self.active_cubes: List[Cube] = cubes
        self.stats: List[CubeStats] = [CubeStats() for _ in self.active_cubes]

    @property
    def n_actions(self) -> int:
        return len(self.active_cubes)

    def get_env_action(self, idx: int) -> np.ndarray:
        center = self.active_cubes[idx].center()  # in [0, 1]^da
        return 2.0 * center - 1.0                 # in [-1, 1]^da

    def register_play(self, idx: int) -> None:
        self.stats[idx].n_play += 1

    def play_counts(self) -> np.ndarray:
        return np.array([s.n_play for s in self.stats], dtype=np.int64)

    def split_threshold(self, cube: Cube) -> int:
        """N_split(B) = (1/s)^2 (d_max = 1).  Kept for callers that want the
        per-cube split trigger directly; see ``n_split_threshold`` for the
        free function used by the factored lifecycle."""
        return n_split_threshold(cube.s)
