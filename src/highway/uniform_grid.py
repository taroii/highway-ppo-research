"""
Uniform action grid.

Mirrors the ActionZooming API (n_actions, get_env_action, register_play,
play_counts, try_split) but with a fixed grid and no splits.  This lets
the DQN core treat uniform and zooming arms interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.highway.zooming import SplitInfo


@dataclass
class _Stats:
    n_play: int = 0


class UniformActionGrid:
    """Regular grid of n^da points in [-1, 1]^da."""

    def __init__(self, da: int = 1, n: int = 8):
        self.da = da
        self.n = n
        # Cartesian grid of cell centers in [0, 1]^da, mapped to [-1, 1]^da.
        axes = [np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n for _ in range(da)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, da)
        self._actions = 2.0 * mesh - 1.0                                # (n**da, da)
        self.stats: List[_Stats] = [_Stats() for _ in range(len(self._actions))]

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    def get_env_action(self, idx: int) -> np.ndarray:
        return self._actions[idx].copy()

    def register_play(self, idx: int) -> None:
        self.stats[idx].n_play += 1

    def play_counts(self) -> np.ndarray:
        return np.array([s.n_play for s in self.stats], dtype=np.int64)

    def try_split(self) -> List[SplitInfo]:  # no-op -- uniform never splits
        return []
