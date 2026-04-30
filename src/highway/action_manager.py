"""
Action grid factory.

A trivial wrapper that builds either a ``ActionZooming`` (adaptive) or a
``UniformActionGrid`` (fixed) given a mode string.  Both grids implement
the same minimal interface used by ``DQN``:

  - ``n_actions`` (int property)
  - ``get_env_action(idx) -> np.ndarray``
  - ``register_play(idx)``
  - ``play_counts() -> np.ndarray``
  - ``try_split() -> List[SplitInfo]``  (no-op for uniform)

DQN consumes the grid directly; there is no per-cluster routing here --
the legacy ``ClusteredActionManager`` lives under ``old/clustering/`` for
reference.
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np

from src.highway.zooming import ActionZooming, SplitInfo
from src.highway.uniform_grid import UniformActionGrid


class ActionGrid(Protocol):
    @property
    def n_actions(self) -> int: ...
    def get_env_action(self, idx: int) -> np.ndarray: ...
    def register_play(self, idx: int) -> None: ...
    def play_counts(self) -> np.ndarray: ...
    def try_split(self) -> List[SplitInfo]: ...


def make_grid(
    mode: str,
    da: int = 1,
    *,
    uniform_n: int = 16,
    init_depth: int = 3,
    max_depth: int = 4,
) -> ActionGrid:
    """Construct an action grid.

    Args:
        mode: ``"uniform"`` or ``"zooming"``.
        da: action dimensionality.
        uniform_n: grid size per axis (mode=uniform).
        init_depth: starting refinement depth (mode=zooming).
        max_depth: maximum refinement depth, capping how fine the
            adaptive grid can get (mode=zooming).
    """
    if mode == "zooming":
        return ActionZooming(da=da, init_depth=init_depth, max_depth=max_depth)
    if mode == "uniform":
        return UniformActionGrid(da=da, n=uniform_n)
    raise ValueError(f"unknown mode: {mode!r}")
