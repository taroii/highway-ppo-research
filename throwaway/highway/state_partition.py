"""
State Partition Tree: adaptive axis-aligned binary partitioning of the
observation space.  Each leaf owns an independent ActionZooming instance,
enabling state-dependent action resolution.

Split criterion: KL divergence of action distributions on each side of a
candidate axis-aligned split (evaluated at the median along each obs dim).
A leaf splits when it has accumulated >= min_samples observations AND the
best candidate split exceeds a KL threshold.

Design decisions (see discussion):
  - Max tree depth: 8  (at most 256 leaves)
  - Min samples before considering a split: 128 (one full rollout)
  - No merging/pruning for now (TODO: add later)
  - Children inherit parent's ActionZooming state (deep copy)
  - Raw 25D observations used directly (no feature engineering)
"""

from __future__ import annotations

import copy
import math
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from throwaway.highway.zooming import Cube, CubeStats


# ---------------------------------------------------------------------------
# ActionZooming — per-leaf action space manager (unchanged from original)
# ---------------------------------------------------------------------------

@dataclass
class ActionSplitInfo:
    """Records one action cube split: old index removed, new indices added."""
    old_idx: int
    new_indices: List[int]


class ActionZooming:
    """Manages a flat list of active cubes in [0,1]^2 as discrete actions."""

    def __init__(self, da: int = 1):
        self.da = da
        root = Cube(lower=np.zeros(da), s=1.0, d=da)
        self.active_cubes: List[Cube] = root.split_children()
        self.stats: List[CubeStats] = [
            CubeStats(Q=0.0) for _ in self.active_cubes
        ]

    @property
    def n_actions(self) -> int:
        return len(self.active_cubes)

    def get_env_action(self, idx: int) -> np.ndarray:
        """Center of cube idx, mapped from [0,1]^2 to [-1,1]^2."""
        cube = self.active_cubes[idx]
        center = cube.lower + 0.5 * cube.s
        return 2.0 * center - 1.0

    def update_play_counts(self, action_indices: List[int]):
        for idx in action_indices:
            self.stats[idx].n_play += 1

    def split_threshold(self, cube: Cube) -> int:
        return math.ceil((1.0 / cube.s) ** 2)

    def try_split(self) -> List[ActionSplitInfo]:
        splits: List[ActionSplitInfo] = []
        to_split = []
        for i, (cube, stat) in enumerate(zip(self.active_cubes, self.stats)):
            if stat.n_play >= self.split_threshold(cube):
                to_split.append(i)

        if not to_split:
            return splits

        for old_idx in reversed(to_split):
            cube = self.active_cubes[old_idx]
            children = cube.split_children()
            self.active_cubes.pop(old_idx)
            self.stats.pop(old_idx)
            new_start = len(self.active_cubes)
            new_indices = list(range(new_start, new_start + len(children)))
            for child in children:
                self.active_cubes.append(child)
                self.stats.append(CubeStats(Q=0.0))
            splits.append(ActionSplitInfo(old_idx=old_idx, new_indices=new_indices))

        return splits

    def deep_copy(self) -> "ActionZooming":
        """Create an independent copy (for inheriting to child leaves)."""
        new = ActionZooming.__new__(ActionZooming)
        new.da = self.da
        new.active_cubes = [
            Cube(lower=c.lower.copy(), s=c.s, d=c.d)
            for c in self.active_cubes
        ]
        new.stats = [
            CubeStats(Q=s.Q, n_play=0)  # reset play counts for the new leaf
            for s in self.stats
        ]
        return new


# ---------------------------------------------------------------------------
# StatePartitionTree — binary tree over observation space
# ---------------------------------------------------------------------------

class StateLeaf:
    """
    A leaf in the state partition tree.  Owns:
      - an ActionZooming instance (its own action discretization)
      - a unique integer leaf_id (used to index into the network's per-leaf heads)
      - accumulated (obs, action_idx) pairs from the current rollout for
        evaluating candidate splits
    """

    def __init__(self, leaf_id: int, zooming: ActionZooming):
        self.leaf_id = leaf_id
        self.zooming = zooming
        # Rollout buffer for split evaluation (cleared each rollout)
        self._obs_buffer: List[np.ndarray] = []
        self._act_buffer: List[int] = []

    def record(self, obs: np.ndarray, action_idx: int):
        """Record an (obs, action) pair for later split evaluation."""
        self._obs_buffer.append(obs)
        self._act_buffer.append(action_idx)

    def clear_buffer(self):
        self._obs_buffer.clear()
        self._act_buffer.clear()

    @property
    def n_samples(self) -> int:
        return len(self._obs_buffer)


class StatePartitionTree:
    """
    Adaptive binary partition of observation space R^obs_dim.

    Each internal node stores a split: (dimension, threshold).
    Each leaf stores a StateLeaf (with its own ActionZooming).

    The tree starts as a single root leaf.
    """

    def __init__(
        self,
        obs_dim: int,
        da: int = 1,
        max_depth: int = 8,
        min_samples_split: int = 128,
        kl_threshold: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.da = da
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.kl_threshold = kl_threshold

        self._next_leaf_id = 0

        # Tree structure: stored as parallel arrays indexed by node_id.
        # node_id 0 = root.
        # Internal nodes: _split_dim[n] >= 0, children at _left[n], _right[n]
        # Leaf nodes: _split_dim[n] == -1, leaf object at _leaves[n]
        self._split_dim: List[int] = []       # -1 for leaves
        self._split_val: List[float] = []     # threshold (only for internal)
        self._left: List[int] = []            # left child node_id
        self._right: List[int] = []           # right child node_id
        self._depth: List[int] = []           # depth of this node
        self._leaves: List[Optional[StateLeaf]] = []  # leaf data (None for internal)

        # Create root leaf
        root_leaf = self._make_leaf()
        self._add_node(split_dim=-1, split_val=0.0, left=-1, right=-1,
                        depth=0, leaf=root_leaf)

    def _make_leaf(self, parent_zooming: Optional[ActionZooming] = None) -> StateLeaf:
        """Create a new leaf, optionally inheriting a parent's ActionZooming."""
        lid = self._next_leaf_id
        self._next_leaf_id += 1
        if parent_zooming is not None:
            zooming = parent_zooming.deep_copy()
        else:
            zooming = ActionZooming(da=self.da)
        return StateLeaf(leaf_id=lid, zooming=zooming)

    def _add_node(self, split_dim: int, split_val: float,
                  left: int, right: int, depth: int,
                  leaf: Optional[StateLeaf]) -> int:
        node_id = len(self._split_dim)
        self._split_dim.append(split_dim)
        self._split_val.append(split_val)
        self._left.append(left)
        self._right.append(right)
        self._depth.append(depth)
        self._leaves.append(leaf)
        return node_id

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_leaf(self, obs: np.ndarray) -> StateLeaf:
        """Traverse tree to find the leaf for this observation."""
        node = 0
        while self._split_dim[node] >= 0:  # internal node
            dim = self._split_dim[node]
            val = self._split_val[node]
            if obs[dim] <= val:
                node = self._left[node]
            else:
                node = self._right[node]
        return self._leaves[node]

    def all_leaves(self) -> List[StateLeaf]:
        """Return all current leaf objects."""
        return [lf for lf in self._leaves if lf is not None]

    @property
    def n_leaves(self) -> int:
        return sum(1 for lf in self._leaves if lf is not None)

    # ------------------------------------------------------------------
    # Leaf-to-ID mapping (for network heads)
    # ------------------------------------------------------------------

    def leaf_id_to_leaf(self) -> Dict[int, StateLeaf]:
        return {lf.leaf_id: lf for lf in self.all_leaves()}

    def max_leaf_id(self) -> int:
        """Highest leaf_id currently in use (for sizing the head dict)."""
        leaves = self.all_leaves()
        return max(lf.leaf_id for lf in leaves) if leaves else 0

    # ------------------------------------------------------------------
    # Split evaluation (KL divergence of action distributions)
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """KL(p || q) for discrete distributions p, q."""
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def _symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
        """Symmetric KL = 0.5 * (KL(p||q) + KL(q||p))."""
        return 0.5 * (StatePartitionTree._kl_divergence(p, q) +
                       StatePartitionTree._kl_divergence(q, p))

    def _evaluate_split_candidate(
        self, leaf: StateLeaf, dim: int
    ) -> Tuple[float, float]:
        """
        Evaluate a candidate split on dimension `dim` at the median.
        Returns (kl_score, split_threshold_value).

        Computes action count histograms on each side, normalizes to
        distributions, and returns symmetric KL divergence.
        """
        obs_arr = np.array(leaf._obs_buffer)  # (N, obs_dim)
        act_arr = np.array(leaf._act_buffer)  # (N,)

        median_val = float(np.median(obs_arr[:, dim]))

        left_mask = obs_arr[:, dim] <= median_val
        right_mask = ~left_mask

        n_left = left_mask.sum()
        n_right = right_mask.sum()

        # Need at least some samples on each side
        min_side = max(10, self.min_samples_split // 8)
        if n_left < min_side or n_right < min_side:
            return 0.0, median_val

        n_actions = leaf.zooming.n_actions
        left_counts = np.bincount(act_arr[left_mask], minlength=n_actions).astype(float)
        right_counts = np.bincount(act_arr[right_mask], minlength=n_actions).astype(float)

        # Normalize to distributions
        left_dist = left_counts / left_counts.sum()
        right_dist = right_counts / right_counts.sum()

        kl = self._symmetric_kl(left_dist, right_dist)
        return kl, median_val

    def _find_best_split(self, leaf: StateLeaf) -> Optional[Tuple[int, float, float]]:
        """
        Search all obs dimensions for the best split.
        Returns (best_dim, best_threshold, best_kl) or None if no good split.
        """
        if leaf.n_samples < self.min_samples_split:
            return None

        best_kl = -1.0
        best_dim = -1
        best_val = 0.0

        for dim in range(self.obs_dim):
            kl, val = self._evaluate_split_candidate(leaf, dim)
            if kl > best_kl:
                best_kl = kl
                best_dim = dim
                best_val = val

        if best_kl < self.kl_threshold:
            return None

        return (best_dim, best_val, best_kl)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def try_split_leaves(self) -> List[Tuple[int, int, int]]:
        """
        Check all leaves for possible splits.  Execute any that pass.

        Returns list of (old_leaf_id, left_leaf_id, right_leaf_id) for
        each split that occurred.  The caller uses this to create new
        network heads.
        """
        splits_done: List[Tuple[int, int, int]] = []

        # Collect (node_id, leaf) pairs — snapshot before mutating
        leaf_nodes: List[Tuple[int, StateLeaf]] = []
        for node_id, lf in enumerate(self._leaves):
            if lf is not None and self._depth[node_id] < self.max_depth:
                leaf_nodes.append((node_id, lf))

        for node_id, leaf in leaf_nodes:
            result = self._find_best_split(leaf)
            if result is None:
                continue

            best_dim, best_val, best_kl = result

            # Create two child leaves, both inheriting parent's zooming
            left_leaf = self._make_leaf(parent_zooming=leaf.zooming)
            right_leaf = self._make_leaf(parent_zooming=leaf.zooming)

            # Add child nodes
            parent_depth = self._depth[node_id]
            left_id = self._add_node(
                split_dim=-1, split_val=0.0, left=-1, right=-1,
                depth=parent_depth + 1, leaf=left_leaf
            )
            right_id = self._add_node(
                split_dim=-1, split_val=0.0, left=-1, right=-1,
                depth=parent_depth + 1, leaf=right_leaf
            )

            # Convert parent from leaf to internal node
            self._split_dim[node_id] = best_dim
            self._split_val[node_id] = best_val
            self._left[node_id] = left_id
            self._right[node_id] = right_id
            self._leaves[node_id] = None  # no longer a leaf

            splits_done.append((leaf.leaf_id, left_leaf.leaf_id, right_leaf.leaf_id))

        return splits_done

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear_all_buffers(self):
        """Clear rollout buffers on all leaves (call at start of each rollout)."""
        for lf in self.all_leaves():
            lf.clear_buffer()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        leaves = self.all_leaves()
        depths = [self._depth[nid] for nid, lf in enumerate(self._leaves) if lf is not None]
        action_counts = [lf.zooming.n_actions for lf in leaves]
        parts = [
            f"leaves={len(leaves)}",
            f"depths={min(depths)}-{max(depths)}" if depths else "depths=0",
            f"actions/leaf={min(action_counts)}-{max(action_counts)}" if action_counts else "",
        ]
        return " ".join(parts)

    # TODO: Add leaf merging/pruning in the future.
    # If a leaf gets very few visits over several rollouts, merge it back
    # with its sibling.  This prevents dead leaves from accumulating.