"""
Gradient-Aware Zooming-Tree PPO for HighwayEnv -- Continuous Actions.

Key improvement over previous versions: the action-cell selection logits
are torch Parameters that receive gradients through the PPO surrogate
loss.  This makes the tree policy a true policy-gradient method rather
than a tabular EMA approach.

Architecture:
  - State partition tree: adaptive binary splits driven by advantage
    variance (CART-style, same as before)
  - Per-leaf action zooming tree: adaptive partition of continuous action
    space, but now each cell has a LEARNABLE LOGIT that is optimized by
    the PPO clipped surrogate loss
  - Value network: MLP V(s) for GAE baseline
  - Entropy bonus: applied to the cell distribution to prevent premature
    collapse
  - Pruning: sibling cells with similar logits are merged back

Policy:
    logits = [cell_0.logit, cell_1.logit, ..., cell_k.logit]
    P(cell_i) = softmax(logits)[i]
    action = center(cell_i)
    log_prob = log_softmax(logits)[i]      <- differentiable w.r.t. logits

The PPO surrogate loss backpropagates through log_softmax to update the
cell logits, giving us genuine policy gradient optimization with adaptive
action discretization.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import List, Tuple, Optional

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


# ===========================================================================
# Action-Space Zooming Tree (differentiable policy)
# ===========================================================================

class ActionCell:
    """
    A cell in the action-space zooming tree.

    Each cell stores its bounds and a reference to its position in the
    parent tree's logit vector.  The actual logit value lives in a
    torch tensor owned by the ActionZoomingTree.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, depth: int = 0):
        self.low = low.copy()
        self.high = high.copy()
        self.depth = depth
        self.center = (low + high) / 2.0
        self.width = high - low

        # Index into the parent ActionZoomingTree's logit tensor.
        # Set by the tree when the cell is created.
        self.logit_idx: int = -1

        # Statistics for split decisions (not for action selection)
        self.n_visits: int = 0
        self.ema_advantage: float = 0.0
        self.ema_variance: float = 0.0
        self.ema_alpha: float = 0.05

        # Tree structure
        self.is_leaf: bool = True
        self.children: List[ActionCell] = []

    @property
    def diameter(self) -> float:
        return float(np.max(self.width))

    def update_stats(self, advantage: float):
        """Update EMA stats for split decisions only."""
        self.n_visits += 1
        alpha = max(self.ema_alpha, 1.0 / self.n_visits)
        delta = advantage - self.ema_advantage
        self.ema_advantage += alpha * delta
        self.ema_variance = (1 - alpha) * self.ema_variance + alpha * delta ** 2

    def get_action(self) -> np.ndarray:
        return self.center.copy()


class ActionZoomingTree:
    """
    Adaptive partition of continuous action space with DIFFERENTIABLE
    cell selection.

    The cell logits are stored in a torch tensor (self.logits) which is
    registered as a parameter and receives gradients through the PPO loss.

    Splitting is still driven by visit counts and advantage variance
    (non-differentiable heuristic), but action SELECTION is fully
    differentiable.
    """

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        max_cells: int = 32,
        split_threshold: int = 10,
        max_depth: int = 8,
        n_splits: int = 2,
        min_cell_diameter: float = 0.01,
        prune_adv_threshold: float = 0.05,
        prune_min_visits: int = 30,
    ):
        self.action_low = action_low.copy()
        self.action_high = action_high.copy()
        self.action_dim = len(action_low)
        self.max_cells = max_cells
        self.split_threshold = split_threshold
        self.max_depth = max_depth
        self.n_splits = n_splits
        self.min_cell_diameter = min_cell_diameter
        self.prune_adv_threshold = prune_adv_threshold
        self.prune_min_visits = prune_min_visits

        # Root cell
        self.root = ActionCell(action_low, action_high, depth=0)
        self.root.logit_idx = 0

        # Differentiable logits -- one per leaf cell
        # Initialized to 0 (uniform distribution)
        self.logits = torch.zeros(1, requires_grad=True)

        # Leaf list cache (invalidated on split/prune)
        self._leaves_cache: Optional[List[ActionCell]] = None
        self._leaves_dirty = True

        self.total_visits: int = 0

    def get_leaves(self) -> List[ActionCell]:
        if self._leaves_dirty or self._leaves_cache is None:
            self._leaves_cache = self._collect_leaves(self.root)
            self._leaves_dirty = False
        return self._leaves_cache

    def _collect_leaves(self, node: ActionCell) -> List[ActionCell]:
        if node.is_leaf:
            return [node]
        result = []
        for child in node.children:
            result.extend(self._collect_leaves(child))
        return result

    @property
    def n_leaves(self) -> int:
        return len(self.get_leaves())

    def _find_leaf(self, action: np.ndarray, node: ActionCell = None) -> ActionCell:
        if node is None:
            node = self.root
        if node.is_leaf:
            return node
        for child in node.children:
            if np.all(action >= child.low) and np.all(action < child.high):
                return self._find_leaf(action, child)
        return node.children[-1]

    def _get_log_probs(self) -> torch.Tensor:
        """Compute log_softmax over current leaf logits. Differentiable."""
        leaves = self.get_leaves()
        # Gather logits for current leaves
        indices = [leaf.logit_idx for leaf in leaves]
        leaf_logits = self.logits[indices]
        return F.log_softmax(leaf_logits, dim=0)

    def sample(self) -> Tuple[np.ndarray, int, ActionCell]:
        """
        Sample a cell from the policy, return its center action.

        Returns:
            action: center of selected cell
            cell_position: index into the current leaf list (for log_prob)
            cell: the selected cell (for stats update)
        """
        leaves = self.get_leaves()
        with torch.no_grad():
            indices = [leaf.logit_idx for leaf in leaves]
            leaf_logits = self.logits[indices]
            probs = F.softmax(leaf_logits, dim=0).numpy()

        cell_pos = np.random.choice(len(leaves), p=probs)
        cell = leaves[cell_pos]
        action = cell.get_action()
        return action, cell_pos, cell

    def log_prob_at(self, cell_position: int) -> torch.Tensor:
        """
        Differentiable log_prob for a given cell position.
        This is what the PPO loss backpropagates through.
        """
        log_probs = self._get_log_probs()
        return log_probs[cell_position]

    def log_prob_for_action(self, action: np.ndarray) -> torch.Tensor:
        """Find which cell an action belongs to and return its log_prob."""
        leaves = self.get_leaves()
        cell = self._find_leaf(action)
        for i, leaf in enumerate(leaves):
            if id(leaf) is id(cell):
                return self.log_prob_at(i)
        # Fallback
        return self.log_prob_at(0)

    def entropy(self) -> torch.Tensor:
        """Entropy of the cell distribution. Differentiable."""
        leaves = self.get_leaves()
        indices = [leaf.logit_idx for leaf in leaves]
        leaf_logits = self.logits[indices]
        probs = F.softmax(leaf_logits, dim=0)
        log_probs = F.log_softmax(leaf_logits, dim=0)
        return -(probs * log_probs).sum()

    def greedy_action(self) -> np.ndarray:
        """Return center of cell with highest logit."""
        leaves = self.get_leaves()
        with torch.no_grad():
            indices = [leaf.logit_idx for leaf in leaves]
            leaf_logits = self.logits[indices]
            best_pos = leaf_logits.argmax().item()
        return leaves[best_pos].center.copy()

    def update_stats(self, cell: ActionCell, advantage: float):
        """Update cell visit stats (for split decisions)."""
        cell.update_stats(advantage)
        self.total_visits += 1
        self._maybe_split(cell)

    def _maybe_split(self, cell: ActionCell):
        if not cell.is_leaf:
            return
        if cell.n_visits < self.split_threshold:
            return
        if cell.depth >= self.max_depth:
            return
        if cell.diameter < self.min_cell_diameter:
            return
        new_needed = self.n_splits - 1
        if self.n_leaves + new_needed > self.max_cells:
            return

        # Split along widest dimension
        split_dim = int(np.argmax(cell.width))
        cell.is_leaf = False
        edges = np.linspace(cell.low[split_dim], cell.high[split_dim],
                            self.n_splits + 1)

        # Get parent's current logit value for initialization
        with torch.no_grad():
            parent_logit = self.logits[cell.logit_idx].item()

        # Create children and expand logit tensor
        for i in range(self.n_splits):
            child_low = cell.low.copy()
            child_high = cell.high.copy()
            child_low[split_dim] = edges[i]
            child_high[split_dim] = edges[i + 1]
            child = ActionCell(child_low, child_high, depth=cell.depth + 1)

            # Assign new logit index and expand tensor
            child.logit_idx = len(self.logits)
            new_logit = torch.tensor([parent_logit], requires_grad=True)
            with torch.no_grad():
                self.logits = torch.cat([
                    self.logits.detach(), new_logit
                ])
            self.logits.requires_grad_(True)

            cell.children.append(child)

        self._leaves_dirty = True

    def maybe_prune(self):
        """
        Merge sibling cells that have similar advantage outcomes AND
        have been visited enough times for the comparison to be meaningful.

        Criterion: siblings are merged only if:
          1. ALL siblings are leaves
          2. ALL siblings have at least prune_min_visits (grace period)
          3. The range of EMA advantages across siblings is below
             prune_adv_threshold (the split doesn't meaningfully
             distinguish good from bad actions)
        """
        pruned = self._prune_recursive(self.root)
        if pruned:
            self._leaves_dirty = True
        return pruned

    def _prune_recursive(self, node: ActionCell) -> bool:
        if node.is_leaf:
            return False

        # First try pruning deeper
        any_pruned = False
        for child in node.children:
            if self._prune_recursive(child):
                any_pruned = True

        # Check if ALL children are leaves
        if not all(c.is_leaf for c in node.children):
            return any_pruned

        # Grace period: all children must have enough visits
        if any(c.n_visits < self.prune_min_visits for c in node.children):
            return any_pruned

        # Advantage-based criterion: are the children's outcomes
        # meaningfully different?
        child_advs = [c.ema_advantage for c in node.children]
        adv_range = max(child_advs) - min(child_advs)

        if adv_range < self.prune_adv_threshold:
            # Merge: make this node a leaf again
            # Set the parent logit to the weighted average of children
            total_visits = sum(c.n_visits for c in node.children)
            if total_visits > 0:
                weighted_adv = sum(
                    c.ema_advantage * c.n_visits for c in node.children
                ) / total_visits
                node.ema_advantage = weighted_adv
                node.n_visits = total_visits
            with torch.no_grad():
                child_logits = [self.logits[c.logit_idx].item()
                                for c in node.children]
                avg_logit = sum(child_logits) / len(child_logits)
                self.logits[node.logit_idx] = avg_logit
            node.children.clear()
            node.is_leaf = True
            return True

        return any_pruned

    def summary(self) -> str:
        leaves = self.get_leaves()
        depths = [c.depth for c in leaves]
        visits = [c.n_visits for c in leaves]
        return (
            f"{len(leaves)} cells, "
            f"depth [{min(depths)}-{max(depths)}], "
            f"visits [{min(visits)}-{max(visits)}]"
        )


# ===========================================================================
# State-Space Partition Tree
# ===========================================================================

class StateNode:
    def __init__(
        self,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_zoom_kwargs: dict,
        depth: int = 0,
    ):
        self.obs_low = obs_low.copy()
        self.obs_high = obs_high.copy()
        self.action_low = action_low
        self.action_high = action_high
        self.action_zoom_kwargs = action_zoom_kwargs
        self.depth = depth

        self.is_leaf = True
        self.action_tree = ActionZoomingTree(
            action_low, action_high, **action_zoom_kwargs
        )

        self.split_dim: Optional[int] = None
        self.split_val: Optional[float] = None
        self.left: Optional[StateNode] = None
        self.right: Optional[StateNode] = None

    def find_leaf(self, obs_flat: np.ndarray) -> "StateNode":
        if self.is_leaf:
            return self
        if obs_flat[self.split_dim] <= self.split_val:
            return self.left.find_leaf(obs_flat)
        else:
            return self.right.find_leaf(obs_flat)

    def split(self, dim: int, val: float):
        self.is_leaf = False
        self.split_dim = dim
        self.split_val = val

        low_l, high_l = self.obs_low.copy(), self.obs_high.copy()
        high_l[dim] = val
        self.left = StateNode(
            low_l, high_l, self.action_low, self.action_high,
            self.action_zoom_kwargs, self.depth + 1,
        )

        low_r, high_r = self.obs_low.copy(), self.obs_high.copy()
        low_r[dim] = val
        self.right = StateNode(
            low_r, high_r, self.action_low, self.action_high,
            self.action_zoom_kwargs, self.depth + 1,
        )

    def get_leaves(self) -> List["StateNode"]:
        if self.is_leaf:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()


class StatePartitionTree:
    def __init__(
        self,
        obs_dim: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_zoom_kwargs: dict,
        max_leaves: int = 64,
        min_samples_split: int = 30,
        min_variance_reduction: float = 0.01,
        max_depth: int = 10,
    ):
        self.obs_dim = obs_dim
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.min_variance_reduction = min_variance_reduction
        self.max_depth = max_depth

        self.root = StateNode(
            obs_low, obs_high, action_low, action_high,
            action_zoom_kwargs, depth=0,
        )

    @property
    def n_leaves(self) -> int:
        return len(self.root.get_leaves())

    def find_leaf(self, obs_flat: np.ndarray) -> StateNode:
        return self.root.find_leaf(obs_flat)

    def get_all_action_tree_logits(self) -> List[torch.Tensor]:
        """Collect logit tensors from all state-tree leaves."""
        return [leaf.action_tree.logits for leaf in self.root.get_leaves()]

    def refine(self, observations: np.ndarray, advantages: np.ndarray,
               top_k_dims: int = 8):
        if self.n_leaves >= self.max_leaves:
            return

        N = len(advantages)
        if N < self.min_samples_split * 2:
            return

        adv_std = advantages.std()
        if adv_std < 1e-10:
            return

        correlations = np.zeros(self.obs_dim)
        for d in range(self.obs_dim):
            obs_d = observations[:, d]
            if obs_d.std() < 1e-10:
                continue
            c = np.corrcoef(obs_d, advantages)[0, 1]
            if not np.isnan(c):
                correlations[d] = abs(c)
        candidate_dims = np.argsort(-correlations)[:top_k_dims]

        leaves = self.root.get_leaves()
        leaf_to_idx: dict[int, List[int]] = {id(l): [] for l in leaves}
        for i in range(N):
            leaf = self.find_leaf(observations[i])
            leaf_to_idx[id(leaf)].append(i)

        best_splits: List[Tuple[StateNode, int, float, float]] = []

        for leaf in leaves:
            if not leaf.is_leaf or leaf.depth >= self.max_depth:
                continue
            idx = leaf_to_idx[id(leaf)]
            if len(idx) < self.min_samples_split:
                continue

            leaf_obs = observations[idx]
            leaf_adv = advantages[idx]
            leaf_var = leaf_adv.var()
            if leaf_var < 1e-10:
                continue

            best_dim, best_val, best_red = -1, 0.0, 0.0

            for d in candidate_dims:
                col = leaf_obs[:, d]
                if col.std() < 1e-10:
                    continue
                split_val = np.median(col)
                left_mask = col <= split_val
                right_mask = ~left_mask
                n_l, n_r = left_mask.sum(), right_mask.sum()
                if n_l < 5 or n_r < 5:
                    continue

                weighted_var = (
                    n_l * leaf_adv[left_mask].var()
                    + n_r * leaf_adv[right_mask].var()
                ) / len(idx)
                reduction = leaf_var - weighted_var

                if reduction > best_red:
                    best_red = reduction
                    best_dim = d
                    best_val = split_val

            if best_dim >= 0 and best_red >= self.min_variance_reduction:
                best_splits.append((leaf, best_dim, best_val, best_red))

        best_splits.sort(key=lambda x: -x[3])
        for leaf, dim, val, _ in best_splits:
            if self.n_leaves >= self.max_leaves:
                break
            leaf.split(dim, val)

    def summary(self) -> str:
        leaves = self.root.get_leaves()
        depths = [l.depth for l in leaves]
        ac = [l.action_tree.n_leaves for l in leaves]
        return (
            f"State tree: {len(leaves)} leaves, "
            f"depth [{min(depths)}-{max(depths)}], "
            f"action cells/leaf [{min(ac)}-{max(ac)}]"
        )

    def dimension_usage(self) -> dict:
        counts = {}
        def _walk(node):
            if node.is_leaf:
                return
            counts[node.split_dim] = counts.get(node.split_dim, 0) + 1
            _walk(node.left)
            _walk(node.right)
        _walk(self.root)
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ===========================================================================
# Value Network
# ===========================================================================

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: List[int] = [256, 256]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ===========================================================================
# Rollout Buffer
# ===========================================================================

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.cell_positions: List[int] = []          # index into leaf list
        self.state_leaf_ids: List[int] = []           # id() of state leaf
        self.action_cells: List[ActionCell] = []

    def store(self, obs, action, reward, done, log_prob, value,
              cell_position, state_leaf_id, action_cell):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.cell_positions.append(cell_position)
        self.state_leaf_ids.append(state_leaf_id)
        self.action_cells.append(action_cell)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ===========================================================================
# Zooming-Tree PPO (Gradient-Aware)
# ===========================================================================

class ZoomingTreePPO:
    """
    PPO with differentiable adaptive action discretization.

    The policy is a softmax over action-tree leaf cells, where the logits
    are torch tensors that receive gradients through the PPO surrogate loss.

    Training step:
      1. Collect rollouts: state -> state-tree leaf -> sample cell from
         softmax -> play cell center
      2. Compute GAE advantages using value network
      3. PPO update: backprop through log_softmax(cell_logits) to update
         cell logits + update value network
      4. Update cell visit stats (for split decisions)
      5. Periodically: refine state tree, split action cells, prune action cells
    """

    def __init__(
        self,
        env: gym.Env,
        # Value network
        hidden: List[int] = [256, 256],
        lr_value: float = 5e-4,
        lr_policy: float = 1e-3,
        # PPO
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.8,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        # State tree
        state_max_leaves: int = 64,
        state_min_samples: int = 30,
        state_min_var_reduction: float = 0.005,
        state_max_depth: int = 10,
        state_refine_interval: int = 5,
        state_top_k_dims: int = 8,
        # Action zooming
        action_max_cells: int = 32,
        action_split_threshold: int = 10,
        action_max_depth: int = 8,
        action_n_splits: int = 2,
        action_min_cell_diameter: float = 0.01,
        action_prune_adv_threshold: float = 0.05,
        action_prune_min_visits: int = 30,
        prune_interval: int = 10,
        # General
        seed: int = 0,
    ):
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.lr_policy = lr_policy
        self.state_refine_interval = state_refine_interval
        self.state_top_k_dims = state_top_k_dims
        self.prune_interval = prune_interval

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_shape = env.observation_space.shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low.astype(np.float32)
        self.action_high = env.action_space.high.astype(np.float32)

        # Value network
        self.value_net = ValueNetwork(self.obs_dim, hidden)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr_value
        )

        # State/action trees
        obs_low = np.full(self.obs_dim, -2.0)
        obs_high = np.full(self.obs_dim, 2.0)

        self.action_zoom_kwargs = dict(
            max_cells=action_max_cells,
            split_threshold=action_split_threshold,
            max_depth=action_max_depth,
            n_splits=action_n_splits,
            min_cell_diameter=action_min_cell_diameter,
            prune_adv_threshold=action_prune_adv_threshold,
            prune_min_visits=action_prune_min_visits,
        )

        self.state_tree = StatePartitionTree(
            obs_dim=self.obs_dim,
            obs_low=obs_low,
            obs_high=obs_high,
            action_low=self.action_low,
            action_high=self.action_high,
            action_zoom_kwargs=self.action_zoom_kwargs,
            max_leaves=state_max_leaves,
            min_samples_split=state_min_samples,
            min_variance_reduction=state_min_var_reduction,
            max_depth=state_max_depth,
        )

        # Policy optimizer -- will be rebuilt when tree structure changes
        self._rebuild_policy_optimizer()

        self.buffer = RolloutBuffer()
        self._obs, _ = env.reset(seed=seed)
        self._done = False
        self._rollout_count = 0
        self._all_obs_for_refine: List[np.ndarray] = []
        self._all_adv_for_refine: List[np.ndarray] = []

    def _rebuild_policy_optimizer(self):
        """Rebuild optimizer over all action-tree logit tensors."""
        logit_tensors = self.state_tree.get_all_action_tree_logits()
        # Each is a separate tensor; we optimize them all
        self.policy_optimizer = torch.optim.Adam(
            logit_tensors, lr=self.lr_policy
        )

    def _get_state_leaf_map(self) -> dict:
        """Map state-leaf id() to the leaf object."""
        return {id(l): l for l in self.state_tree.root.get_leaves()}

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=float)

        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
                next_nonterminal = 1.0 - float(self._done)
            else:
                next_val = values[t + 1]
                next_nonterminal = 1.0 - dones[t]
            delta = (rewards[t]
                     + self.gamma * next_val * next_nonterminal
                     - values[t])
            advantages[t] = last_gae = (
                delta
                + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            )
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self):
        self.buffer.clear()
        self.value_net.eval()

        for _ in range(self.n_steps):
            obs_flat = self._obs.flatten().astype(np.float32)
            obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

            with torch.no_grad():
                value = self.value_net(obs_t).item()

            # Policy: state tree -> action tree -> softmax sample
            state_leaf = self.state_tree.find_leaf(obs_flat)
            action, cell_pos, action_cell = state_leaf.action_tree.sample()
            action = np.clip(action, self.action_low, self.action_high)

            # Record log_prob (detached, for old_log_prob in PPO)
            with torch.no_grad():
                log_prob = state_leaf.action_tree.log_prob_at(cell_pos).item()

            next_obs, reward, done, truncated, info = self.env.step(action)
            self.buffer.store(
                obs_flat, action, reward, done or truncated,
                log_prob, value, cell_pos, id(state_leaf), action_cell,
            )

            if done or truncated:
                next_obs, _ = self.env.reset()
            self._obs = next_obs
            self._done = done or truncated

        # Bootstrap
        with torch.no_grad():
            obs_t = torch.from_numpy(
                self._obs.flatten().astype(np.float32)
            ).unsqueeze(0)
            last_value = self.value_net(obs_t).item()

        return self._compute_gae(last_value)

    # ------------------------------------------------------------------
    # PPO update (differentiable policy + value network)
    # ------------------------------------------------------------------

    def _update(self, advantages: np.ndarray, returns: np.ndarray):
        """
        PPO update with gradients flowing to both:
          - Value network parameters (via value loss)
          - Action-tree cell logits (via clipped surrogate policy loss)
        """
        self.value_net.train()

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        old_lp = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Build map from state_leaf_id -> state_leaf for log_prob recomputation
        leaf_map = self._get_state_leaf_map()

        n = len(self.buffer)
        indices = np.arange(n)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                mb = indices[start:end]

                # --- Value loss ---
                values = self.value_net(obs_t[mb])
                value_loss = F.mse_loss(values, ret_t[mb])

                # --- Policy loss (differentiable through cell logits) ---
                # Recompute log_probs under current logits
                new_log_probs = []
                entropies = []
                for i in mb:
                    sl_id = self.buffer.state_leaf_ids[i]
                    if sl_id in leaf_map:
                        state_leaf = leaf_map[sl_id]
                        action = self.buffer.actions[i]
                        lp = state_leaf.action_tree.log_prob_for_action(action)
                        ent = state_leaf.action_tree.entropy()
                    else:
                        # State leaf was split -- find new leaf for this obs
                        state_leaf = self.state_tree.find_leaf(
                            self.buffer.obs[i]
                        )
                        action = self.buffer.actions[i]
                        lp = state_leaf.action_tree.log_prob_for_action(action)
                        ent = state_leaf.action_tree.entropy()
                    new_log_probs.append(lp)
                    entropies.append(ent)

                new_lp_t = torch.stack(new_log_probs)
                entropy_t = torch.stack(entropies)

                # PPO clipped surrogate
                ratio = torch.exp(new_lp_t - old_lp[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -entropy_t.mean()

                # --- Combined update ---
                # Value network
                self.value_optimizer.zero_grad()
                v_loss = self.vf_coef * value_loss
                v_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

                # Policy (cell logits)
                self.policy_optimizer.zero_grad()
                p_loss = policy_loss + self.ent_coef * entropy_loss
                p_loss.backward()
                # Clip gradients on logits
                for logit_tensor in self.state_tree.get_all_action_tree_logits():
                    if logit_tensor.grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [logit_tensor], self.max_grad_norm
                        )
                self.policy_optimizer.step()

    # ------------------------------------------------------------------
    # Tree structure updates
    # ------------------------------------------------------------------

    def _update_cell_stats(self, advantages: np.ndarray):
        """Update visit stats for split decisions."""
        leaf_map = self._get_state_leaf_map()
        for i in range(len(self.buffer)):
            cell = self.buffer.action_cells[i]
            cell.update_stats(advantages[i])

            sl_id = self.buffer.state_leaf_ids[i]
            if sl_id in leaf_map:
                state_leaf = leaf_map[sl_id]
            else:
                state_leaf = self.state_tree.find_leaf(self.buffer.obs[i])
            state_leaf.action_tree._maybe_split(cell)
            state_leaf.action_tree.total_visits += 1

        self._all_obs_for_refine.append(np.array(self.buffer.obs))
        self._all_adv_for_refine.append(advantages)

    def _maybe_refine_state_tree(self):
        self._rollout_count += 1
        if self._rollout_count % self.state_refine_interval != 0:
            return
        if not self._all_obs_for_refine:
            return

        all_obs = np.concatenate(self._all_obs_for_refine, axis=0)
        all_adv = np.concatenate(self._all_adv_for_refine, axis=0)

        old_leaves = self.state_tree.n_leaves
        self.state_tree.refine(all_obs, all_adv,
                               top_k_dims=self.state_top_k_dims)
        new_leaves = self.state_tree.n_leaves

        if new_leaves > old_leaves:
            print(f"  [State Tree] {old_leaves} -> {new_leaves} leaves")
            dim_usage = self.state_tree.dimension_usage()
            if dim_usage:
                print(f"  [State Tree] Dim splits: {dim_usage}")
            # Rebuild optimizer since new action trees were created
            self._rebuild_policy_optimizer()

        self._all_obs_for_refine.clear()
        self._all_adv_for_refine.clear()

    def _maybe_prune_action_trees(self):
        if self._rollout_count % self.prune_interval != 0:
            return

        total_pruned = 0
        for state_leaf in self.state_tree.root.get_leaves():
            before = state_leaf.action_tree.n_leaves
            state_leaf.action_tree.maybe_prune()
            after = state_leaf.action_tree.n_leaves
            total_pruned += (before - after)

        if total_pruned > 0:
            print(f"  [Prune] Merged {total_pruned} action cells")
            self._rebuild_policy_optimizer()

    def _check_rebuild_optimizer(self):
        """
        After any structural change (split/prune), we need to rebuild
        the optimizer.  We detect this by checking if any logit tensor
        has been replaced.
        """
        # Splits create new tensors via cat, so we rebuild after
        # structural updates
        self._rebuild_policy_optimizer()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 1000):
        steps_done = 0
        episode_rewards: List[float] = []
        current_ep_reward = 0.0

        while steps_done < total_timesteps:
            # 1. Collect rollouts
            advantages, returns = self._collect_rollouts()

            # 2. PPO update (gradients to both value net and cell logits)
            self._update(advantages, returns)

            # 3. Update cell stats (for split decisions)
            self._update_cell_stats(advantages)

            # 4. Rebuild optimizer after potential splits
            self._rebuild_policy_optimizer()

            # 5. Periodically refine state tree and prune action trees
            self._maybe_refine_state_tree()
            self._maybe_prune_action_trees()

            # Track rewards
            for r, d in zip(self.buffer.rewards, self.buffer.dones):
                current_ep_reward += r
                if d:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

            steps_done += self.n_steps

            if episode_rewards and steps_done % print_every < self.n_steps:
                recent = episode_rewards[-50:] if episode_rewards else [0]
                sample_leaf = self.state_tree.root
                while not sample_leaf.is_leaf:
                    sample_leaf = sample_leaf.left
                print(
                    f"Steps: {steps_done}/{total_timesteps}, "
                    f"eps: {len(episode_rewards)}, "
                    f"mean_rew(50): {np.mean(recent):.2f}, "
                    f"state_leaves: {self.state_tree.n_leaves}, "
                    f"sample_act_cells: {sample_leaf.action_tree.n_leaves}"
                )

        return episode_rewards

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_flat = obs.flatten().astype(np.float32)
        state_leaf = self.state_tree.find_leaf(obs_flat)

        if deterministic:
            action = state_leaf.action_tree.greedy_action()
        else:
            action, _, _ = state_leaf.action_tree.sample()

        return np.clip(action, self.action_low, self.action_high)


# ===========================================================================
# Environment
# ===========================================================================

def make_highway_env():
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "ContinuousAction",
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "normalize": True,
                "absolute": False,
            },
            "duration": 30,
            "policy_frequency": 1,
        },
    )
    return env


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 100_000

    env = make_highway_env()
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"  range: [{env.action_space.low}, {env.action_space.high}]")
    print()

    agent = ZoomingTreePPO(
        env,
        # Value net
        hidden=[256, 256],
        lr_value=5e-4,
        lr_policy=1e-3,
        # PPO
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.8,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        # State tree
        state_max_leaves=64,
        state_min_samples=30,
        state_min_var_reduction=0.005,
        state_max_depth=10,
        state_refine_interval=5,
        state_top_k_dims=8,
        # Action zooming
        action_max_cells=32,
        action_split_threshold=10,
        action_max_depth=8,
        action_n_splits=2,
        action_min_cell_diameter=0.01,
        action_prune_adv_threshold=0.05,
        action_prune_min_visits=30,
        prune_interval=10,
        # General
        seed=42,
    )

    print("Starting Gradient-Aware Zooming-Tree PPO...")
    print("=" * 60)
    rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=1000)

    # --- Diagnostics ---
    print("\n" + "=" * 60)
    print("FINAL DIAGNOSTICS")
    print("=" * 60)

    print(f"\n{agent.state_tree.summary()}")

    dim_usage = agent.state_tree.dimension_usage()
    if dim_usage:
        feature_names = []
        for v in ["ego", "veh1", "veh2", "veh3", "veh4"]:
            for f in ["presence", "x", "y", "vx", "vy"]:
                feature_names.append(f"{v}_{f}")
        print("\nState dims used for splitting:")
        for dim, count in dim_usage.items():
            name = feature_names[dim] if dim < len(feature_names) else f"dim_{dim}"
            print(f"  {name} (dim {dim}): {count} splits")

    print("\nAction zooming per state leaf:")
    for i, leaf in enumerate(agent.state_tree.root.get_leaves()):
        at = leaf.action_tree
        if at.total_visits > 0:
            greedy = at.greedy_action()
            print(
                f"  Leaf {i}: {at.summary()}, "
                f"greedy=[{greedy[0]:.3f}, {greedy[1]:.3f}]"
            )

    # Evaluate
    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        ep_r = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            ep_r += r
        eval_rewards.append(ep_r)

    print(
        f"Eval 20 episodes: "
        f"mean={np.mean(eval_rewards):.2f}, "
        f"std={np.std(eval_rewards):.2f}"
    )
    env.close()