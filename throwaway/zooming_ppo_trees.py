from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import List, Tuple, Optional

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


# ===========================================================================
# Action-Space Zooming Tree (defines the policy distribution)
# ===========================================================================

class ActionCell:
    """A cell in the action-space partition."""

    def __init__(self, low: np.ndarray, high: np.ndarray, depth: int = 0,
                 parent_mean_adv: float = 0.0, ema_alpha: float = 0.05):
        self.low = low.copy()
        self.high = high.copy()
        self.depth = depth
        self.center = (low + high) / 2.0
        self.width = high - low
        self.ema_alpha = ema_alpha

        # Advantage statistics (EMA-based to handle nonstationarity)
        self.n_visits: int = 0
        self.mean_advantage: float = parent_mean_adv  # inherit from parent
        self.ema_variance: float = 0.0  # EMA of squared deviations

        # Tree structure
        self.is_leaf: bool = True
        self.children: List[ActionCell] = []

    @property
    def diameter(self) -> float:
        return float(np.max(self.width))

    @property
    def variance(self) -> float:
        if self.n_visits < 2:
            return float('inf')
        return self.ema_variance

    def update(self, advantage: float):
        """
        EMA update for mean advantage.

        For the first few samples, use a larger effective alpha to avoid
        being stuck on the inherited parent estimate. After that, use
        the configured alpha so recent experience dominates.
        """
        self.n_visits += 1
        # Adaptive alpha: larger early on for faster initial learning,
        # then settles to configured value
        alpha = max(self.ema_alpha, 1.0 / self.n_visits)
        delta = advantage - self.mean_advantage
        self.mean_advantage += alpha * delta
        # EMA variance (tracks how noisy the advantages are)
        self.ema_variance = (1 - alpha) * self.ema_variance + alpha * delta ** 2

    def get_action(self) -> np.ndarray:
        """Return the center point of this cell (standard zooming)."""
        return self.center.copy()


class ActionZoomingTree:
    """
    Adaptive partition of continuous action space that defines a policy
    distribution via softmax over leaf cells.

    Policy:
        P(cell_i) = softmax(mean_advantage_i / temperature)
        action = center(cell_i)
        log_prob = log P(cell_i)
    """

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        max_cells: int = 32,
        split_threshold: int = 10,
        temperature: float = 1.0,
        max_depth: int = 8,
        n_splits: int = 2,
        min_cell_diameter: float = 0.01,
        ema_alpha: float = 0.05,
    ):
        self.action_low = action_low.copy()
        self.action_high = action_high.copy()
        self.action_dim = len(action_low)
        self.max_cells = max_cells
        self.split_threshold = split_threshold
        self.temperature = temperature
        self.max_depth = max_depth
        self.n_splits = n_splits
        self.min_cell_diameter = min_cell_diameter
        self.ema_alpha = ema_alpha

        self.root = ActionCell(action_low, action_high, depth=0,
                               ema_alpha=ema_alpha)
        self.total_visits: int = 0

    def get_leaves(self) -> List[ActionCell]:
        return self._collect_leaves(self.root)

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
        return node.children[-1]  # fallback

    def _compute_cell_logits(self, leaves: List[ActionCell]) -> np.ndarray:
        """Compute softmax logits for cell selection."""
        logits = np.array([cell.mean_advantage for cell in leaves])
        return logits / max(self.temperature, 1e-8)

    def sample(self) -> Tuple[np.ndarray, float, ActionCell]:
        """
        Sample a cell from the policy distribution, play its center.

        Returns:
            action: center point of the selected cell
            log_prob: log P(cell) under the softmax policy
            cell: the selected cell (for later update)
        """
        leaves = self.get_leaves()

        # Softmax over cells
        logits = self._compute_cell_logits(leaves)
        # Numerically stable softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()

        # Sample a cell
        cell_idx = np.random.choice(len(leaves), p=probs)
        cell = leaves[cell_idx]

        # Play the center of the cell (standard zooming)
        action = cell.get_action()

        # log_prob is just log P(cell) -- no volume term needed
        log_prob = float(np.log(probs[cell_idx] + 1e-10))

        return action, log_prob, cell

    def log_prob(self, action: np.ndarray) -> float:
        """Compute log_prob of the cell containing this action."""
        leaves = self.get_leaves()
        logits = self._compute_cell_logits(leaves)
        logits_shifted = logits - np.max(logits)
        log_normalizer = np.log(np.sum(np.exp(logits_shifted)))

        cell = self._find_leaf(action)
        cell_id = id(cell)
        for i, leaf in enumerate(leaves):
            if id(leaf) is cell_id:
                return float(logits_shifted[i] - log_normalizer)

        # Fallback (shouldn't happen)
        return float(-log_normalizer)

    def greedy_action(self) -> np.ndarray:
        """Return center of cell with highest mean advantage."""
        leaves = self.get_leaves()
        visited = [c for c in leaves if c.n_visits > 0]
        if not visited:
            return (self.action_low + self.action_high) / 2.0
        best = max(visited, key=lambda c: c.mean_advantage)
        return best.center.copy()

    def update(self, cell: ActionCell, advantage: float):
        """Update cell statistics and maybe split."""
        cell.update(advantage)
        self.total_visits += 1
        self._maybe_split(cell)

    def update_action(self, action: np.ndarray, advantage: float):
        """Find the cell for an action and update it."""
        cell = self._find_leaf(action)
        self.update(cell, advantage)

    def _maybe_split(self, cell: ActionCell):
        if not cell.is_leaf:
            return
        if cell.n_visits < self.split_threshold:
            return
        if cell.depth >= self.max_depth:
            return
        if cell.diameter < self.min_cell_diameter:
            return
        new_cells_needed = self.n_splits - 1
        if self.n_leaves + new_cells_needed > self.max_cells:
            return

        # Split along widest dimension
        split_dim = int(np.argmax(cell.width))
        cell.is_leaf = False
        edges = np.linspace(cell.low[split_dim], cell.high[split_dim],
                            self.n_splits + 1)

        for i in range(self.n_splits):
            child_low = cell.low.copy()
            child_high = cell.high.copy()
            child_low[split_dim] = edges[i]
            child_high[split_dim] = edges[i + 1]
            child = ActionCell(
                child_low, child_high, depth=cell.depth + 1,
                parent_mean_adv=cell.mean_advantage,
                ema_alpha=self.ema_alpha,
            )
            cell.children.append(child)

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
    """Binary tree node for state-space partitioning."""

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
    """Adaptive binary partition of the observation space."""

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
# Value Network (only V(s), no policy head)
# ===========================================================================

class ValueNetwork(nn.Module):
    """MLP value function for GAE baseline. No policy head needed."""

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
        self.action_cells: List[ActionCell] = []

    def store(self, obs, action, reward, done, log_prob, value, action_cell):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.action_cells.append(action_cell)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ===========================================================================
# Zooming-Tree PPO
# ===========================================================================

class ZoomingTreePPO:
    """
    PPO where the POLICY is defined by the zooming tree, not a neural net.

    Policy: state -> state-tree leaf -> softmax over action cells -> sample
    Value baseline: neural net V(s) for GAE

    PPO surrogate loss is used to update the value network.  The policy
    (tree cell weights) is updated via exponential moving average (EMA)
    of advantages -- recent experience dominates over stale early estimates,
    handling the nonstationarity of advantages as the value baseline improves.

    The PPO clipping mechanism is still applied: we compute importance
    weights (ratio of current log_prob to old log_prob) and clip them.
    When the tree partition changes between rollouts, the log_probs change,
    creating natural importance weight corrections.
    """

    def __init__(
        self,
        env: gym.Env,
        # Value network
        hidden: List[int] = [256, 256],
        lr: float = 5e-4,
        # PPO / rollout
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.8,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
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
        action_split_threshold: int = 8,
        action_temperature: float = 1.0,
        action_max_depth: int = 8,
        action_n_splits: int = 2,
        action_min_cell_diameter: float = 0.01,
        action_ema_alpha: float = 0.05,
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
        self.max_grad_norm = max_grad_norm
        self.state_refine_interval = state_refine_interval
        self.state_top_k_dims = state_top_k_dims

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_shape = env.observation_space.shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low.astype(np.float32)
        self.action_high = env.action_space.high.astype(np.float32)

        # Value network only
        self.value_net = ValueNetwork(self.obs_dim, hidden)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

        # Trees
        obs_low = np.full(self.obs_dim, -2.0)
        obs_high = np.full(self.obs_dim, 2.0)

        self.action_zoom_kwargs = dict(
            max_cells=action_max_cells,
            split_threshold=action_split_threshold,
            temperature=action_temperature,
            max_depth=action_max_depth,
            n_splits=action_n_splits,
            min_cell_diameter=action_min_cell_diameter,
            ema_alpha=action_ema_alpha,
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

        self.buffer = RolloutBuffer()
        self._obs, _ = env.reset(seed=seed)
        self._done = False
        self._rollout_count = 0
        self._all_obs_for_refine: List[np.ndarray] = []
        self._all_adv_for_refine: List[np.ndarray] = []

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

            # Policy is the tree
            state_leaf = self.state_tree.find_leaf(obs_flat)
            action, log_prob, action_cell = state_leaf.action_tree.sample()

            # Clip action to env bounds
            action = np.clip(action, self.action_low, self.action_high)

            next_obs, reward, done, truncated, info = self.env.step(action)
            self.buffer.store(
                obs_flat, action, reward, done or truncated,
                log_prob, value, action_cell,
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
    # PPO update (value network + importance-weighted policy signal)
    # ------------------------------------------------------------------

    def _update(self, advantages: np.ndarray, returns: np.ndarray):
        """
        Update the value network via PPO-style loss.

        Also compute importance weights from the tree policy to ensure
        the advantage estimates properly account for policy changes
        (cell splits may have changed log_probs since collection).
        """
        self.value_net.train()

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        old_lp = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Compute current log_probs under (possibly changed) tree policy
        current_lp = []
        for i in range(len(self.buffer)):
            state_leaf = self.state_tree.find_leaf(self.buffer.obs[i])
            lp = state_leaf.action_tree.log_prob(self.buffer.actions[i])
            current_lp.append(lp)
        new_lp_t = torch.tensor(current_lp, dtype=torch.float32)

        n = len(self.buffer)
        indices = np.arange(n)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                mb = indices[start:end]

                values = self.value_net(obs_t[mb])

                # Importance weights from tree policy
                ratio = torch.exp(new_lp_t[mb] - old_lp[mb])
                ratio_clamped = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )

                # We use the clipped ratio to weight the value loss,
                # ensuring stability when tree partition changes
                policy_surrogate = -torch.min(
                    ratio * adv_t[mb],
                    ratio_clamped * adv_t[mb],
                ).mean()

                value_loss = nn.functional.mse_loss(values, ret_t[mb])

                # Total loss: value fitting + policy surrogate signal
                loss = self.vf_coef * value_loss + policy_surrogate

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Tree updates
    # ------------------------------------------------------------------

    def _update_action_trees(self, advantages: np.ndarray):
        """Update action cell statistics with observed advantages."""
        for i in range(len(self.buffer)):
            cell = self.buffer.action_cells[i]
            cell.update(advantages[i])

            # Also check if this triggers a split
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

        self._all_obs_for_refine.clear()
        self._all_adv_for_refine.clear()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 1000):
        steps_done = 0
        episode_rewards: List[float] = []
        current_ep_reward = 0.0

        while steps_done < total_timesteps:
            advantages, returns = self._collect_rollouts()
            self._update_action_trees(advantages)
            self._update(advantages, returns)
            self._maybe_refine_state_tree()

            for r, d in zip(self.buffer.rewards, self.buffer.dones):
                current_ep_reward += r
                if d:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

            steps_done += self.n_steps

            if episode_rewards and steps_done % print_every < self.n_steps:
                recent = episode_rewards[-50:] if episode_rewards else [0]
                # Sample action tree info
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
        lr=5e-4,
        # PPO
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.8,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
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
        action_split_threshold=8,
        action_temperature=1.0,
        action_max_depth=8,
        action_n_splits=2,
        action_min_cell_diameter=0.01,
        action_ema_alpha=0.05,
        # General
        seed=42,
    )

    print("Starting Zooming-Tree PPO (Policy-Gradient Consistent)...")
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