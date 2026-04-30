"""
Zooming-Tree PPO for HighwayEnv.

Integrates two adaptive mechanisms into PPO:
1. **State-space partition tree**: A binary tree that adaptively splits the
   observation space along dimensions where advantage variance is highest,
   performing implicit feature selection over all 26 dimensions (5 vehicles
   x 5 features + 1 ego).  Splits are driven by variance-reduction of GAE
   advantages, mirroring CART regression-tree splitting.
2. **Per-leaf zooming (UCB-style action selection)**: Each leaf of the
   partition tree maintains its own action-value estimates with confidence
   bounds.  Actions are chosen via UCB within the leaf, and the estimates
   are updated with the observed advantages.

The neural-net policy/value heads are kept as a *warm prior* -- the tree
refines action selection on top of the network's suggestions, and the
value network still provides the GAE baseline.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


# ===========================================================================
# Partition Tree (adaptive state-space discretisation)
# ===========================================================================

@dataclass
class LeafStats:
    """Per-leaf action statistics for UCB-based zooming."""
    n_actions: int
    action_counts: np.ndarray = field(default=None)
    action_values: np.ndarray = field(default=None)   # running mean of advantages
    action_sq_values: np.ndarray = field(default=None) # running mean of adv^2 (for variance)
    total_visits: int = 0

    def __post_init__(self):
        if self.action_counts is None:
            self.action_counts = np.zeros(self.n_actions)
            self.action_values = np.zeros(self.n_actions)
            self.action_sq_values = np.zeros(self.n_actions)

    def update(self, action: int, advantage: float):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        # Incremental mean update
        self.action_values[action] += (advantage - self.action_values[action]) / n
        self.action_sq_values[action] += (advantage**2 - self.action_sq_values[action]) / n
        self.total_visits += 1

    def ucb_action(self, exploration_coef: float = 1.0) -> int:
        """Select action via UCB1.  Unexplored actions chosen first."""
        unvisited = np.where(self.action_counts == 0)[0]
        if len(unvisited) > 0:
            return int(np.random.choice(unvisited))
        # UCB1 scores
        ucb = self.action_values + exploration_coef * np.sqrt(
            2 * np.log(self.total_visits) / self.action_counts
        )
        return int(np.argmax(ucb))

    def best_action(self) -> int:
        """Greedy action (for evaluation)."""
        if self.total_visits == 0:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.action_values))


class PartitionNode:
    """Binary tree node for state-space partitioning."""

    def __init__(self, n_actions: int, obs_bounds_low: np.ndarray,
                 obs_bounds_high: np.ndarray, depth: int = 0):
        self.n_actions = n_actions
        self.obs_low = obs_bounds_low.copy()
        self.obs_high = obs_bounds_high.copy()
        self.depth = depth

        # Leaf-specific
        self.is_leaf = True
        self.stats = LeafStats(n_actions)

        # Internal-node-specific (set on split)
        self.split_dim: Optional[int] = None
        self.split_val: Optional[float] = None
        self.left: Optional[PartitionNode] = None   # obs[dim] <= split_val
        self.right: Optional[PartitionNode] = None  # obs[dim] > split_val

    def find_leaf(self, obs_flat: np.ndarray) -> "PartitionNode":
        if self.is_leaf:
            return self
        if obs_flat[self.split_dim] <= self.split_val:
            return self.left.find_leaf(obs_flat)
        else:
            return self.right.find_leaf(obs_flat)

    def split(self, dim: int, val: float):
        """Split this leaf into two children."""
        self.is_leaf = False
        self.split_dim = dim
        self.split_val = val

        low_l, high_l = self.obs_low.copy(), self.obs_high.copy()
        high_l[dim] = val
        self.left = PartitionNode(self.n_actions, low_l, high_l, self.depth + 1)

        low_r, high_r = self.obs_low.copy(), self.obs_high.copy()
        low_r[dim] = val
        self.right = PartitionNode(self.n_actions, low_r, high_r, self.depth + 1)

    def get_leaves(self) -> List["PartitionNode"]:
        if self.is_leaf:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()


class PartitionTree:
    """
    Adaptive binary partition tree over the observation space.

    Splitting criterion: variance reduction of GAE advantages, evaluated
    per candidate dimension at the median split point within each leaf.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        max_leaves: int = 128,
        min_samples_split: int = 30,
        min_variance_reduction: float = 0.01,
        max_depth: int = 12,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.min_variance_reduction = min_variance_reduction
        self.max_depth = max_depth

        self.root = PartitionNode(n_actions, obs_low, obs_high, depth=0)

    @property
    def n_leaves(self) -> int:
        return len(self.root.get_leaves())

    def find_leaf(self, obs_flat: np.ndarray) -> PartitionNode:
        return self.root.find_leaf(obs_flat)

    def refine(self, observations: np.ndarray, advantages: np.ndarray,
               top_k_dims: int = 6):
        """
        Attempt to split leaves where advantage variance is high.

        For efficiency, we first rank dimensions by their global correlation
        with advantages, then only evaluate splits on the top-k dimensions.

        Args:
            observations: (N, obs_dim) array of flattened observations
            advantages: (N,) array of GAE advantages
            top_k_dims: number of candidate dimensions to consider
        """
        if self.n_leaves >= self.max_leaves:
            return

        N = len(advantages)
        if N < self.min_samples_split * 2:
            return

        # --- Global dimension ranking by |correlation| with advantages ---
        # This is our cheap "feature selection" heuristic
        correlations = np.zeros(self.obs_dim)
        adv_std = advantages.std()
        if adv_std < 1e-10:
            return
        for d in range(self.obs_dim):
            obs_d = observations[:, d]
            obs_std = obs_d.std()
            if obs_std < 1e-10:
                continue
            correlations[d] = abs(np.corrcoef(obs_d, advantages)[0, 1])
        candidate_dims = np.argsort(-correlations)[:top_k_dims]

        # --- Assign each observation to its leaf ---
        leaves = self.root.get_leaves()
        leaf_to_idx: dict[int, List[int]] = {id(leaf): [] for leaf in leaves}
        for i in range(N):
            leaf = self.find_leaf(observations[i])
            leaf_to_idx[id(leaf)].append(i)

        # --- Evaluate splits for each leaf ---
        best_splits: List[Tuple[PartitionNode, int, float, float]] = []

        for leaf in leaves:
            if not leaf.is_leaf:
                continue
            if leaf.depth >= self.max_depth:
                continue
            idx = leaf_to_idx[id(leaf)]
            if len(idx) < self.min_samples_split:
                continue

            leaf_obs = observations[idx]
            leaf_adv = advantages[idx]
            leaf_var = leaf_adv.var()
            if leaf_var < 1e-10:
                continue

            best_dim, best_val, best_reduction = -1, 0.0, 0.0

            for d in candidate_dims:
                col = leaf_obs[:, d]
                # Skip dimensions with no variance in this leaf
                if col.std() < 1e-10:
                    continue
                split_val = np.median(col)
                # Avoid degenerate splits
                left_mask = col <= split_val
                right_mask = ~left_mask
                n_left, n_right = left_mask.sum(), right_mask.sum()
                if n_left < 5 or n_right < 5:
                    continue

                # Variance reduction (weighted)
                var_left = leaf_adv[left_mask].var()
                var_right = leaf_adv[right_mask].var()
                weighted_var = (n_left * var_left + n_right * var_right) / len(idx)
                reduction = leaf_var - weighted_var

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_dim = d
                    best_val = split_val

            if best_dim >= 0 and best_reduction >= self.min_variance_reduction:
                best_splits.append((leaf, best_dim, best_val, best_reduction))

        # Sort by variance reduction (descending) and apply splits up to budget
        best_splits.sort(key=lambda x: -x[3])
        for leaf, dim, val, red in best_splits:
            if self.n_leaves >= self.max_leaves:
                break
            leaf.split(dim, val)

    def update_leaf_stats(self, obs_flat: np.ndarray, action: int, advantage: float):
        """Update the zooming statistics for the leaf containing this obs."""
        leaf = self.find_leaf(obs_flat)
        leaf.stats.update(action, advantage)

    def summary(self) -> str:
        leaves = self.root.get_leaves()
        depths = [l.depth for l in leaves]
        visits = [l.stats.total_visits for l in leaves]
        return (
            f"Tree: {len(leaves)} leaves, "
            f"depth range [{min(depths)}-{max(depths)}], "
            f"visits range [{min(visits)}-{max(visits)}]"
        )

    def dimension_usage(self) -> dict:
        """Count how many times each dimension was used as a split."""
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
# Network (same architecture, kept for value baseline)
# ===========================================================================

class ActorCritic(nn.Module):
    """Separate MLP heads for policy (actor) and value (critic)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int] = [256, 256]):
        super().__init__()
        layers_pi: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_pi += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers_pi.append(nn.Linear(prev, n_actions))
        self.pi = nn.Sequential(*layers_pi)

        layers_vf: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_vf += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers_vf.append(nn.Linear(prev, 1))
        self.vf = nn.Sequential(*layers_vf)

    def policy(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.pi(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.vf(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        dist = self.policy(obs)
        action = dist.sample()
        return action, dist.log_prob(action), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.policy(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value


# ===========================================================================
# Rollout buffer
# ===========================================================================

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ===========================================================================
# Zooming-Tree PPO
# ===========================================================================

class ZoomingTreePPO:
    """
    PPO with adaptive state-space partitioning and per-leaf UCB action
    selection (zooming).

    The action-selection policy blends:
      - The neural-net policy (soft prior, provides log-probs for PPO update)
      - The tree's UCB action selection (exploration/exploitation per region)

    During rollouts, actions are chosen by the tree's UCB rule.  The net's
    log-probs are still recorded so that the PPO surrogate loss can update
    the network, keeping it as a useful value baseline and warm-start for
    new leaves.

    Every `refine_interval` rollout collections, the partition tree is
    refined using the advantage variance criterion.
    """

    def __init__(
        self,
        env: gym.Env,
        # Network / PPO hyperparams
        hidden: List[int] = [256, 256],
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 5e-4,
        gamma: float = 0.8,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        # Zooming / tree hyperparams
        ucb_coef: float = 1.5,
        tree_max_leaves: int = 128,
        tree_min_samples_split: int = 30,
        tree_min_var_reduction: float = 0.005,
        tree_max_depth: int = 12,
        refine_interval: int = 5,
        top_k_dims: int = 8,
        # Blend: probability of using tree's UCB vs network's policy
        # Starts high (trust UCB exploration), can be annealed
        tree_action_prob: float = 0.7,
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
        self.ucb_coef = ucb_coef
        self.refine_interval = refine_interval
        self.top_k_dims = top_k_dims
        self.tree_action_prob = tree_action_prob

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_shape = env.observation_space.shape
        self.obs_dim = int(np.prod(obs_shape))
        self.n_actions = env.action_space.n

        self.net = ActorCritic(self.obs_dim, self.n_actions, hidden)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        # Observation bounds for the tree
        # HighwayEnv with normalize=True gives observations in roughly [-1, 1]
        # but we use a slightly wider range to be safe
        obs_low = np.full(self.obs_dim, -2.0)
        obs_high = np.full(self.obs_dim, 2.0)

        self.tree = PartitionTree(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            obs_low=obs_low,
            obs_high=obs_high,
            max_leaves=tree_max_leaves,
            min_samples_split=tree_min_samples_split,
            min_variance_reduction=tree_min_var_reduction,
            max_depth=tree_max_depth,
        )

        self._obs, _ = env.reset(seed=seed)
        self._done = False
        self._rollout_count = 0

        # Storage for advantage data used to refine tree
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
            delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            )
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Action selection (blended tree + network)
    # ------------------------------------------------------------------

    def _select_action(self, obs_flat: np.ndarray, obs_tensor: torch.Tensor):
        """
        Select action using blended strategy:
        - With probability tree_action_prob, use the tree's UCB rule
        - Otherwise, sample from the neural net policy

        Always return the net's log_prob for the chosen action (needed for
        PPO surrogate loss).
        """
        with torch.no_grad():
            dist = self.net.policy(obs_tensor)
            value = self.net.value(obs_tensor).item()

        leaf = self.tree.find_leaf(obs_flat)

        if np.random.random() < self.tree_action_prob:
            action = leaf.stats.ucb_action(self.ucb_coef)
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action)).item()
        return action, log_prob, value

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self):
        self.buffer.clear()
        self.net.eval()

        for _ in range(self.n_steps):
            obs_flat = self._obs.flatten().astype(np.float32)
            obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

            action, log_prob, value = self._select_action(obs_flat, obs_t)

            next_obs, reward, done, truncated, info = self.env.step(action)
            self.buffer.store(obs_flat, action, reward, done or truncated, log_prob, value)

            if done or truncated:
                next_obs, _ = self.env.reset()
            self._obs = next_obs
            self._done = done or truncated

        # Bootstrap
        with torch.no_grad():
            obs_t = torch.from_numpy(
                self._obs.flatten().astype(np.float32)
            ).unsqueeze(0)
            last_value = self.net.value(obs_t).item()

        return self._compute_gae(last_value)

    # ------------------------------------------------------------------
    # PPO update (same as vanilla, operates on network)
    # ------------------------------------------------------------------

    def _update_network(self, advantages: np.ndarray, returns: np.ndarray):
        self.net.train()

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        act_t = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_lp_t = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(self.buffer)
        indices = np.arange(n)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                mb = indices[start:end]

                new_lp, entropy, values = self.net.evaluate(obs_t[mb], act_t[mb])

                ratio = torch.exp(new_lp - old_lp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, ret_t[mb])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Tree updates (zooming stats + refinement)
    # ------------------------------------------------------------------

    def _update_tree(self, advantages: np.ndarray):
        """Update per-leaf action statistics with observed advantages."""
        obs_arr = np.array(self.buffer.obs)
        actions = self.buffer.actions
        for i in range(len(self.buffer)):
            self.tree.update_leaf_stats(obs_arr[i], actions[i], advantages[i])

        # Accumulate data for periodic refinement
        self._all_obs_for_refine.append(obs_arr)
        self._all_adv_for_refine.append(advantages)

    def _maybe_refine_tree(self):
        """Refine the tree every `refine_interval` rollouts."""
        self._rollout_count += 1
        if self._rollout_count % self.refine_interval != 0:
            return

        if len(self._all_obs_for_refine) == 0:
            return

        # Pool recent data for splitting decisions
        all_obs = np.concatenate(self._all_obs_for_refine, axis=0)
        all_adv = np.concatenate(self._all_adv_for_refine, axis=0)

        old_leaves = self.tree.n_leaves
        self.tree.refine(all_obs, all_adv, top_k_dims=self.top_k_dims)
        new_leaves = self.tree.n_leaves

        if new_leaves > old_leaves:
            print(f"  [Tree] Refined: {old_leaves} --> {new_leaves} leaves")
            dim_usage = self.tree.dimension_usage()
            if dim_usage:
                print(f"  [Tree] Dimension splits: {dim_usage}")

        # Clear accumulated data (keep recent only)
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
            # 1. Collect rollouts (using blended tree+net action selection)
            advantages, returns = self._collect_rollouts()

            # 2. Update neural net via PPO
            self._update_network(advantages, returns)

            # 3. Update tree leaf statistics with advantages
            self._update_tree(advantages)

            # 4. Maybe refine tree partition
            self._maybe_refine_tree()

            # Track episode rewards
            for r, d in zip(self.buffer.rewards, self.buffer.dones):
                current_ep_reward += r
                if d:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

            steps_done += self.n_steps

            if episode_rewards and steps_done % print_every < self.n_steps:
                recent = episode_rewards[-50:] if episode_rewards else [0]
                print(
                    f"Steps: {steps_done}/{total_timesteps}, "
                    f"episodes: {len(episode_rewards)}, "
                    f"mean_reward(last 50): {np.mean(recent):.2f}, "
                    f"tree: {self.tree.n_leaves} leaves"
                )

        return episode_rewards

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        At eval time, use the tree's greedy action (best empirical advantage).
        Falls back to network if a leaf has never been visited.
        """
        obs_flat = obs.flatten().astype(np.float32)
        leaf = self.tree.find_leaf(obs_flat)

        if deterministic:
            if leaf.stats.total_visits > 0:
                return leaf.stats.best_action()
            else:
                # Fallback to network
                self.net.eval()
                obs_t = torch.from_numpy(obs_flat).unsqueeze(0)
                with torch.no_grad():
                    dist = self.net.policy(obs_t)
                    return dist.probs.argmax(dim=-1).item()
        else:
            return leaf.stats.ucb_action(self.ucb_coef)


# ===========================================================================
# Environment factory
# ===========================================================================

def make_highway_env():
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "DiscreteAction",
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": 5,
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

    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    print(f"Obs shape: {obs_shape} -> flat dim: {obs_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Features: 5 vehicles x 5 features (presence, x, y, vx, vy) = 25 dims")
    print()

    agent = ZoomingTreePPO(
        env,
        # Network / PPO
        hidden=[256, 256],
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        lr=5e-4,
        gamma=0.8,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        # Zooming / tree
        ucb_coef=1.5,
        tree_max_leaves=128,
        tree_min_samples_split=30,
        tree_min_var_reduction=0.005,
        tree_max_depth=12,
        refine_interval=5,
        top_k_dims=8,
        tree_action_prob=0.7,
        # General
        seed=42,
    )

    print("Starting Zooming-Tree PPO training...")
    print("=" * 60)
    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=1000)

    # Final tree summary
    print()
    print("=" * 60)
    print("Final tree state:")
    print(f"  {agent.tree.summary()}")
    dim_usage = agent.tree.dimension_usage()
    if dim_usage:
        # Map dimension indices to feature names for interpretability
        feature_names = []
        vehicles = ["ego", "vehicle_1", "vehicle_2", "vehicle_3", "vehicle_4"]
        features = ["presence", "x", "y", "vx", "vy"]
        for v in vehicles:
            for f in features:
                feature_names.append(f"{v}_{f}")
        print("  Dimensions used for splitting:")
        for dim, count in dim_usage.items():
            name = feature_names[dim] if dim < len(feature_names) else f"dim_{dim}"
            print(f"    {name} (dim {dim}): {count} splits")

    # Evaluate
    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        eval_rewards.append(total_reward)

    print(f"Eval over 20 episodes/playthroughs: mean={np.mean(eval_rewards):.2f}, std={np.std(eval_rewards):.2f}")

    env.close()