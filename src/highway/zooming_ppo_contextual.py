"""
Contextual Zooming PPO: State-dependent adaptive action discretization.

Extends ZoomingPPO with a StatePartitionTree that adaptively partitions the
observation space.  Each leaf of the state tree owns:
  - Its own ActionZooming instance (independent action resolution)
  - Its own policy output head in the network

The shared policy backbone maps observations to hidden features.  The
appropriate per-leaf head then maps features to logits over that leaf's
action set.  The value network is global (shared across all leaves).

State splits are triggered by KL divergence of action distributions across
candidate axis-aligned observation splits.  Action splits follow the
original zooming play-count criterion, independently per leaf.

References:
  - Kleinberg, Slivkins & Upfal (2008). Multi-Armed Bandits in Metric Spaces.
  - Slivkins (2014). Contextual Bandits with Similarity Information.
  - Sinclair, Banerjee & Yu (2019). Adaptive Discretization for Episodic RL.
  - Lakshminarayanan, Roy & Teh (2014). Mondrian Forests.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

from zooming import Cube, CubeStats
from state_partition import (
    StatePartitionTree,
    StateLeaf,
    ActionZooming,
    ActionSplitInfo,
)
from ppo import CustomRewardWrapper


# ---------------------------------------------------------------------------
# ContextualActorCritic — shared backbone + per-leaf policy heads
# ---------------------------------------------------------------------------

class ContextualActorCritic(nn.Module):
    """
    Actor-critic with per-leaf policy output heads.

    Architecture:
      - Shared policy backbone: obs -> hidden features  (persistent)
      - Per-leaf policy heads: hidden -> logits over that leaf's action set
      - Shared value network: obs -> scalar value  (global, no per-leaf split)

    When a state leaf splits, we create two new heads (copying parent weights
    + noise) and remove the parent head.  When an action cube splits within
    a leaf, we rebuild that leaf's head (same logic as original ZoomingPPO).
    """

    def __init__(
        self,
        obs_dim: int,
        leaf_action_counts: Dict[int, int],
        hidden: List[int] = [256, 256],
    ):
        super().__init__()
        self.hidden_dim = hidden[-1]
        self.obs_dim = obs_dim

        # --- Shared policy backbone ---
        layers_pi: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_pi += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.pi_hidden = nn.Sequential(*layers_pi)

        # --- Per-leaf policy heads (nn.ModuleDict for proper parameter tracking) ---
        self.pi_heads = nn.ModuleDict()
        for leaf_id, n_actions in leaf_action_counts.items():
            self.pi_heads[str(leaf_id)] = nn.Linear(self.hidden_dim, n_actions)

        # --- Shared value network ---
        layers_vf: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_vf += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers_vf.append(nn.Linear(prev, 1))
        self.vf = nn.Sequential(*layers_vf)

    # ------------------------------------------------------------------
    # Head management
    # ------------------------------------------------------------------

    def add_head(self, leaf_id: int, n_actions: int,
                 copy_from_leaf_id: Optional[int] = None):
        """
        Add a new per-leaf policy head.  Optionally copy weights from an
        existing head (for state-split inheritance) with small noise.
        """
        new_head = nn.Linear(self.hidden_dim, n_actions)

        if copy_from_leaf_id is not None:
            src_key = str(copy_from_leaf_id)
            if src_key in self.pi_heads:
                src = self.pi_heads[src_key]
                with torch.no_grad():
                    # Source and dest may have different n_actions if action
                    # sets differ, but at state-split time they're identical
                    n_copy = min(new_head.weight.shape[0], src.weight.shape[0])
                    new_head.weight.data[:n_copy] = src.weight.data[:n_copy] + \
                        torch.randn_like(src.weight.data[:n_copy]) * 0.01
                    new_head.bias.data[:n_copy] = src.bias.data[:n_copy] + \
                        torch.randn_like(src.bias.data[:n_copy]) * 0.01

        self.pi_heads[str(leaf_id)] = new_head

    def remove_head(self, leaf_id: int):
        """Remove a head (when its leaf becomes an internal node)."""
        key = str(leaf_id)
        if key in self.pi_heads:
            del self.pi_heads[key]

    def rebuild_leaf_head(self, leaf_id: int, new_n_actions: int,
                          splits: List[ActionSplitInfo], old_n_actions: int):
        """
        Rebuild a single leaf's policy head after action-cube splits.
        Same logic as original ZoomingActorCritic.rebuild_policy_head but
        applied to one specific leaf head.
        """
        key = str(leaf_id)
        old_head = self.pi_heads[key]
        old_weight = old_head.weight.data
        old_bias = old_head.bias.data

        new_head = nn.Linear(self.hidden_dim, new_n_actions)

        removed = set(s.old_idx for s in splits)
        surviving_old = [i for i in range(old_n_actions) if i not in removed]

        with torch.no_grad():
            for new_idx, old_idx in enumerate(surviving_old):
                new_head.weight.data[new_idx] = old_weight[old_idx]
                new_head.bias.data[new_idx] = old_bias[old_idx]

            for split in splits:
                parent_w = old_weight[split.old_idx]
                parent_b = old_bias[split.old_idx]
                for new_idx in split.new_indices:
                    new_head.weight.data[new_idx] = parent_w + \
                        torch.randn_like(parent_w) * 0.01
                    new_head.bias.data[new_idx] = parent_b + \
                        torch.randn_like(parent_b) * 0.01

        self.pi_heads[key] = new_head

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _get_hidden(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi_hidden(obs)

    def policy_for_leaf(self, obs: torch.Tensor, leaf_id: int) -> Categorical:
        """Get action distribution for a specific leaf."""
        h = self._get_hidden(obs)
        logits = self.pi_heads[str(leaf_id)](h)
        return Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.vf(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, leaf_id: int):
        """Sample action, return (action_idx, log_prob, value)."""
        dist = self.policy_for_leaf(obs, leaf_id)
        action = dist.sample()
        return action, dist.log_prob(action), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor,
                 leaf_ids: torch.Tensor):
        """
        Evaluate log_prob, entropy, value for a batch with mixed leaf_ids.

        This is the tricky part: different rows in the batch may belong to
        different leaves (different heads, different action counts).
        We group by leaf_id and process each group through its head.
        """
        h = self._get_hidden(obs)
        values = self.vf(obs).squeeze(-1)

        batch_size = obs.shape[0]
        log_probs = torch.zeros(batch_size)
        entropies = torch.zeros(batch_size)

        unique_leaves = torch.unique(leaf_ids)
        for lid_tensor in unique_leaves:
            lid = lid_tensor.item()
            mask = (leaf_ids == lid)
            indices = mask.nonzero(as_tuple=True)[0]

            key = str(lid)
            if key not in self.pi_heads:
                # This leaf was split since the rollout was collected.
                # Fall back: use uniform log_prob (these samples are stale).
                # This is rare and only happens for one update cycle.
                log_probs[indices] = 0.0
                entropies[indices] = 0.0
                continue

            logits = self.pi_heads[key](h[indices])
            dist = Categorical(logits=logits)

            acts = actions[indices]
            # Clamp action indices to valid range for this head
            max_act = logits.shape[-1] - 1
            acts = acts.clamp(0, max_act)

            log_probs[indices] = dist.log_prob(acts)
            entropies[indices] = dist.entropy()

        return log_probs, entropies, values


# ---------------------------------------------------------------------------
# Rollout buffer (extended with leaf_id)
# ---------------------------------------------------------------------------

class ContextualRolloutBuffer:
    """Stores one rollout with leaf_id per step."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.leaf_ids: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def store(self, obs, action, leaf_id, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.leaf_ids.append(leaf_id)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ---------------------------------------------------------------------------
# ContextualZoomingPPO — main algorithm
# ---------------------------------------------------------------------------

class ContextualZoomingPPO:
    def __init__(
        self,
        env: gym.Env,
        hidden: List[int] = [256, 256],
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        # State partition parameters
        max_tree_depth: int = 8,
        min_samples_split: int = 128,
        kl_threshold: float = 0.1,
        seed: int = 0,
    ):
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.hidden = hidden

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim = int(np.prod(env.observation_space.shape))

        # State partition tree (starts as single leaf)
        self.tree = StatePartitionTree(
            obs_dim=self.obs_dim,
            da=1,
            max_depth=max_tree_depth,
            min_samples_split=min_samples_split,
            kl_threshold=kl_threshold,
        )

        # Build network with initial leaf layout
        leaf_action_counts = {
            lf.leaf_id: lf.zooming.n_actions
            for lf in self.tree.all_leaves()
        }
        self.net = ContextualActorCritic(self.obs_dim, leaf_action_counts, hidden)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.buffer = ContextualRolloutBuffer()
        self._obs, _ = env.reset(seed=seed)
        self._done = False

    # ------------------------------------------------------------------
    # GAE (unchanged)
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
    # Rollout collection (now routes through state partition tree)
    # ------------------------------------------------------------------

    def _collect_rollouts(self):
        self.buffer.clear()
        self.tree.clear_all_buffers()
        self.net.eval()

        for _ in range(self.n_steps):
            obs_flat = self._obs.flatten().astype(np.float32)
            obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

            # Route observation through state partition tree
            leaf = self.tree.get_leaf(obs_flat)

            with torch.no_grad():
                action_idx, log_prob, value = self.net.act(obs_t, leaf.leaf_id)

            a_idx = action_idx.item()
            lp = log_prob.item()
            v = value.item()

            # Map action index -> env action via this leaf's zooming
            env_action = leaf.zooming.get_env_action(a_idx)

            # Record in leaf's buffer for split evaluation
            leaf.record(obs_flat, a_idx)

            next_obs, reward, done, truncated, info = self.env.step(env_action)
            self.buffer.store(obs_flat, a_idx, leaf.leaf_id,
                              reward, done or truncated, lp, v)

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
    # PPO update (handles mixed leaf_ids in batch)
    # ------------------------------------------------------------------

    def _update(self, advantages: np.ndarray, returns: np.ndarray):
        self.net.train()

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        act_t = torch.tensor(self.buffer.actions, dtype=torch.long)
        leaf_t = torch.tensor(self.buffer.leaf_ids, dtype=torch.long)
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

                new_lp, entropy, values = self.net.evaluate(
                    obs_t[mb], act_t[mb], leaf_t[mb]
                )

                ratio = torch.exp(new_lp - old_lp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                     1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, ret_t[mb])
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                          self.max_grad_norm)
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Action zooming splits (per-leaf, independent)
    # ------------------------------------------------------------------

    def _check_action_splits(self):
        """
        For each leaf, update play counts from the rollout buffer entries
        belonging to that leaf, then try action splits.
        """
        # Group buffer actions by leaf_id
        leaf_actions: Dict[int, List[int]] = {}
        for act, lid in zip(self.buffer.actions, self.buffer.leaf_ids):
            leaf_actions.setdefault(lid, []).append(act)

        for leaf in self.tree.all_leaves():
            actions_for_leaf = leaf_actions.get(leaf.leaf_id, [])
            if not actions_for_leaf:
                continue

            leaf.zooming.update_play_counts(actions_for_leaf)

            old_n = leaf.zooming.n_actions
            splits = leaf.zooming.try_split()

            if splits:
                new_n = leaf.zooming.n_actions
                self._total_action_splits += len(splits)
                self.net.rebuild_leaf_head(
                    leaf.leaf_id, new_n, splits, old_n
                )

    # ------------------------------------------------------------------
    # State partition splits
    # ------------------------------------------------------------------

    def _check_state_splits(self):
        """
        Evaluate all leaves for state-space splits.  For each split:
          1. Two new leaves are created (inheriting parent's zooming)
          2. Two new network heads are created (inheriting parent's weights)
          3. Parent head is removed
        """
        state_splits = self.tree.try_split_leaves()

        for old_lid, left_lid, right_lid in state_splits:
            self._total_state_splits += 1

            # Find the new leaves to get their action counts
            lid_map = self.tree.leaf_id_to_leaf()
            left_leaf = lid_map[left_lid]
            right_leaf = lid_map[right_lid]

            # Create new heads inheriting from parent
            self.net.add_head(left_lid, left_leaf.zooming.n_actions,
                              copy_from_leaf_id=old_lid)
            self.net.add_head(right_lid, right_leaf.zooming.n_actions,
                              copy_from_leaf_id=old_lid)

            # Remove parent head
            self.net.remove_head(old_lid)

    def _check_and_split(self):
        """Run both action splits and state splits, then rebuild optimizer."""
        old_param_count = sum(p.numel() for p in self.net.parameters())

        self._check_action_splits()
        self._check_state_splits()

        new_param_count = sum(p.numel() for p in self.net.parameters())
        if new_param_count != old_param_count:
            # Parameters changed, rebuild optimizer
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _action_summary(self) -> str:
        """Summarize action cube sizes across all leaves."""
        all_sizes = []
        for lf in self.tree.all_leaves():
            all_sizes.extend(c.s for c in lf.zooming.active_cubes)
        counts = Counter(all_sizes)
        parts = [f"s={s:.3f}:{n}" for s, n in sorted(counts.items(), reverse=True)]
        return " ".join(parts)

    def _total_actions(self) -> int:
        return sum(lf.zooming.n_actions for lf in self.tree.all_leaves())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 10_000):
        steps_done = 0
        episode_rewards: List[float] = []
        current_ep_reward = 0.0
        self._total_action_splits = 0
        self._total_state_splits = 0
        last_print = 0

        while steps_done < total_timesteps:
            advantages, returns = self._collect_rollouts()
            self._update(advantages, returns)
            self._check_and_split()

            for r, d in zip(self.buffer.rewards, self.buffer.dones):
                current_ep_reward += r
                if d:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

            steps_done += self.n_steps

            if steps_done // print_every > last_print // print_every:
                last_print = steps_done
                recent = episode_rewards[-50:] if episode_rewards else [0]
                print(
                    f"[{steps_done:>7d}/{total_timesteps}] "
                    f"ep={len(episode_rewards):>4d}  "
                    f"reward(last50)={np.mean(recent):>7.2f}  "
                    f"state_tree=[{self.tree.summary()}]  "
                    f"total_actions={self._total_actions():>3d}  "
                    f"splits(state={self._total_state_splits},"
                    f"action={self._total_action_splits})  "
                    f"cubes=[{self._action_summary()}]"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.net.eval()
        obs_flat = obs.flatten().astype(np.float32)
        obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

        leaf = self.tree.get_leaf(obs_flat)

        with torch.no_grad():
            dist = self.net.policy_for_leaf(obs_t, leaf.leaf_id)
            if deterministic:
                action_idx = dist.probs.argmax(dim=-1).item()
            else:
                action_idx = dist.sample().item()

        return leaf.zooming.get_env_action(action_idx)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: List[float] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Serialize tree structure
        tree_data = {
            "split_dim": self.tree._split_dim,
            "split_val": self.tree._split_val,
            "left": self.tree._left,
            "right": self.tree._right,
            "depth": self.tree._depth,
            "next_leaf_id": self.tree._next_leaf_id,
            "obs_dim": self.tree.obs_dim,
            "da": self.tree.da,
            "max_depth": self.tree.max_depth,
            "min_samples_split": self.tree.min_samples_split,
            "kl_threshold": self.tree.kl_threshold,
        }

        # Serialize leaves (node_id -> leaf data)
        leaves_data = {}
        for node_id, lf in enumerate(self.tree._leaves):
            if lf is not None:
                zooming_cubes = [
                    {"lower": c.lower.tolist(), "s": c.s, "d": c.d,
                     "n_play": s.n_play}
                    for c, s in zip(lf.zooming.active_cubes, lf.zooming.stats)
                ]
                leaves_data[node_id] = {
                    "leaf_id": lf.leaf_id,
                    "zooming": zooming_cubes,
                }

        torch.save({
            "net_state_dict": self.net.state_dict(),
            "episode_rewards": episode_rewards or [],
            "obs_dim": self.obs_dim,
            "hidden": self.hidden,
            "tree_data": tree_data,
            "leaves_data": leaves_data,
        }, path)
        print(f"Saved ContextualZoomingPPO checkpoint to {path}")

    @classmethod
    def load(cls, path: str, env: "gym.Env") -> "ContextualZoomingPPO":
        data = torch.load(path, weights_only=False)
        agent = cls(env, hidden=data["hidden"])

        # Reconstruct tree structure
        td = data["tree_data"]
        agent.tree._split_dim = td["split_dim"]
        agent.tree._split_val = td["split_val"]
        agent.tree._left = td["left"]
        agent.tree._right = td["right"]
        agent.tree._depth = td["depth"]
        agent.tree._next_leaf_id = td["next_leaf_id"]

        # Reconstruct leaves
        agent.tree._leaves = [None] * len(agent.tree._split_dim)
        for node_id_str, ld in data["leaves_data"].items():
            node_id = int(node_id_str)
            zooming = ActionZooming(da=td["da"])
            zooming.active_cubes = []
            zooming.stats = []
            for cs in ld["zooming"]:
                cube = Cube(lower=np.array(cs["lower"]), s=cs["s"], d=cs["d"])
                zooming.active_cubes.append(cube)
                zooming.stats.append(CubeStats(Q=0.0, n_play=cs["n_play"]))
            leaf = StateLeaf(leaf_id=ld["leaf_id"], zooming=zooming)
            agent.tree._leaves[node_id] = leaf

        # Rebuild network with correct leaf layout
        leaf_action_counts = {
            lf.leaf_id: lf.zooming.n_actions
            for lf in agent.tree.all_leaves()
        }
        agent.net = ContextualActorCritic(
            data["obs_dim"], leaf_action_counts, data["hidden"]
        )
        agent.net.load_state_dict(data["net_state_dict"])
        return agent


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env_continuous():
    """Highway env with ContinuousAction (steering only)."""
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
        },
    )
    return CustomRewardWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 100_000
    env = make_highway_env_continuous()

    agent = ContextualZoomingPPO(
        env,
        hidden=[256, 256],
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        # State partition params
        max_tree_depth=8,
        min_samples_split=128,
        kl_threshold=0.1,
        seed=42,
    )

    print("Starting Contextual Zooming PPO training...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Obs dim (flat): {agent.obs_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Initial state leaves: {agent.tree.n_leaves}")
    print(f"Initial actions per leaf: "
          f"{[lf.zooming.n_actions for lf in agent.tree.all_leaves()]}")
    print()

    episode_rewards = agent.learn(
        total_timesteps=TOTAL_TIMESTEPS, print_every=10000
    )

    agent.save("checkpoints/contextual_zooming_ppo.pt", episode_rewards)

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

    print(f"Eval over 20 episodes: "
          f"mean={np.mean(eval_rewards):.2f}, std={np.std(eval_rewards):.2f}")
    print(f"Final state tree: {agent.tree.summary()}")
    print(f"Final total actions: {agent._total_actions()}")

    env.close()