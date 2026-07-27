"""
Branching DQN over a factored action grid.

Mirrors src/highway/dqn.py but for ``FactoredActionZooming``:

  - Q-net: shared trunk + ``da`` independent linear heads, head ``i``
    sized to that axis's current bin count.
  - Replay buffer stores ``da`` action indices per transition.
  - TD target follows the action-branching recipe (Tavakoli, Pardo,
    Kormushev, AAAI 2018):
        target = r + gamma * (1/da) * sum_i max_{a'_i} Q_target_i(s', a'_i)
    The per-axis Bellman losses are summed.
  - On a split in axis ``i`` only that head is rebuilt; child rows are
    warm-started from their parent (parent_row + small noise), and the
    target head's child rows are snapped to the online head's children.

Action-selection policies (``EpsGreedy``, ``UCB``) and the
``soft_update``/``weight_init`` helpers are reused from ``dqn.py``.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.highway.dqn import SelectionPolicy, soft_update, weight_init
from src.highway.eval_utils import evaluate_policy
from src.highway.zooming_factored import FactoredActionZooming


# ---------------------------------------------------------------------------
# Branching Q-network
# ---------------------------------------------------------------------------

class BranchingQNetwork(nn.Module):
    """Shared trunk + one linear head per action axis."""

    def __init__(self, obs_dim: int, n_per_axis: List[int],
                 hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in n_per_axis]
        )
        self.apply(weight_init)

    def n_per_axis(self) -> List[int]:
        return [h.out_features for h in self.heads]

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(obs)
        return [head(h) for head in self.heads]

    def grow_head_axis(self, axis: int, add_n: int) -> None:
        """Append ``add_n`` rows to axis ``axis``'s head for freshly-created
        buffering children.  Existing rows keep their exact positions (pure
        append -- no survivor reindexing); the new rows are freshly
        initialized (NO parent inheritance, per the paper's erratum: a
        buffering child must accrue its own evidence and is unselectable
        until it graduates, so its arbitrary starting Q can never be
        exploited)."""
        old = self.heads[axis]
        new = nn.Linear(old.in_features, old.out_features + add_n).to(old.weight.device)
        with torch.no_grad():
            nn.init.orthogonal_(new.weight.data)
            new.bias.data.fill_(0.0)
            new.weight.data[:old.out_features] = old.weight.data
            new.bias.data[:old.out_features] = old.bias.data
        self.heads[axis] = new

    def shrink_head_axis(self, axis: int, removed_indices: List[int]) -> None:
        """Drop the rows at ``removed_indices`` (retired parent cubes) and
        compact the survivors, preserving order.  Mirrors the grid's
        ``_retire`` so head row k continues to correspond to cube k."""
        old = self.heads[axis]
        removed = set(removed_indices)
        survivors = [k for k in range(old.out_features) if k not in removed]
        new = nn.Linear(old.in_features, len(survivors)).to(old.weight.device)
        with torch.no_grad():
            for new_i, old_i in enumerate(survivors):
                new.weight.data[new_i] = old.weight.data[old_i]
                new.bias.data[new_i] = old.bias.data[old_i]
        self.heads[axis] = new


# ---------------------------------------------------------------------------
# Replay buffer (multi-axis actions)
# ---------------------------------------------------------------------------

class BranchingReplayBuffer:
    def __init__(self, obs_dim: int, da: int, capacity: int,
                 device: torch.device = torch.device("cpu")):
        self.capacity = capacity
        self.da = da
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, da), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.not_dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action_per_axis, reward, next_obs, done):
        i = self.idx
        np.copyto(self.obs[i], obs)
        np.copyto(self.next_obs[i], next_obs)
        self.actions[i] = action_per_axis
        self.rewards[i] = reward
        self.not_dones[i] = 0.0 if done else 1.0
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def remap_axis(self, axis: int, remap: np.ndarray) -> None:
        """Rewrite stored action indices on ``axis`` through ``remap`` (old
        index -> new index) after the grid retired/compacted cubes on that
        axis, so buffered transitions never point at a stale row."""
        limit = self.capacity if self.full else self.idx
        if limit == 0:
            return
        self.actions[:limit, axis] = remap[self.actions[:limit, axis]]

    def sample(self, batch_size: int):
        limit = self.capacity if self.full else self.idx
        idx = np.random.randint(0, limit, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.not_dones[idx], device=self.device),
        )

    def __len__(self):
        return self.capacity if self.full else self.idx


# ---------------------------------------------------------------------------
# Branching DQN agent
# ---------------------------------------------------------------------------

class BranchingDQN:
    def __init__(
        self,
        env: gym.Env,
        grid: FactoredActionZooming,
        selection_policy: SelectionPolicy,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 256,
        buffer_capacity: int = 1_000_000,
        learning_starts: int = 10_000,
        target_update_freq: int = 2,
        split_check_freq: int = 2000,
        split_delay: int = 60_000,
        buffer_period: int = 16,
        seed: int = 0,
    ):
        self.env = env
        self.grid = grid
        self.policy = selection_policy
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.split_check_freq = split_check_freq
        self.split_delay = split_delay
        # Buffering service period: every ``buffer_period``-th play of a
        # parent with buffering children, the sample is redirected to a
        # child (relabel-by-containment).  Adapts the paper's ``H+1`` (which
        # came from its tabular alpha_t=(H+1)/(H+t) rule, unused here).
        self.buffer_period = buffer_period
        self.hidden_dim = hidden_dim
        self.da = grid.da

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = int(np.prod(env.observation_space.shape))

        n_per_axis = grid.n_per_axis()
        self.q = BranchingQNetwork(self.obs_dim, n_per_axis, hidden_dim).to(self.device)
        self.q_target = BranchingQNetwork(self.obs_dim, n_per_axis, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = BranchingReplayBuffer(self.obs_dim, self.da, buffer_capacity,
                                            self.device)
        self.total_splits = 0
        # (timestep, axis_idx, n_splits_on_axis) for each split event.
        self.split_history: List[Tuple[int, int, int]] = []
        # (step, deterministic-eval mean, std) sampled during learn().
        self.eval_curve: List[Tuple[int, float, float]] = []
        # wall-clock accounting (set by learn()); train_seconds excludes the
        # periodic deterministic-eval episodes so it reflects training compute.
        self.train_seconds = 0.0
        self.wall_seconds = 0.0
        self.eval_seconds = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, step: int,
                      deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs.flatten(), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_lists = self.q(obs_t)
        play_counts = self.grid.play_counts_per_axis()
        masks = self.grid.active_mask_per_axis()   # buffering cubes excluded
        idx_per_axis = np.empty(self.da, dtype=np.int64)
        for i, q in enumerate(q_lists):
            q_np = q.squeeze(0).cpu().numpy()
            mask = masks[i]
            if deterministic:
                scores = np.where(mask, q_np, -np.inf)
                idx_per_axis[i] = int(scores.argmax())
            else:
                idx_per_axis[i] = self.policy.select(q_np, step, play_counts[i], mask)
        return idx_per_axis, self.grid.get_env_action(idx_per_axis)

    # ------------------------------------------------------------------
    # Q update
    # ------------------------------------------------------------------

    def _update(self):
        obs, actions, rewards, next_obs, not_dones = self.buffer.sample(self.batch_size)
        # actions: (B, da) int64

        with torch.no_grad():
            next_q_list = self.q_target(next_obs)               # list of (B, n_i)
            masks = self.grid.active_mask_per_axis()
            per_axis_max = []
            for i, q in enumerate(next_q_list):
                m = torch.as_tensor(masks[i], dtype=torch.bool, device=q.device)
                if bool(m.any()):
                    # exclude buffering rows from the bootstrap max: their
                    # Q is arbitrary until they graduate
                    q = q.masked_fill(~m.unsqueeze(0), float("-inf"))
                per_axis_max.append(q.max(dim=1).values)
            next_max_per_axis = torch.stack(per_axis_max, dim=1)  # (B, da)
            next_v = next_max_per_axis.mean(dim=1)                # (B,)
            target = rewards + not_dones * self.gamma * next_v    # (B,)

        q_list = self.q(obs)                                    # list of (B, n_i)
        loss = torch.zeros((), dtype=q_list[0].dtype, device=q_list[0].device)
        for i, q in enumerate(q_list):
            q_taken = q.gather(1, actions[:, i:i + 1]).squeeze(1)
            loss = loss + F.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Split handling
    # ------------------------------------------------------------------

    def _sync_target_new_rows(self, axis: int, add_n: int) -> None:
        """Snap the target head's freshly-appended rows to match the online
        head so post-split TD targets agree (survivor rows keep the target's
        slow-averaged values)."""
        with torch.no_grad():
            n = self.q.heads[axis].out_features
            for r in range(n - add_n, n):
                self.q_target.heads[axis].weight.data[r] = \
                    self.q.heads[axis].weight.data[r]
                self.q_target.heads[axis].bias.data[r] = \
                    self.q.heads[axis].bias.data[r]

    def _rebuild_optimizer_preserving_state(self) -> None:
        """Rebuild Adam over current parameters, preserving state for
        params whose tensor identity survived (the trunk + the heads we
        didn't replace this round)."""
        old_state = dict(self.optimizer.state)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        for p in self.q.parameters():
            if p in old_state:
                self.optimizer.state[p] = old_state[p]

    def _maintain(self, step: int) -> int:
        """Advance the buffering lifecycle and apply the resulting Q-head
        edits.  Per axis: retire rows (shrink) first, then append new
        buffering child rows (grow) -- the same order the grid mutates its
        cube lists, so head row k stays aligned with cube k.  Returns the
        number of splits (child groups created) this round."""
        removals, appends, remaps = self.grid.maintain()
        changed = False
        n_splits = 0
        for i in range(self.da):
            if removals[i]:
                # remap buffered action indices BEFORE the head shrinks, so a
                # retired parent's transitions follow onto its child rows
                self.buffer.remap_axis(i, remaps[i])
                self.q.shrink_head_axis(i, removals[i])
                self.q_target.shrink_head_axis(i, removals[i])
                changed = True
            if appends[i]:
                self.q.grow_head_axis(i, appends[i])
                self.q_target.grow_head_axis(i, appends[i])
                self._sync_target_new_rows(i, appends[i])
                # 1-D children come in pairs -> appends[i] // 2 split events
                n_splits += appends[i] // 2
                self.split_history.append((step, i, appends[i] // 2))
                changed = True
        if changed:
            self._rebuild_optimizer_preserving_state()
        return n_splits

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_eval(self, eval_env, eval_episodes: int, step: int) -> None:
        mean, std = evaluate_policy(
            lambda o: self.predict(o, deterministic=True),
            eval_env, eval_episodes,
        )
        self.eval_curve.append((step, mean, std))

    def learn(self, total_timesteps: int, print_every: int = 10_000,
              eval_env: Optional[gym.Env] = None,
              eval_every: int = 10_000, eval_episodes: int = 10) -> List[float]:
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        self.eval_curve = []
        self.eval_seconds = 0.0
        current = 0.0
        _t_start = time.perf_counter()

        for step in range(1, total_timesteps + 1):
            obs_flat = obs.flatten().astype(np.float32)
            # Select over ACTIVE cubes only; on_step registers the play and,
            # on a buffering-service step, redirects to a child (returning the
            # child's action + the index to store the transition under).
            idx_per_axis, _ = self.select_action(obs_flat, step)
            env_action, store_idx = self.grid.on_step(idx_per_axis, self.buffer_period)
            next_obs, reward, done, truncated, _ = self.env.step(env_action)
            terminal = done or truncated
            next_obs_flat = next_obs.flatten().astype(np.float32)

            self.buffer.add(obs_flat, store_idx, reward, next_obs_flat, terminal)
            current += reward

            if terminal:
                episode_rewards.append(current)
                current = 0.0
                next_obs, _ = self.env.reset()
            obs = next_obs

            if step >= self.learning_starts:
                self._update()
                if step % self.target_update_freq == 0:
                    soft_update(self.q, self.q_target, self.tau)

            if step >= self.split_delay and step % self.split_check_freq == 0:
                self.total_splits += self._maintain(step)

            if eval_env is not None and (step % eval_every == 0
                                         or step == total_timesteps):
                _t_eval = time.perf_counter()
                self._run_eval(eval_env, eval_episodes, step)
                self.eval_seconds += time.perf_counter() - _t_eval
                e_step, e_mean, e_std = self.eval_curve[-1]
                print(f"[{step:>7d}/{total_timesteps}]  "
                      f"eval({eval_episodes})={e_mean:>7.2f} +/- {e_std:<6.2f}  "
                      f"n_per_axis={self.grid.n_per_axis()}")

            if episode_rewards and step % print_every == 0:
                recent = episode_rewards[-50:]
                print(f"[{step:>7d}/{total_timesteps}]  ep={len(episode_rewards):>4d}  "
                      f"reward(last50)={np.mean(recent):>7.2f}  "
                      f"n_per_axis={self.grid.n_per_axis()}  "
                      f"total_splits={self.total_splits}")

        self.wall_seconds = time.perf_counter() - _t_start
        self.train_seconds = self.wall_seconds - self.eval_seconds
        print(f"[train complete] {total_timesteps} steps in "
              f"{self.train_seconds:.1f}s training "
              f"({total_timesteps / max(self.train_seconds, 1e-9):.0f} steps/s), "
              f"+{self.eval_seconds:.1f}s eval")
        return episode_rewards

    # ------------------------------------------------------------------
    # Save / predict
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: Optional[List[float]] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q": self.q.state_dict(),
            "n_per_axis": self.grid.n_per_axis(),
            "episode_rewards": episode_rewards or [],
            "eval_curve": self.eval_curve,
            "train_seconds": self.train_seconds,
            "wall_seconds": self.wall_seconds,
            "eval_seconds": self.eval_seconds,
            "split_history": self.split_history,
            "total_splits": self.total_splits,
            "total_cells": self.grid.total_cells,
            "total_budget": getattr(self.grid, "total_budget", None),
        }, path)
        print(f"Saved BranchingDQN checkpoint to {path}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        _, env_action = self.select_action(
            obs.flatten().astype(np.float32),
            step=10**9, deterministic=deterministic,
        )
        return env_action
