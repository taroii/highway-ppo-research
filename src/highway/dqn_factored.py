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
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.highway.dqn import SelectionPolicy, soft_update, weight_init
from src.highway.zooming import SplitInfo
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

    def rebuild_head_axis(self, axis: int, new_n: int,
                          splits: List[SplitInfo]) -> None:
        """Resize axis ``axis``'s head to ``new_n``; transfer surviving
        rows; init children from parent rows + small noise."""
        old = self.heads[axis]
        old_w, old_b = old.weight.data, old.bias.data
        new = nn.Linear(old.in_features, new_n).to(old.weight.device)
        removed = {s.old_idx for s in splits}
        surviving_old = [i for i in range(old.out_features) if i not in removed]
        with torch.no_grad():
            nn.init.orthogonal_(new.weight.data)
            new.bias.data.fill_(0.0)
            for new_idx, old_idx in enumerate(surviving_old):
                new.weight.data[new_idx] = old_w[old_idx]
                new.bias.data[new_idx] = old_b[old_idx]
            for split in splits:
                pw, pb = old_w[split.old_idx], old_b[split.old_idx]
                for new_idx in split.new_indices:
                    new.weight.data[new_idx] = pw + torch.randn_like(pw) * 0.01
                    new.bias.data[new_idx] = pb + torch.randn_like(pb) * 0.01
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
        idx_per_axis = np.empty(self.da, dtype=np.int64)
        for i, q in enumerate(q_lists):
            q_np = q.squeeze(0).cpu().numpy()
            if deterministic:
                idx_per_axis[i] = int(q_np.argmax())
            else:
                idx_per_axis[i] = self.policy.select(q_np, step, play_counts[i])
        return idx_per_axis, self.grid.get_env_action(idx_per_axis)

    # ------------------------------------------------------------------
    # Q update
    # ------------------------------------------------------------------

    def _update(self):
        obs, actions, rewards, next_obs, not_dones = self.buffer.sample(self.batch_size)
        # actions: (B, da) int64

        with torch.no_grad():
            next_q_list = self.q_target(next_obs)               # list of (B, n_i)
            next_max_per_axis = torch.stack(
                [q.max(dim=1).values for q in next_q_list], dim=1
            )                                                   # (B, da)
            next_v = next_max_per_axis.mean(dim=1)              # (B,)
            target = rewards + not_dones * self.gamma * next_v  # (B,)

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

    def _sync_target_axis_children(self, axis: int,
                                   splits: List[SplitInfo]) -> None:
        with torch.no_grad():
            for split in splits:
                for new_idx in split.new_indices:
                    self.q_target.heads[axis].weight.data[new_idx] = \
                        self.q.heads[axis].weight.data[new_idx]
                    self.q_target.heads[axis].bias.data[new_idx] = \
                        self.q.heads[axis].bias.data[new_idx]

    def _rebuild_optimizer_preserving_state(self) -> None:
        """Rebuild Adam over current parameters, preserving state for
        params whose tensor identity survived (the trunk + the heads we
        didn't replace this round)."""
        old_state = dict(self.optimizer.state)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        for p in self.q.parameters():
            if p in old_state:
                self.optimizer.state[p] = old_state[p]

    def _check_splits(self) -> int:
        per_axis_splits = self.grid.try_split()
        n_splits = sum(len(s) for s in per_axis_splits)
        if n_splits == 0:
            return 0
        for i, splits_i in enumerate(per_axis_splits):
            if not splits_i:
                continue
            new_n_i = self.grid.axes[i].n_actions
            self.q.rebuild_head_axis(i, new_n_i, splits_i)
            self.q_target.rebuild_head_axis(i, new_n_i, splits_i)
            self._sync_target_axis_children(i, splits_i)
        self._rebuild_optimizer_preserving_state()
        return n_splits

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 10_000) -> List[float]:
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        current = 0.0

        for step in range(1, total_timesteps + 1):
            obs_flat = obs.flatten().astype(np.float32)
            idx_per_axis, env_action = self.select_action(obs_flat, step)
            next_obs, reward, done, truncated, _ = self.env.step(env_action)
            terminal = done or truncated
            next_obs_flat = next_obs.flatten().astype(np.float32)

            self.buffer.add(obs_flat, idx_per_axis, reward, next_obs_flat, terminal)
            self.grid.register_play(idx_per_axis)
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
                self.total_splits += self._check_splits()

            if episode_rewards and step % print_every == 0:
                recent = episode_rewards[-50:]
                print(f"[{step:>7d}/{total_timesteps}]  ep={len(episode_rewards):>4d}  "
                      f"reward(last50)={np.mean(recent):>7.2f}  "
                      f"n_per_axis={self.grid.n_per_axis()}  "
                      f"total_splits={self.total_splits}")

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
        }, path)
        print(f"Saved BranchingDQN checkpoint to {path}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        _, env_action = self.select_action(
            obs.flatten().astype(np.float32),
            step=10**9, deterministic=deterministic,
        )
        return env_action
