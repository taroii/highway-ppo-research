"""
DQN over a discrete action grid (uniform or zooming).

Design:
  - Shared MLP trunk over flat obs + a single linear Q-head whose output
    dim equals the grid's current n_actions.
  - On a zooming split, the head is rebuilt: child rows inherit their
    parent's Q-row + small noise; survivor rows are copied over.
  - The target network is rebuilt the same way, then its child rows are
    snapped to match the online network's child rows (survivor rows are
    left alone so the slow-averaging target signal is preserved).
  - Adam state for the shared trunk is preserved across head rebuilds.
  - Action selection is a pluggable policy: ``EpsGreedy`` (uniform arms)
    or ``UCB`` (zooming arms, uses per-cube play counts as the
    confidence-bonus denominator).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.highway.action_manager import ActionGrid
from src.highway.zooming import SplitInfo


# ---------------------------------------------------------------------------
# Action-selection policies
# ---------------------------------------------------------------------------

class SelectionPolicy(Protocol):
    def select(self, q_values: np.ndarray, step: int,
               play_counts: np.ndarray) -> int: ...


class EpsGreedy:
    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.05,
                 decay_steps: int = 60_000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps

    def select(self, q_values, step, play_counts):
        frac = min(1.0, step / max(1, self.decay_steps))
        eps = self.eps_start + (self.eps_end - self.eps_start) * frac
        if np.random.random() < eps:
            return int(np.random.randint(len(q_values)))
        return int(q_values.argmax())


class UCB:
    """Q(s, a) + c(step) * sqrt(log(total) / n(a)).

    ``c`` anneals linearly from ``c_start`` to ``c_end`` over
    ``decay_steps``.  Without annealing, UCB keeps exploring forever —
    every freshly-split cube has n=0 and dominates argmax, which on
    racetrack means forced steering extremes that crash the car.
    """

    def __init__(self, c_start: float = 0.3, c_end: float = 0.03,
                 decay_steps: int = 60_000):
        self.c_start = c_start
        self.c_end = c_end
        self.decay_steps = decay_steps

    def _c(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.decay_steps))
        return self.c_start + (self.c_end - self.c_start) * frac

    def select(self, q_values, step, play_counts):
        c = self._c(step)
        total = max(2, int(play_counts.sum()))
        with np.errstate(divide="ignore"):
            bonus = c * np.sqrt(np.log(total) / np.maximum(1, play_counts))
        return int((q_values + bonus).argmax())


# ---------------------------------------------------------------------------
# Q-network: shared trunk + single output head
# ---------------------------------------------------------------------------

def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_actions)
        self.apply(weight_init)

    @property
    def n_actions(self) -> int:
        return self.head.out_features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))

    def rebuild_head(self, new_n: int, splits: List[SplitInfo]) -> None:
        """Resize the head to ``new_n``; transfer surviving rows; init
        children from their parent's row + small noise."""
        old = self.head
        old_w, old_b = old.weight.data, old.bias.data
        new = nn.Linear(old.in_features, new_n)
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

        self.head = new


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class DQNReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.not_dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        np.copyto(self.obs[i], obs)
        np.copyto(self.next_obs[i], next_obs)
        self.actions[i] = action
        self.rewards[i] = reward
        self.not_dones[i] = 0.0 if done else 1.0
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        limit = self.capacity if self.full else self.idx
        idx = np.random.randint(0, limit, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.not_dones[idx]),
        )

    def __len__(self):
        return self.capacity if self.full else self.idx


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class DQN:
    def __init__(
        self,
        env: gym.Env,
        grid: ActionGrid,
        selection_policy: SelectionPolicy,
        hidden_dim: int = 256,
        gamma: float = 0.9,
        tau: float = 0.01,
        lr: float = 5e-4,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        learning_starts: int = 2000,
        target_update_freq: int = 2,
        split_check_freq: int = 2000,
        split_delay: int = 30_000,
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

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu")
        self.obs_dim = int(np.prod(env.observation_space.shape))

        self.q = QNetwork(self.obs_dim, grid.n_actions, hidden_dim).to(self.device)
        self.q_target = QNetwork(self.obs_dim, grid.n_actions, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = DQNReplayBuffer(self.obs_dim, buffer_capacity)
        self.total_splits = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, step: int,
                      deterministic: bool = False) -> Tuple[int, np.ndarray]:
        obs_t = torch.as_tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(obs_t).squeeze(0).cpu().numpy()
        if deterministic:
            local_idx = int(q_values.argmax())
        else:
            local_idx = self.policy.select(q_values, step, self.grid.play_counts())
        return local_idx, self.grid.get_env_action(local_idx)

    # ------------------------------------------------------------------
    # Q update
    # ------------------------------------------------------------------

    def _update(self):
        obs, actions, rewards, next_obs, not_dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            q_next = self.q_target(next_obs).max(dim=1).values
            target = rewards + not_dones * self.gamma * q_next

        q_taken = self.q(obs).gather(1, actions.view(-1, 1)).squeeze(1)
        loss = F.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Split handling
    # ------------------------------------------------------------------

    def _sync_target_children(self, splits: List[SplitInfo]) -> None:
        """Snap target-Q child rows to match online-Q child rows so
        post-split TD targets agree with online Q. Survivor rows keep the
        target's slow-averaged values."""
        with torch.no_grad():
            for split in splits:
                for new_idx in split.new_indices:
                    self.q_target.head.weight.data[new_idx] = self.q.head.weight.data[new_idx]
                    self.q_target.head.bias.data[new_idx] = self.q.head.bias.data[new_idx]

    def _rebuild_optimizer_preserving_state(self) -> None:
        """Rebuild Adam over current parameters, preserving state for
        params whose tensor identity survived (i.e., the shared trunk —
        only the replaced head's parameters lose their Adam state)."""
        old_state = dict(self.optimizer.state)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        for p in self.q.parameters():
            if p in old_state:
                self.optimizer.state[p] = old_state[p]

    def _check_splits(self) -> int:
        splits = self.grid.try_split()
        if not splits:
            return 0
        self.q.rebuild_head(self.grid.n_actions, splits)
        self.q_target.rebuild_head(self.grid.n_actions, splits)
        self._sync_target_children(splits)
        self._rebuild_optimizer_preserving_state()
        return len(splits)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 10_000) -> List[float]:
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        current = 0.0

        for step in range(1, total_timesteps + 1):
            obs_flat = obs.flatten().astype(np.float32)
            local_idx, env_action = self.select_action(obs_flat, step)
            next_obs, reward, done, truncated, _ = self.env.step(env_action)
            terminal = done or truncated
            next_obs_flat = next_obs.flatten().astype(np.float32)

            self.buffer.add(obs_flat, local_idx, reward, next_obs_flat, terminal)
            self.grid.register_play(local_idx)
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
                      f"n_actions={self.grid.n_actions}  "
                      f"total_splits={self.total_splits}")

        return episode_rewards

    # ------------------------------------------------------------------
    # Save / predict
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: Optional[List[float]] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q": self.q.state_dict(),
            "n_actions": self.grid.n_actions,
            "episode_rewards": episode_rewards or [],
        }, path)
        print(f"Saved DQN checkpoint to {path}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        _, env_action = self.select_action(obs.flatten().astype(np.float32),
                                           step=10**9, deterministic=deterministic)
        return env_action
