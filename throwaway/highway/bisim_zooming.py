r"""
Bisimulation-guided Clustered Zooming Pipeline.

Three-phase approach:
  Phase 1: Train BisimSAC to learn encoder \phi (bisimulation metric
           representation) on the continuous-action HighwayEnv.
  Phase 2: Embed replay buffer observations through \phi, cluster the
           latent space with k-means.
  Phase 3: Train k independent Zooming SAC instances--one per state
           cluster--each operating on the latent z produced by the
           frozen encoder.  Uses UCB optimism bonus in the actor
           objective to drive exploration of under-visited cubes,
           while keeping critic updates pessimistic for stability.

Motivation: The bisimulation encoder learns a compact, task-relevant
state representation (25-dim kinematic -> 10-dim latent).  K-means
clustering in this latent space groups behaviourally similar states.
Each cluster gets its own Zooming SAC with independent adaptive action
discretization, enabling fine-grained control within each behavioural
regime without the curse of dimensionality.
"""

from __future__ import annotations

import math
import random as pyrandom
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

from throwaway.highway.ppo import CustomRewardWrapper
from throwaway.highway.continuous_bisim import BisimSAC, MLPEncoder
from throwaway.highway.zooming_ppo import ActionZooming, SplitInfo


# ---------------------------------------------------------------------------
# K-means (simple implementation to avoid sklearn dependency)
# ---------------------------------------------------------------------------

def kmeans(data: np.ndarray, k: int, max_iters: int = 100,
           seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (centers [k, d], labels [n])."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    indices = rng.choice(n, k, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iters):
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.empty_like(centers)
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                new_centers[i] = members.mean(axis=0)
            else:
                new_centers[i] = centers[i]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    labels = dists.argmin(axis=1)
    return centers, labels


def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score (no sklearn dependency).

    For each sample, silhouette = (b - a) / max(a, b) where
      a = mean intra-cluster distance,
      b = mean distance to nearest other cluster.

    To keep this fast on large buffers, we subsample if n > 5000.
    """
    n = data.shape[0]
    if n > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 5000, replace=False)
        data = data[idx]
        labels = labels[idx]
        n = 5000

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    scores = np.zeros(n)
    for i in range(n):
        same = data[labels == labels[i]]
        if len(same) > 1:
            a = np.mean(np.linalg.norm(same - data[i], axis=1))
        else:
            a = 0.0

        b = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = data[labels == lbl]
            b = min(b, np.mean(np.linalg.norm(other - data[i], axis=1)))

        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(np.mean(scores))


def analyze_feature_dim(all_z: np.ndarray, feature_dim: int):
    """Analyze variance spectrum of encoder outputs to diagnose feature_dim.

    Prints per-dimension explained variance ratio and warns if the
    current feature_dim looks too large (wasted dims) or too small
    (all dims are saturated).
    """
    # Center the data
    z_centered = all_z - all_z.mean(axis=0)
    # Covariance eigenvalues (descending)
    cov = np.cov(z_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    total_var = eigenvalues.sum()
    ratios = eigenvalues / total_var if total_var > 0 else eigenvalues

    print(f"\n  Variance spectrum (feature_dim={feature_dim}):")
    cumulative = 0.0
    for i, (ev, r) in enumerate(zip(eigenvalues, ratios)):
        cumulative += r
        bar = "#" * int(r * 50)
        print(f"    dim {i:>2d}: var={ev:.4f}  ratio={r:.3f}  cum={cumulative:.3f}  {bar}")

    # Find how many dims capture 95% of variance
    cum_ratios = np.cumsum(ratios)
    dims_95 = int(np.searchsorted(cum_ratios, 0.95)) + 1
    dims_99 = int(np.searchsorted(cum_ratios, 0.99)) + 1

    print(f"\n  Dims for 95% variance: {dims_95}/{feature_dim}")
    print(f"  Dims for 99% variance: {dims_99}/{feature_dim}")

    if dims_95 < feature_dim * 0.5:
        print(f"  WARNING: Only {dims_95} dims needed for 95% variance. "
              f"feature_dim={feature_dim} may be too large -- "
              f"consider reducing to ~{max(dims_99, dims_95 + 1)}.")
    elif dims_95 == feature_dim:
        print(f"  WARNING: All {feature_dim} dims are needed for 95% variance. "
              f"feature_dim may be too small -- consider increasing it.")
    else:
        print(f"  feature_dim={feature_dim} looks reasonable.")


def select_n_clusters(all_z: np.ndarray, k_range: range,
                      seed: int = 0) -> int:
    """Try each k in k_range, return the one with highest silhouette score."""
    print(f"\n  Evaluating k in {list(k_range)}:")
    best_k = k_range[0]
    best_score = -1.0

    for k in k_range:
        centers, labels = kmeans(all_z, k, seed=seed)
        score = silhouette_score(all_z, labels)
        marker = ""
        if score > best_score:
            best_score = score
            best_k = k
            marker = "  <-- best so far"
        print(f"    k={k}: silhouette={score:.4f}{marker}")

    print(f"\n  Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k


# ---------------------------------------------------------------------------
# Network with rebuildable output head (used for Q-nets and actor)
# ---------------------------------------------------------------------------

class RebuildableHead(nn.Module):
    """MLP with a rebuildable output layer for zooming cube splits."""

    def __init__(self, input_dim: int, n_outputs: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def rebuild_head(self, new_n: int, splits: List[SplitInfo], old_n: int):
        """Rebuild output layer after cube splits.

        Surviving cubes keep their weights.  Children inherit parent
        weights + small noise.
        """
        old_w = self.head.weight.data
        old_b = self.head.bias.data
        new_head = nn.Linear(self.hidden_dim, new_n)

        removed = set(s.old_idx for s in splits)
        surviving = [i for i in range(old_n) if i not in removed]

        with torch.no_grad():
            for new_idx, old_idx in enumerate(surviving):
                new_head.weight.data[new_idx] = old_w[old_idx]
                new_head.bias.data[new_idx] = old_b[old_idx]
            for split in splits:
                pw = old_w[split.old_idx]
                pb = old_b[split.old_idx]
                for new_idx in split.new_indices:
                    new_head.weight.data[new_idx] = pw + torch.randn_like(pw) * 0.01
                    new_head.bias.data[new_idx] = pb + torch.randn_like(pb) * 0.01

        self.head = new_head


# ---------------------------------------------------------------------------
# Per-cluster replay buffer (stores continuous actions for split-safety)
# ---------------------------------------------------------------------------

class ClusterReplayBuffer:
    """Off-policy replay buffer for a single cluster.

    Stores the continuous action value (not cube index) so that
    transitions remain valid after cube splits.
    """

    def __init__(self, feature_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self.z = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_z = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)
        self.next_cids = np.zeros(capacity, dtype=np.int64)

        self.idx = 0
        self.full = False

    def add(self, z, action: float, reward: float, next_z, done: bool,
            next_cid: int):
        np.copyto(self.z[self.idx], z)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        np.copyto(self.next_z[self.idx], next_z)
        self.not_dones[self.idx] = 0.0 if done else 1.0
        self.next_cids[self.idx] = next_cid
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        limit = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, limit, size=batch_size)
        return (
            torch.as_tensor(self.z[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.next_z[idxs], device=self.device),
            torch.as_tensor(self.not_dones[idxs], device=self.device),
            torch.as_tensor(self.next_cids[idxs], device=self.device),
        )

    def __len__(self):
        return self.capacity if self.full else self.idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


def find_cube_index(zooming: ActionZooming, env_action: float) -> int:
    """Map continuous action in [-1, 1] back to the current cube index."""
    a01 = np.clip((env_action + 1.0) / 2.0, 0.0, 1.0)
    for i, cube in enumerate(zooming.active_cubes):
        low = cube.lower[0]
        high = low + cube.s
        if low - 1e-9 <= a01 <= high + 1e-9:
            return i
    # Fallback: nearest center
    centers = np.array([c.lower[0] + 0.5 * c.s for c in zooming.active_cubes])
    return int(np.argmin(np.abs(centers - a01)))


# ---------------------------------------------------------------------------
# ClusteredZoomingSAC
# ---------------------------------------------------------------------------

class ClusteredZoomingSAC:
    r"""K independent discrete-action SAC policies with zooming, one per
    latent-space cluster.

    Each cluster has its own:
      - ActionZooming (adaptive action discretization, persistent visit counts)
      - Twin Q-networks + targets (output Q-values for all cubes)
      - Categorical actor (with UCB bonus in its objective)
      - Replay buffer (stores continuous actions for split-safety)
      - Auto-tuned entropy temperature \alpha

    Design: Critic updates are pessimistic (min of twin Q for stability).
    Actor updates are optimistic (UCB bonus added to Q-values to drive
    exploration of under-visited cubes).  This separates the exploration
    and exploitation signals cleanly.
    """

    def __init__(
        self,
        encoder: MLPEncoder,
        cluster_centers: np.ndarray,
        env: gym.Env,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        batch_size: int = 128,
        buffer_capacity_per_cluster: int = 50_000,
        learning_starts_per_cluster: int = 256,
        actor_update_freq: int = 2,
        critic_target_update_freq: int = 2,
        ucb_coef: float = 1.0,
        init_temperature: float = 0.1,
        seed: int = 0,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts_per_cluster
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.ucb_coef = ucb_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        torch.manual_seed(seed)
        np.random.seed(seed)
        pyrandom.seed(seed)

        self.device = torch.device("cpu")

        # Frozen encoder
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.cluster_centers = torch.as_tensor(
            cluster_centers, dtype=torch.float32
        )
        self.k = len(cluster_centers)
        self.feature_dim = cluster_centers.shape[1]

        # Per-cluster components
        self.zoomings: List[ActionZooming] = []
        self.actors: List[RebuildableHead] = []
        self.q1s: List[RebuildableHead] = []
        self.q2s: List[RebuildableHead] = []
        self.q1_targets: List[RebuildableHead] = []
        self.q2_targets: List[RebuildableHead] = []
        self.log_alphas: List[torch.Tensor] = []
        self.buffers: List[ClusterReplayBuffer] = []

        self.actor_optimizers: List[torch.optim.Adam] = []
        self.critic_optimizers: List[torch.optim.Adam] = []
        self.alpha_optimizers: List[torch.optim.Adam] = []

        for _ in range(self.k):
            zooming = ActionZooming(da=1)
            n_act = zooming.n_actions

            actor = RebuildableHead(self.feature_dim, n_act, hidden_dim)
            q1 = RebuildableHead(self.feature_dim, n_act, hidden_dim)
            q2 = RebuildableHead(self.feature_dim, n_act, hidden_dim)
            q1_tgt = RebuildableHead(self.feature_dim, n_act, hidden_dim)
            q2_tgt = RebuildableHead(self.feature_dim, n_act, hidden_dim)
            q1_tgt.load_state_dict(q1.state_dict())
            q2_tgt.load_state_dict(q2.state_dict())

            log_alpha = torch.tensor(
                np.log(init_temperature), dtype=torch.float32,
                requires_grad=True
            )

            buf = ClusterReplayBuffer(
                self.feature_dim, buffer_capacity_per_cluster, self.device
            )

            self.zoomings.append(zooming)
            self.actors.append(actor)
            self.q1s.append(q1)
            self.q2s.append(q2)
            self.q1_targets.append(q1_tgt)
            self.q2_targets.append(q2_tgt)
            self.log_alphas.append(log_alpha)
            self.buffers.append(buf)

            self.actor_optimizers.append(
                torch.optim.Adam(actor.parameters(), lr=actor_lr)
            )
            self.critic_optimizers.append(
                torch.optim.Adam(
                    list(q1.parameters()) + list(q2.parameters()), lr=critic_lr
                )
            )
            self.alpha_optimizers.append(
                torch.optim.Adam([log_alpha], lr=alpha_lr)
            )

        self._total_splits = [0] * self.k

    # ------------------------------------------------------------------
    # Encode / assign cluster
    # ------------------------------------------------------------------

    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        """obs -> z tensor of shape (1, feature_dim)."""
        obs_flat = obs.flatten().astype(np.float32)
        obs_t = torch.from_numpy(obs_flat).unsqueeze(0)
        with torch.no_grad():
            return self.encoder(obs_t)

    def _assign_cluster(self, z: torch.Tensor) -> int:
        """z tensor of shape (feature_dim,) -> cluster id."""
        dists = torch.norm(self.cluster_centers - z, dim=1)
        return dists.argmin().item()

    # ------------------------------------------------------------------
    # UCB bonus
    # ------------------------------------------------------------------

    def _ucb_bonus(self, c: int) -> torch.Tensor:
        """Compute UCB exploration bonus for each cube in cluster c.

        Returns tensor of shape (n_actions_c,).
        """
        n_actions = self.zoomings[c].n_actions
        total_visits = sum(s.n_play for s in self.zoomings[c].stats) + 1
        log_N = math.log(total_visits)

        bonus = torch.zeros(n_actions)
        for i, stat in enumerate(self.zoomings[c].stats):
            n = max(stat.n_play, 1)
            bonus[i] = self.ucb_coef * math.sqrt(log_N / n)
        return bonus

    # ------------------------------------------------------------------
    # Map continuous action -> current cube index (batch)
    # ------------------------------------------------------------------

    def _actions_to_indices(self, c: int,
                            actions: torch.Tensor) -> torch.Tensor:
        """Convert continuous actions to current cube indices for cluster c."""
        return torch.tensor(
            [find_cube_index(self.zoomings[c], a.item()) for a in actions],
            dtype=torch.long,
        )

    # ------------------------------------------------------------------
    # Compute V(z') using the appropriate cluster's target networks
    # ------------------------------------------------------------------

    def _compute_next_values(self, next_z: torch.Tensor,
                             next_cids: torch.Tensor) -> torch.Tensor:
        """Compute target V(z') for a batch, routing each next state to
        its cluster's target networks and actor."""
        v_next = torch.zeros(len(next_z), 1)

        for c_prime in range(self.k):
            mask = next_cids == c_prime
            if not mask.any():
                continue

            z_prime = next_z[mask]
            alpha = self.log_alphas[c_prime].exp().detach()

            with torch.no_grad():
                logits = self.actors[c_prime](z_prime)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-8)

                q1_next = self.q1_targets[c_prime](z_prime)
                q2_next = self.q2_targets[c_prime](z_prime)
                q_next = torch.min(q1_next, q2_next)

                v = (probs * (q_next - alpha * log_probs)).sum(
                    dim=-1, keepdim=True
                )
            v_next[mask] = v

        return v_next

    # ------------------------------------------------------------------
    # Critic update for cluster c (pessimistic, no bonus)
    # ------------------------------------------------------------------

    def _update_critic(self, c: int):
        z, actions_cont, rewards, next_z, not_dones, next_cids = (
            self.buffers[c].sample(self.batch_size)
        )

        action_indices = self._actions_to_indices(c, actions_cont)

        # Target value from appropriate next-state clusters
        with torch.no_grad():
            v_next = self._compute_next_values(next_z, next_cids)
            target_q = rewards + not_dones * self.gamma * v_next

        q1_all = self.q1s[c](z)
        q2_all = self.q2s[c](z)
        q1_pred = q1_all.gather(1, action_indices.unsqueeze(1))
        q2_pred = q2_all.gather(1, action_indices.unsqueeze(1))

        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(
            q2_pred, target_q
        )

        self.critic_optimizers[c].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[c].step()

    # ------------------------------------------------------------------
    # Actor + alpha update for cluster c (optimistic: Q + UCB bonus)
    # ------------------------------------------------------------------

    def _update_actor_and_alpha(self, c: int):
        z, *_ = self.buffers[c].sample(self.batch_size)
        alpha = self.log_alphas[c].exp()

        logits = self.actors[c](z)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)

        with torch.no_grad():
            q1_all = self.q1s[c](z)
            q2_all = self.q2s[c](z)
            q_min = torch.min(q1_all, q2_all)

        # Optimistic Q for actor: add UCB bonus
        bonus = self._ucb_bonus(c).unsqueeze(0)  # (1, n_actions)
        q_optimistic = q_min + bonus

        # Actor loss: E_a~\pi [\alpha log \pi(a|z) - Q_optimistic(z, a)]
        actor_loss = (
            probs * (alpha.detach() * log_probs - q_optimistic)
        ).sum(dim=-1).mean()

        self.actor_optimizers[c].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[c].step()

        # Alpha loss
        n_actions = self.zoomings[c].n_actions
        target_entropy = -math.log(1.0 / n_actions) * 0.98
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = (
            self.log_alphas[c] * (entropy - target_entropy)
        ).mean()

        self.alpha_optimizers[c].zero_grad()
        alpha_loss.backward()
        self.alpha_optimizers[c].step()

    # ------------------------------------------------------------------
    # Check and split (per cluster)
    # ------------------------------------------------------------------

    def _check_and_split(self):
        for c in range(self.k):
            old_n = self.zoomings[c].n_actions
            splits = self.zoomings[c].try_split()

            if not splits:
                continue

            new_n = self.zoomings[c].n_actions
            self._total_splits[c] += len(splits)

            # Rebuild all networks for this cluster
            for net in [self.actors[c], self.q1s[c], self.q2s[c],
                        self.q1_targets[c], self.q2_targets[c]]:
                net.rebuild_head(new_n, splits, old_n)

            # Rebuild optimizers with new parameters
            self.actor_optimizers[c] = torch.optim.Adam(
                self.actors[c].parameters(), lr=self.actor_lr
            )
            self.critic_optimizers[c] = torch.optim.Adam(
                list(self.q1s[c].parameters()) +
                list(self.q2s[c].parameters()),
                lr=self.critic_lr,
            )
            # log_alpha doesn't change, optimizer stays

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 10_000):
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        current_ep_reward = 0.0

        for step in range(1, total_timesteps + 1):
            z_t = self._encode(obs)
            z_vec = z_t.squeeze(0)
            c = self._assign_cluster(z_vec)

            # Action selection
            if len(self.buffers[c]) < self.learning_starts:
                action_idx = pyrandom.randint(0, self.zoomings[c].n_actions - 1)
            else:
                with torch.no_grad():
                    logits = self.actors[c](z_t)
                    dist = Categorical(logits=logits)
                    action_idx = dist.sample().item()

            env_action = self.zoomings[c].get_env_action(action_idx)

            # Update per-cube visit count
            self.zoomings[c].stats[action_idx].n_play += 1

            # Step environment
            next_obs, reward, done, truncated, _ = self.env.step(env_action)
            terminal = done or truncated

            next_z_t = self._encode(next_obs)
            next_c = self._assign_cluster(next_z_t.squeeze(0))

            # Store in cluster c's replay buffer
            self.buffers[c].add(
                z_vec.numpy(), env_action[0] if hasattr(env_action, '__len__') else float(env_action),
                reward, next_z_t.squeeze(0).numpy(), terminal, next_c,
            )

            current_ep_reward += reward
            if terminal:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                next_obs, _ = self.env.reset()
            obs = next_obs

            # Update clusters that have enough data
            for c_upd in range(self.k):
                if len(self.buffers[c_upd]) < self.learning_starts:
                    continue
                self._update_critic(c_upd)
                if step % self.actor_update_freq == 0:
                    self._update_actor_and_alpha(c_upd)
                if step % self.critic_target_update_freq == 0:
                    soft_update_params(
                        self.q1s[c_upd], self.q1_targets[c_upd], self.tau
                    )
                    soft_update_params(
                        self.q2s[c_upd], self.q2_targets[c_upd], self.tau
                    )

            # Check cube splits
            self._check_and_split()

            # Logging
            if episode_rewards and step % print_every == 0:
                recent = episode_rewards[-50:]
                buf_sizes = " ".join(
                    f"c{i}:{len(b)}" for i, b in enumerate(self.buffers)
                )
                n_actions_str = " ".join(
                    f"c{i}:{z.n_actions}" for i, z in enumerate(self.zoomings)
                )
                print(
                    f"[{step:>7d}/{total_timesteps}] "
                    f"ep={len(episode_rewards):>4d}  "
                    f"reward(last50)={np.mean(recent):>7.2f}  "
                    f"bufs=[{buf_sizes}]  "
                    f"actions=[{n_actions_str}]  "
                    f"splits={self._total_splits}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        z_t = self._encode(obs)
        c = self._assign_cluster(z_t.squeeze(0))
        with torch.no_grad():
            logits = self.actors[c](z_t)
            if deterministic:
                action_idx = logits.argmax(dim=-1).item()
            else:
                action_idx = Categorical(logits=logits).sample().item()
        return self.zoomings[c].get_env_action(action_idx)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: List[float] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cluster_data = []
        for c in range(self.k):
            zooming_state = [
                {"lower": cube.lower.tolist(), "s": cube.s, "d": cube.d,
                 "n_play": stat.n_play}
                for cube, stat in zip(
                    self.zoomings[c].active_cubes, self.zoomings[c].stats
                )
            ]
            cluster_data.append({
                "actor": self.actors[c].state_dict(),
                "q1": self.q1s[c].state_dict(),
                "q2": self.q2s[c].state_dict(),
                "log_alpha": self.log_alphas[c].detach().item(),
                "zooming_state": zooming_state,
            })

        torch.save({
            "encoder": self.encoder.state_dict(),
            "cluster_centers": self.cluster_centers.numpy(),
            "cluster_data": cluster_data,
            "episode_rewards": episode_rewards or [],
        }, path)
        print(f"Saved ClusteredZoomingSAC checkpoint to {path}")

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "ClusteredZoomingSAC":
        data = torch.load(path, weights_only=False)
        cluster_centers = data["cluster_centers"]
        feature_dim = cluster_centers.shape[1]

        # Reconstruct encoder
        obs_dim = int(np.prod(env.observation_space.shape))
        encoder = MLPEncoder(obs_dim, feature_dim)
        encoder.load_state_dict(data["encoder"])

        agent = cls(encoder=encoder, cluster_centers=cluster_centers, env=env)

        # Restore per-cluster zooming state and networks
        for c, cd in enumerate(data["cluster_data"]):
            # Rebuild zooming cubes
            agent.zoomings[c].active_cubes = []
            agent.zoomings[c].stats = []
            for cs in cd["zooming_state"]:
                from throwaway.highway.zooming import Cube, CubeStats
                cube = Cube(lower=np.array(cs["lower"]), s=cs["s"], d=cs["d"])
                agent.zoomings[c].active_cubes.append(cube)
                agent.zoomings[c].stats.append(CubeStats(Q=0.0, n_play=cs["n_play"]))

            n_act = agent.zoomings[c].n_actions

            # Rebuild networks with correct action count
            agent.actors[c] = RebuildableHead(feature_dim, n_act)
            agent.q1s[c] = RebuildableHead(feature_dim, n_act)
            agent.q2s[c] = RebuildableHead(feature_dim, n_act)
            agent.q1_targets[c] = RebuildableHead(feature_dim, n_act)
            agent.q2_targets[c] = RebuildableHead(feature_dim, n_act)

            agent.actors[c].load_state_dict(cd["actor"])
            agent.q1s[c].load_state_dict(cd["q1"])
            agent.q2s[c].load_state_dict(cd["q2"])
            agent.q1_targets[c].load_state_dict(agent.q1s[c].state_dict())
            agent.q2_targets[c].load_state_dict(agent.q2s[c].state_dict())

        return agent


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env_continuous():
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
    BISIM_TIMESTEPS = 100_000
    ZOOMING_TIMESTEPS = 100_000
    FEATURE_DIM = 10
    K_RANGE = range(2, 9)  # candidates for N_CLUSTERS; best chosen by silhouette
    SEED = 42

    # ===== Phase 1: Train BisimSAC to learn encoder varphi =====
    print("=" * 60)
    print("Phase 1: Training BisimSAC encoder")
    print("=" * 60)

    env = make_highway_env_continuous()

    bisim_agent = BisimSAC(
        env,
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        gamma=0.99,
        batch_size=128,
        buffer_capacity=100_000,
        learning_starts=1000,
        bisim_coef=0.5,
        seed=SEED,
    )

    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    bisim_rewards = bisim_agent.learn(
        total_timesteps=BISIM_TIMESTEPS, print_every=10_000
    )
    bisim_agent.save("checkpoints/bisim_phase1.pt", bisim_rewards)

    # ===== Phase 1.5: Analyze encoder representation =====
    print()
    print("=" * 60)
    print("Phase 1.5: Analyzing encoder & selecting N_CLUSTERS")
    print("=" * 60)

    encoder = bisim_agent.encoder
    encoder.eval()

    n_samples = len(bisim_agent.buffer)
    raw_obs = bisim_agent.buffer.obs[:n_samples]
    obs_tensor = torch.as_tensor(raw_obs, dtype=torch.float32)

    with torch.no_grad():
        zs = []
        for i in range(0, n_samples, 1024):
            batch = obs_tensor[i:i + 1024]
            zs.append(encoder(batch).numpy())
        all_z = np.concatenate(zs, axis=0)

    print(f"Encoded {n_samples} observations -> latent shape {all_z.shape}")

    # Diagnose feature_dim
    analyze_feature_dim(all_z, FEATURE_DIM)

    # Select best N_CLUSTERS via silhouette score
    N_CLUSTERS = select_n_clusters(all_z, K_RANGE, seed=SEED)

    # ===== Phase 2: Cluster latent space =====
    print()
    print("=" * 60)
    print(f"Phase 2: Clustering latent space with k-means (k={N_CLUSTERS})")
    print("=" * 60)

    centers, labels = kmeans(all_z, N_CLUSTERS, seed=SEED)

    for c in range(N_CLUSTERS):
        count = (labels == c).sum()
        print(f"  Cluster {c}: {count} samples ({100 * count / n_samples:.1f}%)")

    # ===== Phase 3: Train clustered Zooming SAC =====
    print()
    print("=" * 60)
    print("Phase 3: Training clustered Zooming SAC (with UCB optimism)")
    print("=" * 60)

    env_phase3 = make_highway_env_continuous()

    agent = ClusteredZoomingSAC(
        encoder=encoder,
        cluster_centers=centers,
        env=env_phase3,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        batch_size=128,
        buffer_capacity_per_cluster=50_000,
        learning_starts_per_cluster=256,
        actor_update_freq=2,
        critic_target_update_freq=2,
        ucb_coef=1.0,
        init_temperature=0.1,
        seed=SEED,
    )

    zooming_rewards = agent.learn(
        total_timesteps=ZOOMING_TIMESTEPS, print_every=10_000
    )
    agent.save("checkpoints/bisim_zooming.pt", zooming_rewards)

    # Evaluate
    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(20):
        obs, _ = env_phase3.reset()
        total_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env_phase3.step(action)
            total_reward += reward
        eval_rewards.append(total_reward)

    print(f"Eval over 20 episodes: mean={np.mean(eval_rewards):.2f}, std={np.std(eval_rewards):.2f}")
    env.close()
    env_phase3.close()
