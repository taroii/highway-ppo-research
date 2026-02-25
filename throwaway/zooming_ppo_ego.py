"""
Zooming PPO with ego-state discretization for HighwayEnv.

Approach:
- Discretize the ego vehicle's [x, y, vx, vy] into n_bins per feature,
  giving n_bins^4 state cells (e.g. 3^4 = 81).
- Each state cell maintains its own independent 1D zooming partition
  over the action space.
- Policy: softmax over interval preferences within the active partition.
- Preferences updated via policy gradient with TD(0) advantages.
- Intervals split when advantage variance is high.

This is a middle ground between:
- zooming_ppo_action.py (action-only zooming, state-blind)
- zooming_ppo_full.py (full joint state-action zooming, intractable at 26D)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import List, Tuple, Dict

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1D action interval
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """A 1D interval [lower, lower + s] in normalized action space [0, 1]."""
    lower: float
    s: float

    @property
    def center(self) -> float:
        return self.lower + self.s / 2.0

    def contains(self, a: float, eps: float = 1e-12) -> bool:
        return self.lower - eps <= a <= self.lower + self.s + eps

    def split_children(self) -> List[Interval]:
        half = self.s / 2.0
        return [Interval(self.lower, half), Interval(self.lower + half, half)]

    def key(self) -> Tuple[float, float]:
        return (float(self.lower), float(self.s))


@dataclass
class IntervalStats:
    preference: float = 0.0
    n_visits: int = 0
    advantage_sum: float = 0.0
    advantage_sq_sum: float = 0.0

    def advantage_variance(self) -> float:
        if self.n_visits < 2:
            return float("inf")
        mean = self.advantage_sum / self.n_visits
        return (self.advantage_sq_sum / self.n_visits) - mean ** 2


# ---------------------------------------------------------------------------
# Per-cell action partition
# ---------------------------------------------------------------------------

class ActionPartition:
    """1D zooming partition over [0, 1].

    Maintains a set of non-overlapping intervals that tile [0, 1].
    Splitting replaces an interval with its two halves.
    """

    def __init__(self):
        root = Interval(0.0, 1.0)
        self.intervals: Dict[Tuple[float, float], Tuple[Interval, IntervalStats]] = {
            root.key(): (root, IntervalStats())
        }

    def all_intervals(self) -> List[Tuple[Interval, IntervalStats]]:
        return list(self.intervals.values())

    def split(self, interval: Interval, stats: IntervalStats):
        """Replace interval with two children, inheriting parent preference."""
        del self.intervals[interval.key()]
        for child in interval.split_children():
            self.intervals[child.key()] = (
                child,
                IntervalStats(preference=stats.preference),
            )

    def n_intervals(self) -> int:
        return len(self.intervals)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ZoomingPPOEgo:
    """
    Zooming adaptive discretization with ego-state binning.

    State cell = discretized (x, y, vx, vy) of the ego vehicle.
    Each cell owns an independent 1D zooming partition over the action.
    """

    def __init__(
        self,
        ego_feature_indices: List[int],
        n_bins: int,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        gamma: float = 0.99,
        lr: float = 0.1,
        temperature: float = 1.0,
        split_threshold: float = 0.5,
        min_visits_to_split: int = 10,
        seed: int = 0,
    ):
        self.ego_feature_indices = ego_feature_indices
        self.n_ego_features = len(ego_feature_indices)
        self.n_bins = n_bins
        self.n_cells = n_bins ** self.n_ego_features
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature
        self.split_threshold = split_threshold
        self.min_visits_to_split = min_visits_to_split
        self.rng = np.random.default_rng(seed)

        self.act_low, self.act_high = action_bounds

        # One ActionPartition per state cell (lazily created)
        self.partitions: Dict[Tuple[int, ...], ActionPartition] = {}

        # Value function keyed by state cell
        self.value_estimates: Dict[Tuple[int, ...], Tuple[float, int]] = {}

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    def _get_partition(self, cell: Tuple[int, ...]) -> ActionPartition:
        if cell not in self.partitions:
            self.partitions[cell] = ActionPartition()
        return self.partitions[cell]

    def discretize_ego(self, obs: np.ndarray) -> Tuple[int, ...]:
        """Map ego vehicle features to a state cell.

        obs shape: (vehicles_count, n_features).
        Ego is row 0.  With normalize=True in highway-env the features
        are roughly in [-1, 1]; we map to [0, 1] then bin.
        """
        ego_feats = obs[0, self.ego_feature_indices]
        ego_norm = np.clip((ego_feats + 1.0) / 2.0, 0.0, 1.0)
        bins = np.clip((ego_norm * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(bins.tolist())

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def denormalize_action(self, a_norm: float) -> np.ndarray:
        return self.act_low + a_norm * (self.act_high - self.act_low)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def policy_probs(
        self, intervals: List[Tuple[Interval, IntervalStats]]
    ) -> np.ndarray:
        prefs = np.array([st.preference for _, st in intervals])
        prefs = prefs - np.max(prefs)
        exp_prefs = np.exp(prefs / self.temperature)
        return exp_prefs / (exp_prefs.sum() + 1e-12)

    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[Tuple[int, ...], Interval, IntervalStats, float, np.ndarray]:
        """Pick an action given the raw (unflatttened) observation.

        Returns (state_cell, interval, stats, log_prob, action).
        """
        cell = self.discretize_ego(obs)
        partition = self._get_partition(cell)
        intervals = partition.all_intervals()

        probs = self.policy_probs(intervals)
        idx = self.rng.choice(len(intervals), p=probs)
        interval, stats = intervals[idx]
        log_prob = np.log(probs[idx] + 1e-12)

        action = self.denormalize_action(interval.center)
        return cell, interval, stats, log_prob, action

    # ------------------------------------------------------------------
    # Value function
    # ------------------------------------------------------------------

    def get_value(self, cell: Tuple[int, ...]) -> float:
        if cell in self.value_estimates:
            total, count = self.value_estimates[cell]
            return total / count
        return 0.0

    def update_value(self, cell: Tuple[int, ...], ret: float):
        if cell in self.value_estimates:
            total, count = self.value_estimates[cell]
            self.value_estimates[cell] = (total + ret, count + 1)
        else:
            self.value_estimates[cell] = (ret, 1)

    # ------------------------------------------------------------------
    # Advantages
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        rewards: List[float],
        cells: List[Tuple[int, ...]],
        dones: List[bool],
    ) -> List[float]:
        advantages = []
        for i, (r, cell, done) in enumerate(zip(rewards, cells, dones)):
            v = self.get_value(cell)
            if done or i == len(rewards) - 1:
                v_next = 0.0
            else:
                v_next = self.get_value(cells[i + 1])
            td_target = r + self.gamma * v_next * (1 - done)
            advantages.append(td_target - v)
        return advantages

    # ------------------------------------------------------------------
    # Trajectory update
    # ------------------------------------------------------------------

    def update_from_trajectory(
        self,
        cells: List[Tuple[int, ...]],
        intervals: List[Interval],
        log_probs: List[float],
        rewards: List[float],
        dones: List[bool],
    ):
        # Compute returns for value function
        returns: List[float] = []
        G = 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)

        for cell, ret in zip(cells, returns):
            self.update_value(cell, ret)

        # Advantages
        advantages = self.compute_advantages(rewards, cells, dones)
        adv_arr = np.array(advantages)
        if len(adv_arr) > 1 and adv_arr.std() > 1e-8:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Update interval preferences
        for cell, interval, adv in zip(cells, intervals, adv_arr):
            partition = self._get_partition(cell)
            key = interval.key()
            if key in partition.intervals:
                _, stats = partition.intervals[key]
                stats.preference += self.lr * float(adv)
                stats.n_visits += 1
                stats.advantage_sum += float(adv)
                stats.advantage_sq_sum += float(adv) ** 2

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def maybe_split(self):
        for partition in self.partitions.values():
            to_split = []
            for key, (interval, stats) in partition.intervals.items():
                if stats.n_visits < self.min_visits_to_split:
                    continue
                if stats.advantage_variance() > self.split_threshold:
                    to_split.append((interval, stats))
            for interval, stats in to_split:
                partition.split(interval, stats)

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def run_episode(self, env: gym.Env) -> Tuple[float, int]:
        obs, _ = env.reset()

        cells: List[Tuple[int, ...]] = []
        intervals: List[Interval] = []
        log_probs: List[float] = []
        rewards: List[float] = []
        dones: List[bool] = []

        total_reward = 0.0
        done = truncated = False

        while not (done or truncated):
            cell, interval, stats, log_prob, action = self.select_action(obs)

            next_obs, reward, done, truncated, _ = env.step(action)

            cells.append(cell)
            intervals.append(interval)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            dones.append(done or truncated)

            total_reward += reward
            obs = next_obs

        if cells:
            self.update_from_trajectory(cells, intervals, log_probs, rewards, dones)
            self.maybe_split()

        return total_reward, len(cells)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self, env: gym.Env, n_episodes: int = 1000, print_every: int = 100
    ):
        episode_rewards: List[float] = []

        for ep in range(n_episodes):
            reward, steps = self.run_episode(env)
            episode_rewards.append(reward)

            if (ep + 1) % print_every == 0:
                recent = episode_rewards[-print_every:]
                mean_r = np.mean(recent)
                total_intervals = sum(
                    p.n_intervals() for p in self.partitions.values()
                )
                visited_cells = sum(
                    1
                    for p in self.partitions.values()
                    if any(st.n_visits > 0 for _, st in p.intervals.values())
                )
                print(
                    f"Episode {ep+1}: mean_reward={mean_r:.2f}, "
                    f"total_intervals={total_intervals}, "
                    f"visited_cells={visited_cells}/{self.n_cells}"
                )

        return episode_rewards

    def summary(self) -> Dict:
        total_intervals = sum(
            p.n_intervals() for p in self.partitions.values()
        )
        visited_cells = sum(
            1
            for p in self.partitions.values()
            if any(st.n_visits > 0 for _, st in p.intervals.values())
        )
        return {
            "total_intervals": total_intervals,
            "visited_cells": visited_cells,
            "n_cells": self.n_cells,
        }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env():
    """Highway env with 5 vehicles, Kinematics observation."""
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": False,
                "lateral": True,
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
            "duration": 40,
            "policy_frequency": 2,
        },
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_highway_env()

    act_low = np.array([-np.pi / 4], dtype=np.float32)
    act_high = np.array([np.pi / 4], dtype=np.float32)

    N_BINS = 3

    agent = ZoomingPPOEgo(
        ego_feature_indices=[1, 2, 3, 4],  # x, y, vx, vy (skip presence)
        n_bins=N_BINS,
        action_bounds=(act_low, act_high),
        gamma=0.99,
        lr=0.1,
        temperature=1.0,
        split_threshold=0.5,
        min_visits_to_split=10,
        seed=42,
    )

    print("Starting training (ego-state discretization + action zooming)...")
    print(f"Ego features: [x, y, vx, vy]")
    print(f"Bins per feature: {N_BINS}, state cells: {agent.n_cells}")
    print(f"Each cell has an independent 1D action zooming partition")
    print()

    rewards = agent.train(env, n_episodes=500, print_every=50)

    print()
    stats = agent.summary()
    print(f"Total action intervals: {stats['total_intervals']}")
    print(f"Visited state cells: {stats['visited_cells']}/{stats['n_cells']}")
    print(f"Mean reward (last 50): {np.mean(rewards[-50:]):.2f}")

    env.close()
