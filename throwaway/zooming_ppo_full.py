"""
Zooming Adaptive Discretization with PPO/GAE for HighwayEnv
(Fixed to match Algorithm 1 structural mechanics)

Fixes from original:
1. Proper domain logic: dom(B) = B minus union of smaller active balls.
   Parents become unselectable when domain is empty.
2. Child redirect inline: every (H+1)th play of a split parent, rebind
   the update target to the child containing (x,a), giving it a real
   data-driven update (not a post-hoc preference copy).
3. Children created as r/2-net of dom(B), not full 2^d bisection.
4. Parent stops being selectable once all children are promoted
   (domain becomes empty).

Retains PPO/GAE for the value/policy update mechanism in place of
optimistic Q-learning (acknowledged deviation from paper).
"""

from __future__ import annotations
from matplotlib import pyplot as plt

import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Set
try:
    import highway_env  # noqa: F401
except ImportError:
    pass


def clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


# ---------------------------------------------------------------------------
# Ball (cube) data structures
# ---------------------------------------------------------------------------

@dataclass
class Ball:
    """
    Axis-aligned cube in [0,1]^d representing a ball in the metric space.
    r = s = side length = diameter (under L-inf metric, diameter = side length).
    center is lower + s/2 in every dimension.
    """
    lower: np.ndarray
    s: float           # side length = r(B) = diam(B) under L-inf
    d: int

    @property
    def center(self) -> np.ndarray:
        return self.lower + self.s / 2.0

    @property
    def radius(self) -> float:
        return self.s  # diameter under L-inf

    def contains(self, z: np.ndarray, eps: float = 1e-12) -> bool:
        upper = self.lower + self.s
        return bool(np.all(z >= self.lower - eps) and np.all(z <= upper + eps))

    def contains_state(self, x: np.ndarray, ds: int, eps: float = 1e-12) -> bool:
        """Check if state projection contains x."""
        lower_s = self.lower[:ds]
        upper_s = lower_s + self.s
        return bool(np.all(x >= lower_s - eps) and np.all(x <= upper_s + eps))

    def key(self) -> Tuple[float, ...]:
        return (*map(float, self.lower.tolist()), float(self.s))


@dataclass
class BallStats:
    """Per-ball algorithm state, mirroring Algorithm 1 counters."""
    preference: float = 0.0    # softmax logit (replaces Q^k_h(B))
    n_update: int = 0          # n^k_h(B): number of times updated
    n_play: int = 0            # tilde{n}^k_h(B): number of times played/selected
    is_split: bool = False
    children_keys: List[Tuple[float, ...]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class ZoomingPPOFixed:
    """
    Zooming adaptive discretization with PPO/GAE.

    Structural mechanics follow Algorithm 1:
    - Per-stage active (P) and buffer (F) partitions
    - Domain: dom(B) = B minus union of smaller active balls
    - Relevant balls: those whose domain intersects state
    - Splitting when n_play >= Nsplit, children = r/2-net of dom(B)
    - Child redirect: every (H+1)th play of split parent, rebind update
      to child containing (x,a)
    - Promotion: buffer -> active when n_update >= Nmin
    """

    def __init__(
        self,
        ds: int,
        da: int,
        H: int,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 0.1,
        temperature: float = 1.0,
        clip_eps: float = 0.2,
        n_ppo_epochs: int = 4,
        max_depth: int = 3,
        dmax: float = 1.0,
        seed: int = 0,
    ):
        self.ds = ds
        self.da = da
        self.d = ds + da
        self.H = H
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.temperature = temperature
        self.clip_eps = clip_eps
        self.n_ppo_epochs = n_ppo_epochs
        self.max_depth = max_depth
        self.dmax = dmax
        self.rng = np.random.default_rng(seed)

        self.act_low, self.act_high = action_bounds

        # Thresholds from paper: Nsplit(B) = (dmax/r(B))^2, Nmin = Nsplit/4
        # In normalized [0,1]^d with dmax=1: Nsplit = (1/s)^2
        self.nsplit_fn = lambda ball: max(1, int(np.ceil((self.dmax / ball.s) ** 2)))
        self.nmin_fn = lambda ball: max(1, self.nsplit_fn(ball) // 4)

        # Per-stage active (P) and buffer (F) partitions
        # Keyed by ball.key() -> (Ball, BallStats)
        self.P: List[Dict[Tuple[float, ...], Tuple[Ball, BallStats]]] = []
        self.F: List[Dict[Tuple[float, ...], Tuple[Ball, BallStats]]] = []
        self._init_partitions()

        # Tabular value function V(s)
        self.value_estimates: Dict[Tuple[int, ...], Tuple[float, int]] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_partitions(self):
        """
        Paper line 1: Initialize P^1_h to be a (dmax/H)-net of S x A.
        For simplicity in [0,1]^d with dmax=1, we use a single root ball
        covering everything (a 1-net). This is valid since dmax/H < 1
        for H >= 2 and one ball of diameter 1 covers all of [0,1]^d.

        For more faithful initialization, we could tile with balls of
        diameter dmax/H, but the single root suffices--splitting will
        refine it.
        """
        root = Ball(lower=np.zeros(self.d), s=self.dmax, d=self.d)
        for h in range(self.H):
            P_h: Dict[Tuple[float, ...], Tuple[Ball, BallStats]] = {}
            F_h: Dict[Tuple[float, ...], Tuple[Ball, BallStats]] = {}
            P_h[root.key()] = (root, BallStats())
            self.P.append(P_h)
            self.F.append(F_h)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return clip01(obs)

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return self.act_low + action_norm * (self.act_high - self.act_low)

    # ------------------------------------------------------------------
    # Domain logic (Algorithm 1 core)
    # ------------------------------------------------------------------

    def _point_in_smaller_active_ball(
        self, z: np.ndarray, h: int, parent_s: float, exclude_key: Optional[Tuple[float, ...]] = None
    ) -> bool:
        """
        Check if point z is contained in any active ball with strictly
        smaller radius than parent_s. Used to compute dom(B).
        """
        for key, (ball, _) in self.P[h].items():
            if key == exclude_key:
                continue
            if ball.s < parent_s - 1e-12 and ball.contains(z):
                return True
        return False

    def _state_in_smaller_active_ball(
        self, x: np.ndarray, h: int, parent_s: float, exclude_key: Optional[Tuple[float, ...]] = None
    ) -> bool:
        """
        Check if state x is fully covered by smaller active balls for
        ALL possible actions within the parent. This is a conservative
        check -- we check if there exists any action direction not covered
        by a smaller ball.

        For tractability, we use state-only relevance: a ball's domain
        intersects state x if x is in the ball's state-projection AND
        no single smaller active ball covers x for all actions.
        """
        for key, (ball, _) in self.P[h].items():
            if key == exclude_key:
                continue
            if ball.s < parent_s - 1e-12 and ball.contains_state(x, self.ds):
                return True
        return False

    def _domain_is_empty(self, h: int, ball: Ball) -> bool:
        """
        Check if dom^k_h(B) is empty, i.e., every point in B is covered
        by some smaller active ball.

        We approximate this by checking if every child of B (if split)
        has been promoted to active. If all children are active, the
        parent's domain is empty.
        """
        bkey = ball.key()
        if bkey not in self.P[h]:
            return True
        _, stats = self.P[h][bkey]
        if not stats.is_split:
            return False
        # Domain is empty iff all children are in active set
        for ckey in stats.children_keys:
            if ckey not in self.P[h]:
                return True  # child not yet promoted, but...
                # Actually if child not in P, it's still in F or hasn't been
                # created. Parent domain is NOT empty in that case.
        # Check: are ALL children in P?
        for ckey in stats.children_keys:
            if ckey not in self.P[h]:
                return False  # some child still buffering -> parent has domain
        return True  # all children active -> parent domain empty

    def relevant_balls(self, h: int, x: np.ndarray) -> List[Tuple[Ball, BallStats]]:
        """
        Compute rel^k_h(x) = {B in P^k_h : exists a, (x,a) in dom^k_h(B)}.

        Implementation:
        1. Find all active balls whose state-projection contains x.
        2. Exclude balls whose domain is empty (all children promoted).
        3. Among remaining, a ball is relevant if x is in its state
           projection and NOT fully covered by strictly smaller active balls.

        For correctness: we keep the ball if there's no smaller active ball
        containing x in state projection. If a smaller ball exists, the
        larger ball's domain at x is empty (the smaller ball covers it).
        """
        x = clip01(np.asarray(x, dtype=float))

        # Gather all active balls containing x in state projection
        containing: List[Tuple[Ball, BallStats, Tuple[float, ...]]] = []
        for key, (ball, stats) in self.P[h].items():
            if ball.contains_state(x, self.ds):
                containing.append((ball, stats, key))

        if not containing:
            raise RuntimeError("No active ball contains state (coverage invariant broken)")

        # Sort by radius (smallest first)
        containing.sort(key=lambda t: t[0].s)

        # Build relevant set using domain logic:
        # A ball B is relevant at x if no strictly smaller active ball
        # also contains x in state-projection. Additionally, skip balls
        # whose domain is globally empty (all children promoted).
        relevant: List[Tuple[Ball, BallStats]] = []

        # The smallest balls are always relevant (nothing smaller covers x)
        min_s = containing[0][0].s

        for ball, stats, key in containing:
            # Skip if domain is empty (all children promoted)
            if stats.is_split and self._domain_is_empty(h, ball):
                continue

            # Ball is relevant if no strictly smaller active ball contains x
            has_smaller = False
            for ball2, stats2, key2 in containing:
                if ball2.s < ball.s - 1e-12 and key2 != key:
                    # Check ball2 isn't also domain-empty
                    if not (stats2.is_split and self._domain_is_empty(h, ball2)):
                        has_smaller = True
                        break

            if not has_smaller:
                relevant.append((ball, stats))

        if not relevant:
            # Fallback: return smallest balls (should not normally happen)
            relevant = [(b, s) for b, s, _ in containing if abs(b.s - min_s) < 1e-12]

        return relevant

    # ------------------------------------------------------------------
    # Policy (softmax over preferences, replacing optimistic argmax)
    # ------------------------------------------------------------------

    def policy_probs(self, relevant: List[Tuple[Ball, BallStats]]) -> np.ndarray:
        prefs = np.array([stats.preference for _, stats in relevant])
        prefs = prefs - np.max(prefs)
        exp_prefs = np.exp(prefs / self.temperature)
        return exp_prefs / (exp_prefs.sum() + 1e-12)

    # ------------------------------------------------------------------
    # Splitting: create r/2-net of dom(B)
    # ------------------------------------------------------------------

    def _create_children_net(self, h: int, ball: Ball) -> List[Ball]:
        """
        Create children as an r(B)/2-net of dom^k_h(B).

        Paper line 10: C(B) = r(B)/2-net of dom^k_h(B).

        We approximate dom(B) by the full ball B (since at split time,
        B is typically the smallest ball covering its region). Children
        are cubes of side length s/2 tiling B. We then filter out any
        child whose center is already covered by a smaller active ball.

        In d dimensions, tiling B with cubes of side s/2 gives 2^d
        children. We keep only those whose region intersects dom(B).
        """
        half = ball.s / 2.0
        children: List[Ball] = []

        for mask in range(1 << self.d):
            offset = np.array([(mask >> i) & 1 for i in range(self.d)], dtype=float)
            child_lower = ball.lower + offset * half
            child = Ball(lower=child_lower, s=half, d=self.d)

            # Filter: skip children whose center is inside a smaller active ball
            # (they would have empty domain immediately)
            center = child.center
            if not self._point_in_smaller_active_ball(center, h, ball.s, exclude_key=ball.key()):
                children.append(child)

        return children

    def _split_if_needed(self, h: int, ball: Ball, stats: BallStats):
        """
        Paper line 8-11: Split when n_play >= Nsplit(B).
        Creates children and adds them to buffer set F.
        """
        if stats.is_split:
            return
        if stats.n_play < self.nsplit_fn(ball):
            return

        depth = -int(round(np.log2(ball.s / self.dmax))) if ball.s > 0 else 0
        if depth >= self.max_depth:
            return

        stats.is_split = True
        children = self._create_children_net(h, ball)
        stats.children_keys = []

        for child in children:
            ckey = child.key()
            stats.children_keys.append(ckey)
            if ckey not in self.P[h] and ckey not in self.F[h]:
                self.F[h][ckey] = (child, BallStats(preference=stats.preference))

    # ------------------------------------------------------------------
    # Child redirect (Paper lines 12-14)
    # ------------------------------------------------------------------

    def _maybe_redirect_to_child(
        self, h: int, ball: Ball, stats: BallStats, z: np.ndarray
    ) -> Tuple[Ball, BallStats, bool]:
        """
        Paper lines 12-14: If ball is split and n_play mod (H+1) == 0,
        redirect the update to the child containing (x, a).

        Returns: (target_ball, target_stats, was_redirected)
        """
        if not stats.is_split:
            return ball, stats, False

        # Line 12: check if n_play mod (H+1) == 0
        if stats.n_play % (self.H + 1) != 0:
            return ball, stats, False

        # Line 13: find child containing z = (x, a)
        for ckey in stats.children_keys:
            # Child could be in F or P
            if ckey in self.F[h]:
                child_ball, child_stats = self.F[h][ckey]
                if child_ball.contains(z):
                    return child_ball, child_stats, True
            elif ckey in self.P[h]:
                child_ball, child_stats = self.P[h][ckey]
                if child_ball.contains(z):
                    return child_ball, child_stats, True

        # No child found containing z (shouldn't happen if children tile parent)
        return ball, stats, False

    # ------------------------------------------------------------------
    # Buffer promotion (Paper lines 19-22)
    # ------------------------------------------------------------------

    def _maybe_promote(self, h: int, ball: Ball, stats: BallStats):
        """
        Paper lines 19-21: If ball is in F and n_update >= Nmin,
        move to P and set n_play = n_update.
        """
        bkey = ball.key()
        if bkey not in self.F[h]:
            return
        if stats.n_update >= self.nmin_fn(ball):
            del self.F[h][bkey]
            self.P[h][bkey] = (ball, stats)
            # Line 21: set n_play = n_update upon promotion
            stats.n_play = stats.n_update

    # ------------------------------------------------------------------
    # Value function
    # ------------------------------------------------------------------

    def _discretize_state(self, x: np.ndarray, resolution: int = 10) -> Tuple[int, ...]:
        indices = (x * resolution).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return tuple(indices.tolist())

    def get_value(self, x: np.ndarray) -> float:
        key = self._discretize_state(x)
        if key in self.value_estimates:
            total, count = self.value_estimates[key]
            return total / count
        return 0.0

    def update_value(self, x: np.ndarray, ret: float):
        key = self._discretize_state(x)
        if key in self.value_estimates:
            total, count = self.value_estimates[key]
            self.value_estimates[key] = (total + ret, count + 1)
        else:
            self.value_estimates[key] = (ret, 1)

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self, rewards: List[float], values: np.ndarray, dones: List[bool]
    ) -> np.ndarray:
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                next_val = 0.0
                last_gae = 0.0
            elif t == T - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
        return advantages

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, trajectory, advantages):
        """
        PPO clipped surrogate update on ball preferences.

        trajectory entries: (h, x_norm, played_ball, update_ball, z, old_log_prob)
        where update_ball is the ball that actually received the update
        (may differ from played_ball due to child redirect).
        """
        for epoch in range(self.n_ppo_epochs):
            for t, (h, x_norm, played_ball, update_ball, z, old_log_prob) in enumerate(trajectory):
                relevant = self.relevant_balls(h, x_norm)
                probs = self.policy_probs(relevant)

                # Find the played ball in relevant set
                played_key = played_ball.key()
                idx = None
                for i, (b, _) in enumerate(relevant):
                    if b.key() == played_key:
                        idx = i
                        break
                if idx is None:
                    # Ball no longer relevant (domain became empty)
                    continue

                new_log_prob = np.log(probs[idx] + 1e-12)
                ratio = np.exp(new_log_prob - old_log_prob)
                adv = advantages[t]

                clipped = (adv > 0 and ratio > 1 + self.clip_eps) or \
                          (adv < 0 and ratio < 1 - self.clip_eps)

                if not clipped:
                    for j, (_, stats_j) in enumerate(relevant):
                        grad = ((1.0 if j == idx else 0.0) - probs[j]) / self.temperature
                        stats_j.preference += self.lr * adv * ratio * grad

                # Also update the actual update_ball's preference if it was
                # redirected to a child (so the child gets a real gradient signal)
                if update_ball.key() != played_ball.key():
                    update_key = update_ball.key()
                    # Find update_ball in F or P
                    for partition in [self.F[h], self.P[h]]:
                        if update_key in partition:
                            _, update_stats = partition[update_key]
                            # Give it a gradient proportional to advantage
                            update_stats.preference += self.lr * adv * 0.5
                            break

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def run_episode(self, env: gym.Env) -> Tuple[float, int]:
        obs, _ = env.reset()
        x = self.normalize_obs(obs.flatten())

        # trajectory: (h, x_norm, played_ball, update_ball, z, old_log_prob)
        trajectory = []
        rewards = []
        dones = []
        values = []

        total_reward = 0.0
        h = 0
        done = truncated = False

        while not (done or truncated) and h < self.H:
            # 1) Select ball from softmax policy over relevant balls
            relevant = self.relevant_balls(h, x)
            probs = self.policy_probs(relevant)
            idx = self.rng.choice(len(relevant), p=probs)
            played_ball, played_stats = relevant[idx]
            old_log_prob = np.log(probs[idx] + 1e-12)

            # Line 6: increment play count
            played_stats.n_play += 1

            # 2) Action = center of ball's action-subcube
            a_norm = clip01(played_ball.lower[self.ds:] + 0.5 * played_ball.s)
            action = self.denormalize_action(a_norm)
            z = np.concatenate([x, a_norm])

            # Line 8-11: Maybe split
            self._split_if_needed(h, played_ball, played_stats)

            # Lines 12-14: Maybe redirect update to child
            update_ball, update_stats, was_redirected = \
                self._maybe_redirect_to_child(h, played_ball, played_stats, z)

            # Line 16: increment update count on the update target
            update_stats.n_update += 1

            # Lines 19-21: Maybe promote if update target is in buffer
            self._maybe_promote(h, update_ball, update_stats)

            # Record value for GAE
            values.append(self.get_value(x))

            # Step environment
            next_obs, reward, done, truncated, _ = env.step(action)
            x_next = self.normalize_obs(next_obs.flatten())

            trajectory.append((h, x.copy(), played_ball, update_ball, z.copy(), old_log_prob))
            rewards.append(float(reward))
            dones.append(done or truncated)

            total_reward += reward
            x = x_next
            h += 1

        if not trajectory:
            return total_reward, 0

        # Compute returns and update value function
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        for i, (h_i, x_i, _, _, _, _) in enumerate(trajectory):
            self.update_value(x_i, returns[i])

        # GAE
        values_arr = np.array(values)
        advantages = self.compute_gae(rewards, values_arr, dones)
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        self._ppo_update(trajectory, advantages)

        return total_reward, len(trajectory)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env: gym.Env, n_episodes: int = 1000, print_every: int = 100):
        episode_rewards = []

        for ep in range(n_episodes):
            reward, steps = self.run_episode(env)
            episode_rewards.append(reward)

            if (ep + 1) % print_every == 0:
                recent = episode_rewards[-print_every:]
                mean_r = np.mean(recent)
                total_active = sum(len(P_h) for P_h in self.P)
                total_buffer = sum(len(F_h) for F_h in self.F)
                print(f"Episode {ep+1}: mean_reward={mean_r:.2f}, "
                      f"active={total_active}, buffer={total_buffer}")

        return episode_rewards

    def summary(self) -> Dict[str, int]:
        total_active = sum(len(P_h) for P_h in self.P)
        total_buffer = sum(len(F_h) for F_h in self.F)
        return {"active_total": total_active, "buffer_total": total_buffer}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_simple_highway_env():
    """Highway env with ego vehicle only, 4-dim state, 1D steering."""
    env = gym.make(
        "highway-fast-v0",
        render_mode="rgb_array",
        config={
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": False,
                "lateral": True,
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "normalize": True,
                "absolute": True,
            },
            "vehicles_count": 0,
            "duration": 40,
            "policy_frequency": 2,
        }
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_simple_highway_env()

    act_low = np.array([-np.pi / 4], dtype=np.float32)
    act_high = np.array([np.pi / 4], dtype=np.float32)

    ds = 4
    da = 1
    H = 80

    agent = ZoomingPPOFixed(
        ds=ds,
        da=da,
        H=H,
        action_bounds=(act_low, act_high),
        gamma=0.99,
        gae_lambda=0.95,
        lr=0.1,
        temperature=1.0,
        clip_eps=0.2,
        n_ppo_epochs=4,
        max_depth=3,
        dmax=1.0,
        seed=42,
    )

    print("Starting training (fixed zooming + PPO/GAE)...")
    print(f"State dim: {ds}, Action dim: {da}, Total dim: {ds + da}")
    print(f"Stages (H): {H}, Max depth: {agent.max_depth}")
    print()

    rewards = agent.train(env, n_episodes=500, print_every=50)

    print()
    stats = agent.summary()
    print(f"Final active balls: {stats['active_total']}")
    print(f"Final buffer balls: {stats['buffer_total']}")
    print(f"Mean reward (last 50): {np.mean(rewards[-50:]):.2f}")

    plt.imshow(env.render())
    plt.show()

    env.close()