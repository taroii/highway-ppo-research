from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


def clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


@dataclass
class Cube:
    """
    Axis-aligned cube in [0,1]^d where d = ds + da (state+action dims).

    Stored by:
      - lower: bottom-left corner (d,)
      - s: side length (float)
    """
    lower: np.ndarray
    s: float
    d: int

    def contains(self, z: np.ndarray, eps: float = 1e-12) -> bool:
        """Containment in the full joint space."""
        upper = self.lower + self.s
        return bool(np.all(z >= self.lower - eps) and np.all(z <= upper + eps))

    def contains_state(self, x: np.ndarray, ds: int, eps: float = 1e-12) -> bool:
        """Containment only in the state coordinates (projection onto first ds dims)."""
        lower_s = self.lower[:ds]
        upper_s = lower_s + self.s
        return bool(np.all(x >= lower_s - eps) and np.all(x <= upper_s + eps))

    def split_children(self) -> List["Cube"]:
        """
        Split into 2^d cubes of half side length.
        Children lower corners are lower + offset*(s/2), offset ∈ {0,1}^d.
        """
        half = self.s / 2.0
        children: List[Cube] = []
        for mask in range(1 << self.d):
            offset = np.array([(mask >> i) & 1 for i in range(self.d)], dtype=float)
            child_lower = self.lower + offset * half
            children.append(Cube(lower=child_lower, s=half, d=self.d))
        return children

    def key(self) -> Tuple[float, ...]:
        """Hashable key for dicts."""
        return (*map(float, self.lower.tolist()), float(self.s))


@dataclass
class CubeStats:
    # Optimistic estimate / Q-like value
    Q: float
    # Counters
    n_play: int = 0
    n_update: int = 0
    # Children created?
    is_split: bool = False
    children: List[Cube] = field(default_factory=list)


class ZoomingAdaptiveQLearning:
    """
    Adaptive Q-learning with zooming dimension (practical skeleton).

    Key change from the original version:
      - The algorithm maintains cubes in the JOINT space [0,1]^(ds+da),
        but "relevant cubes" are defined by the STATE only:
            relevant(x) = { active cubes whose state-projection contains x
                            AND have the smallest side length among those }
        i.e. if a larger cube contains x but some smaller active cube also contains x,
        we do NOT include the larger cube.

    We also separate state/action interfaces:
      - env_step(h, x, a) -> (x_next, reward)
      - x0_sampler(k) -> x0

    Action selection (placeholder):
      - choose action as the center of the cube's action-subcube (or random in it).
    """

    def __init__(
        self,
        ds: int,
        da: int,
        H: int,
        K: int,
        seed: int = 0,
        # thresholds: replace with paper's exact formulas later
        nsplit_fn: Optional[Callable[[Cube], int]] = None,
        nmin_fn: Optional[Callable[[Cube], int]] = None,
        # Q initialization: override if you want deterministic/external init
        q_init_fn: Optional[Callable[[Cube, int], float]] = None,
        # action selection inside chosen cube
        action_mode: str = "center",  # "center" or "random"
    ):
        if ds <= 0 or da <= 0:
            raise ValueError("ds and da must both be positive integers.")
        self.ds = ds
        self.da = da
        self.d = ds + da

        self.H = H
        self.K = K
        self.rng = np.random.default_rng(seed)

        # Default thresholds similar in spirit to the paper.
        # You can (and probably will) swap these to match the paper's Nsplit/Nmin.
        self.nsplit_fn = nsplit_fn or (lambda cube: int(np.ceil((1.0 / cube.s) ** 2)))
        self.nmin_fn = nmin_fn or (lambda cube: max(1, self.nsplit_fn(cube) // 4))

        # Default Q init: optimistic upper bound
        self.q_init_fn = q_init_fn or (lambda cube, h: float(self.H))

        self.action_mode = action_mode
        if self.action_mode not in ("center", "random"):
            raise ValueError("action_mode must be 'center' or 'random'")

        # Per-stage storage: dict[cube_key] -> (Cube, CubeStats)
        self.P: List[Dict[Tuple[float, ...], Tuple[Cube, CubeStats]]] = []  # active/playing
        self.F: List[Dict[Tuple[float, ...], Tuple[Cube, CubeStats]]] = []  # buffer

        self._initialize_partitions()

    def _initialize_partitions(self):
        # Start with a single cube covering the full [0,1]^d for each stage.
        root = Cube(lower=np.zeros(self.d), s=1.0, d=self.d)
        for h in range(self.H):
            P_h: Dict[Tuple[float, ...], Tuple[Cube, CubeStats]] = {}
            F_h: Dict[Tuple[float, ...], Tuple[Cube, CubeStats]] = {}
            stats = CubeStats(Q=self.q_init_fn(root, h + 1))
            P_h[root.key()] = (root, stats)
            self.P.append(P_h)
            self.F.append(F_h)

    # ---------- Relevance / selection (STATE-ONLY relevance) ----------

    def relevant_active_cubes(self, h: int, x: np.ndarray) -> List[Tuple[Cube, CubeStats]]:
        """
        Relevant cubes for state x at stage h:

        1) Consider all ACTIVE cubes whose state projection contains x.
        2) Among those, keep only those with MINIMUM side length s.

        This enforces: if a cube contains x but some smaller active cube also contains x,
        we exclude the larger cube.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.ds,):
            raise ValueError(f"x must have shape ({self.ds},), got {x.shape}")
        x = clip01(x)

        candidates: List[Tuple[Cube, CubeStats]] = []
        for cube, stats in self.P[h].values():
            if cube.contains_state(x, self.ds):
                candidates.append((cube, stats))

        if not candidates:
            raise RuntimeError("No active cube contains the state x (coverage invariant broken).")

        min_s = min(cube.s for cube, _ in candidates)
        tol = 1e-15
        relevant = [(cube, stats) for cube, stats in candidates if abs(cube.s - min_s) <= tol]
        return relevant

    def select_cube(self, h: int, x: np.ndarray) -> Tuple[Cube, CubeStats]:
        """Select argmax-Q cube among relevant cubes (defined by state-only)."""
        rel = self.relevant_active_cubes(h, x)
        best_cube, best_stats = max(rel, key=lambda cs: cs[1].Q)
        best_stats.n_play += 1
        return best_cube, best_stats

    # ---------- Action choice within a cube ----------

    def action_from_cube(self, cube: Cube) -> np.ndarray:
        """
        Choose an action inside the cube's action-subcube.
        The cube covers joint coords [x,a]. The action region is dims ds:ds+da.

        Placeholder policies:
          - 'center': pick center of the action-subcube
          - 'random': uniform random in the action-subcube
        """
        a_low = cube.lower[self.ds:]
        if self.action_mode == "center":
            a = a_low + 0.5 * cube.s
        else:
            a = a_low + self.rng.random(self.da) * cube.s
        return clip01(a)

    # ---------- Storage helpers ----------

    def _get_stats(self, h: int, cube: Cube) -> CubeStats:
        key = cube.key()
        if key in self.P[h]:
            return self.P[h][key][1]
        if key in self.F[h]:
            return self.F[h][key][1]
        raise KeyError("Cube not found in active or buffer sets.")

    def _add_to_buffer(self, h: int, cube: Cube):
        key = cube.key()
        if key in self.P[h] or key in self.F[h]:
            return
        self.F[h][key] = (cube, CubeStats(Q=self.q_init_fn(cube, h + 1)))

    def _move_buffer_to_active_if_ready(self, h: int, cube: Cube):
        key = cube.key()
        if key not in self.F[h]:
            return
        cube_obj, stats = self.F[h][key]
        if stats.n_update >= self.nmin_fn(cube_obj):
            # Move to active/playing
            del self.F[h][key]
            self.P[h][key] = (cube_obj, stats)
            # Like line 21 in the pseudocode: set play count to update count (optional)
            stats.n_play = stats.n_update

    # ---------- Splitting / child redirection ----------

    def _split_if_needed(self, h: int, cube: Cube, stats: CubeStats):
        """
        If play-count reaches Nsplit, split cube (if not already split),
        create 2^d children and add them to buffer.
        """
        if stats.n_play < self.nsplit_fn(cube):
            return

        if not stats.is_split:
            stats.is_split = True
            stats.children = cube.split_children()
            for child in stats.children:
                self._add_to_buffer(h, child)

    def _maybe_redirect_update_to_child(self, h: int, cube: Cube, stats: CubeStats, z: np.ndarray) -> Cube:
        """
        Implements the idea of line 12-13:
        every (H+1)-th play after splitting, update the CHILD cube that contains (x,a) in joint space.
        """
        if not stats.is_split:
            return cube

        if stats.n_play % (self.H + 1) == 0:
            for child in stats.children:
                if child.contains(z):
                    return child
        return cube

    # ---------- Value estimate (STATE-ONLY) ----------

    def estimate_next_value(self, h_next: int, x_next: np.ndarray) -> float:
        """
        V_{h+1}(x_{h+1}) analog:
        Use max Q among relevant cubes for the next STATE (state-only relevance).
        """
        if h_next >= self.H:
            return 0.0
        rel = self.relevant_active_cubes(h_next, x_next)
        return float(min(self.H, max(stats.Q for _, stats in rel)))

    # ---------- Q update (placeholder) ----------

    def update_Q_random(self, h: int, cube: Cube, reward: float, next_value: float):
        """
        Placeholder update so the loop runs end-to-end.
        Replace with your actual optimistic Q-learning update rule later.
        """
        stats = self._get_stats(h, cube)
        stats.n_update += 1

        target = float(reward + next_value)
        alpha = 0.2
        noise = self.rng.normal(0.0, 0.01)
        stats.Q = (1 - alpha) * stats.Q + alpha * (target + noise)

        # If cube is in buffer, see if it should be activated
        self._move_buffer_to_active_if_ready(h, cube)

    # ---------- Main training loop (state/action separated) ----------

    def run(
        self,
        env_step: Callable[[int, np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        x0_sampler: Callable[[int], np.ndarray],
        verbose: bool = False,
    ):
        """
        env_step(h, x, a) -> (x_next, reward)
          - h is stage index 0..H-1
          - x is ds-dim state in [0,1]^ds
          - a is da-dim action in [0,1]^da
        x0_sampler(k) -> initial state x0

        Mechanics:
          - choose cube using state-only relevance
          - choose action within cube's action region
          - build joint point z=[x,a] for child-redirection logic
          - split/update as before
        """
        for k in range(self.K):
            x = clip01(np.asarray(x0_sampler(k), dtype=float))
            if x.shape != (self.ds,):
                raise ValueError(f"x0_sampler must return shape ({self.ds},), got {x.shape}")

            if verbose:
                print(f"\nEpisode {k+1}/{self.K}")

            for h in range(self.H):
                # 1) choose cube by optimism among relevant cubes (state-only relevance)
                cube, stats = self.select_cube(h, x)

                # 2) choose action inside that cube's action-subcube
                a = self.action_from_cube(cube)

                # joint vector used for redirect-to-child containing (x,a)
                z = np.concatenate([x, a])

                # 3) split if needed (based on plays of the chosen cube)
                self._split_if_needed(h, cube, stats)

                # 4) possibly redirect update to child (based on joint containment)
                update_cube = self._maybe_redirect_update_to_child(h, cube, stats, z)

                # 5) interact with env
                x_next, reward = env_step(h, x, a)
                x_next = clip01(np.asarray(x_next, dtype=float))
                if x_next.shape != (self.ds,):
                    raise ValueError(f"env_step must return x_next shape ({self.ds},), got {x_next.shape}")

                # 6) next value from next state (state-only relevance)
                next_value = self.estimate_next_value(h + 1, x_next)

                # 7) update chosen (possibly redirected) cube
                self.update_Q_random(h, update_cube, float(reward), next_value)

                if verbose:
                    print(
                        f"  stage {h+1}: "
                        f"x={x}, a={a}, "
                        f"picked_s={cube.s:.4f}, update_s={update_cube.s:.4f}, "
                        f"r={reward:.3f}, Vnext={next_value:.3f}"
                    )

                x = x_next

    # ---------- Debug helpers ----------

    def summary(self) -> Dict[str, int]:
        total_active = sum(len(P_h) for P_h in self.P)
        total_buffer = sum(len(F_h) for F_h in self.F)
        return {"active_total": total_active, "buffer_total": total_buffer}


# --------------------------
# Example usage (toy env)
# --------------------------
if __name__ == "__main__":
    ds, da = 2, 1
    H, K = 5, 30
    alg = ZoomingAdaptiveQLearning(ds=ds, da=da, H=H, K=K, seed=42, action_mode="center")

    def x0_sampler(k: int) -> np.ndarray:
        return np.random.rand(ds)

    def env_step(h: int, x: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float]:
        # toy dynamics: state does a small random walk influenced a bit by action
        x_next = x + 0.05 * np.random.randn(ds) + 0.02 * (a.mean() - 0.5)
        # toy reward: higher near center of state space, penalize large action
        reward = float(1.0 - np.linalg.norm(x - 0.5) / np.sqrt(ds) - 0.1 * np.linalg.norm(a - 0.5))
        reward = max(0.0, min(1.0, reward))
        return x_next, reward

    alg.run(env_step=env_step, x0_sampler=x0_sampler, verbose=True)
    print("\nSummary:", alg.summary())