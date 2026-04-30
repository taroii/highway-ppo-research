# MuJoCo (Gymnasium MuJoCo) experiments

This folder is a placeholder for porting the action-discretization study
to Gymnasium's MuJoCo environments (HalfCheetah, Hopper, Walker2d, Ant,
Humanoid, etc.).

## Goal

Reproduce the matched-action-budget A/B from `src/highway/`:

- **SAC** as the continuous upper-bound reference.
- **Uniform DQN** with a fixed `N`-action grid.
- **Zooming DQN** with adaptive refinement up to `N` cubes (matched
  budget so any difference is about *placement*, not resolution).

Then sweep `N \in {8, 16, 32, 64, ...}` x seeds to confirm or refute the
finding from the highway results: zooming's advantage grows with `N`
because uniform pays the full `N`-output Q-learning cost upfront while
zooming bootstraps from a coarser starting grid.

## Environment caveats

Most MuJoCo tasks have **multi-dimensional continuous actions** (e.g.,
HalfCheetah is 6-D). The zooming primitives in `src/highway/zooming.py`
already support `da > 1` -- a cube of side `s` in `[0, 1]^da` splits
into `2^da` children. But action-set sizes grow exponentially:

  - `da=6`, `init_depth=1` -> 64 starting cubes.
  - `da=6`, `max_depth=2` -> 4096 cubes max -- too many.

Practical options:
  1. Use `init_depth=0` (start from one cube, the centroid) and let
     zooming refine. UCB-driven splits should keep the active set small.
  2. Implement a **factored** zooming scheme: one `ActionZooming` per
     action axis. Total active cubes = `da * n_per_axis` instead of
     `n_per_axis^da`. Closer to factored Q-learning.
  3. Cap `max_depth` aggressively (e.g., 1 or 2) and accept that
     adaptive refinement is shallow on high-dim action spaces.

Pick the option that best preserves the experiment's interpretability;
factored zooming is probably the cleanest for a paper.

## Suggested files (mirror `src/highway/`)

```
src/mujoco/
  env.py              # gym.make("HalfCheetah-v4") etc., per-task factories
  run_sac.py          # SAC reference
  run_uniform.py      # uniform DQN
  run_zooming.py      # zooming DQN
  run_scarcity_sweep.py
  compare.py
```

Reuse `src/highway/sac.py`, `src/highway/dqn.py`,
`src/highway/zooming.py`, `src/highway/uniform_grid.py`,
`src/highway/action_manager.py` as-is. Only the env factory and run
script defaults need to change.

Output checkpoints to `checkpoints/mujoco/`, plots to `plots/mujoco/`.

## Open questions for collaborators

- Which subset of MuJoCo tasks to evaluate? Suggested starting point:
  HalfCheetah (smooth, dense reward), Hopper (sparser, easier to crash --
  may show the largest zooming benefit), Walker2d. 
  TARO's ANSWSER: Use Ant or Humanoid. We only need one to show benefit on high dimensional action space.
- Hyperparameters: SAC ones from `src/highway/sac.py` are tuned for
  racetrack (gamma=0.9, short episodes). MuJoCo typically wants
  gamma=0.99 and longer training (~1M steps).
- Action factoring vs joint zooming -- see "Environment caveats" above.
