# DeepMind Control Suite experiments

This folder is a placeholder for porting the action-discretization study
to the DeepMind Control Suite (DMCS).

## Goal

Same as `src/mujoco/`: reproduce the matched-action-budget A/B
(SAC reference, Uniform DQN, Zooming DQN) and sweep action counts x seeds.

## DMCS specifics

DMCS exposes tasks via `dm_control.suite` (or `gymnasium-dm-control`).
Common tasks (cartpole-swingup, walker-walk, cheetah-run,
finger-spin) have **continuous multi-dimensional actions** and **dense
rewards**, so they're a fairer terrain for action-discretization
methods than racetrack.

You'll likely want the `gymnasium` wrapper:

```python
import gymnasium as gym
env = gym.make("dm_control/cartpole-swingup-v0")  # or via shimmy
```

## Environment caveats

Same as MuJoCo: multi-dim actions mean `2^da` cubes per split.  See
`src/mujoco/README.md` section Environment caveats for the factored vs joint
zooming discussion.

## Suggested files (mirror `src/highway/`)

```
src/dmcs/
  env.py              # task factories (cartpole-swingup, cheetah-run, ...)
  run_sac.py
  run_uniform.py
  run_zooming.py
  run_scarcity_sweep.py
  compare.py
```

Reuse the core modules from `src/highway/` (SAC, DQN, zooming, uniform
grid, action_manager). Only the env factory and hyperparameter defaults
should differ.

Output checkpoints to `checkpoints/dmcs/`, plots to `plots/dmcs/`.

## Open questions for collaborators

- Task selection. Cartpole-swingup is small and fast (good for sweep
  iteration); cheetah-run and walker-walk are the classic DMCS
  benchmarks. TARO's Notes: these three tasks are enough.
- Pixel vs feature observations. DMCS exposes both; feature obs are
  lower-dim and faster.
- Episode length. DMCS episodes are 1000 steps by default; consider
  using shorter ones for faster wall-clock during the sweep.
