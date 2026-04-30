# Adaptive Discretization for Policy Optimization

Research project exploring theoretically-grounded adaptive action space discretization for continuous MDPs.

## Installation

```bash
conda create -n highway python=3.11
conda activate highway
pip install -r requirements.txt
git clone https://github.com/eleurent/highway-env.git HighwayEnv
```

## Running experiments

Three independent scripts per task family under `scripts/`. GPU is auto-detected if available.

### Per-task pipeline (recommended)

For each task, run the three phases in order. Phase 2 and Phase 3 each require an `N_ACTIONS` value picked from Phase 1's plot.

```bash
# Phase 1 -- Action sweep: where does zooming beat uniform?
#   24 runs (4 N values x 3 seeds x 2 arms) at fixed 300k timesteps.
#   Output: plots/dmcs/<task>_action_sweep.png
./scripts/run_dmcs_action_sweep.sh cartpole-swingup

# Inspect the plot, pick a target N (the one where zooming most clearly
# wins over uniform, or where curves diverge most). Default if flat: 32.

# Phase 2 -- Architectures sweep at chosen N: SAC vs Uniform vs Zooming.
#   15 runs (5 seeds x 3 arms) at the per-task DQN_TIMESTEPS default.
#   Output: plots/dmcs/<task>_architectures.png  (the headline plot.)
N_ACTIONS=32 ./scripts/run_dmcs_architectures.sh cartpole-swingup

# Phase 3 -- Timestep sweep at chosen N: asymptotic behavior.
#   6 runs (3 seeds x 2 arms) at TS_TIMESTEPS=1M each.
#   Output: plots/dmcs/<task>_timestep_sweep.png
TS_N_ACTIONS=32 ./scripts/run_dmcs_timestep_sweep.sh cartpole-swingup
```

Repeat for the remaining DMCS tasks; pick `N_ACTIONS` / `TS_N_ACTIONS` fresh for each based on its own action-sweep plot:

```bash
./scripts/run_dmcs_action_sweep.sh cheetah-run
N_ACTIONS=<chosen> ./scripts/run_dmcs_architectures.sh cheetah-run
TS_N_ACTIONS=<chosen> ./scripts/run_dmcs_timestep_sweep.sh cheetah-run

./scripts/run_dmcs_action_sweep.sh walker-walk
N_ACTIONS=<chosen> ./scripts/run_dmcs_architectures.sh walker-walk
TS_N_ACTIONS=<chosen> ./scripts/run_dmcs_timestep_sweep.sh walker-walk
```

### Highway (racetrack-v0, 1-D action)

Same three-phase structure; `racetrack-v0` is the only task so no positional arg.

```bash
./scripts/run_highway_action_sweep.sh                              # Uniform vs Zooming, N \in {8,16,32,64}
N_ACTIONS=<chosen> ./scripts/run_highway_architectures.sh          # SAC vs Uniform vs Zooming
TS_N_ACTIONS=<chosen> ./scripts/run_highway_timestep_sweep.sh      # long-horizon at chosen N
```

All scripts accept env-var overrides (`SEEDS`, `N_ACTIONS`, `DQN_TIMESTEPS`, etc.). See each script's header for the full list.

Default seed counts are tuned to give usable error bars without burning the GPU:
- architectures sweeps -- 5 seeds (`SEEDS="42 43 44 45 46"`)
- timestep sweeps -- 3 seeds (`TS_SEEDS="42 43 44"`)
- action sweeps -- 3 seeds, hardcoded in `src/{dmcs,highway}/run_action_sweep.py`

For a quick smoke test, override with a single seed (e.g. `SEEDS=42 ./scripts/run_dmcs_architectures.sh` or `TS_SEEDS=42 ./scripts/run_highway_timestep_sweep.sh`); for the action sweeps, edit `SEEDS = [42, 43, 44]` in the corresponding `run_action_sweep.py`.

Training-timestep defaults vary by experiment and (for DMCS) by task:
- DMCS architectures -- 150k for cartpole-swingup, 300k for walker-walk, 500k for cheetah-run (override via `DQN_TIMESTEPS` / `SAC_TIMESTEPS`).
- DMCS timestep sweep -- 1M (`TS_TIMESTEPS`), long enough to read asymptotic behavior at large N.
- DMCS action sweep -- 300k, hardcoded; intentionally compute-bounded so the uniform-vs-zooming gap at large N reflects the limited-budget regime.
- Highway architectures -- 150k (`DQN_TIMESTEPS` / `SAC_TIMESTEPS`).
- Highway timestep sweep -- 600k (`TS_TIMESTEPS`).
- Highway action sweep -- 150k, hardcoded.

Outputs land under `checkpoints/<family>/<task>/<phase>/` and `plots/<family>/...`.

## Single-arm runs

Each arm is invocable directly for ad-hoc experiments:

```bash
# Highway (racetrack-v0)
python src/highway/run_sac.py     --seed 42
python src/highway/run_uniform.py --seed 42 --n_actions 16
python src/highway/run_zooming.py --seed 42 --n_actions 16 --init_depth 3

# DMCS
python src/dmcs/run_sac.py     --task walker-walk --seed 42
python src/dmcs/run_uniform.py --task walker-walk --seed 42 --n_actions 16
python src/dmcs/run_zooming.py --task walker-walk --seed 42 --n_actions 16 --init_depth 1
```

Matched-budget A/B: pass the same `--n_actions` to `run_uniform.py` and `run_zooming.py`. Zooming's total cell count is then capped at `n_actions * da` (the same as uniform's), so any difference between the arms is about adaptive placement, not resolution.

`--init_depth` controls how coarse zooming starts: `2 ** init_depth` bins per axis at step 0, refined toward the budget over training. Lower `init_depth` gives the algorithm more split decisions to make at the cost of poorer early performance; higher gives a warmer start with less adaptive room.
