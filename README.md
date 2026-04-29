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

Each script runs the architectures A/B (SAC vs Uniform vs Zooming) for several seeds, then a long-horizon timestep sweep at a larger action budget. GPU is auto-detected if available.

```bash
# Highway (racetrack-v0, 1-D action)
./scripts/run_highway_pipeline.sh

# DMCS — task slug as positional arg
./scripts/run_dmcs_pipeline.sh                    # cartpole-swingup (default, 1-D)
./scripts/run_dmcs_pipeline.sh walker-walk        # 6-D action
./scripts/run_dmcs_pipeline.sh cheetah-run        # 6-D action
```

Both pipelines accept env-var overrides (`SEEDS`, `N_ACTIONS`, `DQN_TIMESTEPS`, etc.). See each script's header for the full list.

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
