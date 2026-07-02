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

**One script per family** under `scripts/`, with a *phase* selector. GPU is auto-detected. All plots read the **deterministic-eval curve** persisted in each checkpoint (the online training reward understates the greedy policy by up to ~3x, so it is never the headline metric).

```bash
scripts/run_dmcs.sh <task> [phase]      # task: cartpole-swingup | walker-walk | cheetah-run
scripts/run_highway.sh [phase]          # racetrack-v0 (da=1); no task arg
```

### The three phases

- **`efficiency`** (default, the headline figure) — uniform + zooming across `N`, with SAC as a flat ceiling band. This is the performance-vs-`N` curve: read it matched at each `N`, as the `N` where zooming reaches uniform's best plateau, and as each arm's best-over-`N` envelope vs the SAC ceiling. Merges the old *action-sweep* + *architectures* phases. → `plots/<family>/<task>_efficiency.png`
- **`robustness`** — adjudicates the auto-tuning claim ("you don't have to pick `N`"). Adds zooming knob variants (`buffer_period`, `init_depth`) at `REF_N` and compares the **spread** of zooming-across-knobs to uniform-across-`N` (reusing the efficiency checkpoints). Prints a quantitative verdict (range ratio). → `plots/<family>/<task>_robustness.png`
- **`sample-efficiency`** — the repurposed timestep sweep: long (1M-step) runs at 1–2 diagnostic `N` for uniform + zooming + SAC, plotted as **reward vs env-steps** for the "cheaper than SAC" thread. → `plots/<family>/<task>_sample_efficiency.png`
- **`all`** — the three in order.

### Recommended workflow

Run **one task's `efficiency` end-to-end first**, eyeball the curve, *then* fan out — don't launch every task × phase on faith right after a refactor.

```bash
scripts/run_dmcs.sh walker-walk efficiency     # validate the fixed code on the cleanest task
scripts/run_dmcs.sh walker-walk robustness     # then the auto-tuning story
scripts/run_dmcs.sh walker-walk all            # or the whole thing
scripts/run_highway.sh efficiency              # da=1 sanity task
```

Preview any invocation without running it: `DRY_RUN=1 scripts/run_dmcs.sh walker-walk all`.

### Task roles (a planning decision, not interchangeable rows)

- **cartpole-swingup, walker-walk** — "discretization works here"; worth the full efficiency + robustness story (more seeds).
- **cheetah-run** — the boundary case (action dim 6, where SAC is expected to win). Default phase is `efficiency` only: one honest figure, not the full machinery.

### Knobs (env-var overrides; see each script's header)

`SEEDS` (default `42 43 44`), `N_VALUES`, `REF_N`, `BUFFER_PERIODS`/`INIT_DEPTHS` (zooming robustness grid), `SE_N`/`SE_TIMESTEPS` (sample-efficiency), `DQN_TIMESTEPS`/`SAC_TIMESTEPS` (per-task defaults), `PYTHON`, `DRY_RUN`. `buffer_period` is zooming's one novel knob (the paper's `H+1` buffering cadence, adapted to neural+replay); smaller = children graduate faster.

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
