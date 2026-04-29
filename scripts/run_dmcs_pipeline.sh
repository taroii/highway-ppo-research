#!/usr/bin/env bash
#
# Architectures + timestep sweep on a DMCS task.
# SAC vs (factored) Uniform vs (factored) Zooming for each seed,
# then plot, then a long-horizon timestep sweep at higher action budget.
#
# Outputs:
#   checkpoints/dmcs/<task>/architectures/<arm>_seed<S>.pt
#   checkpoints/dmcs/<task>/timestep_sweep/<arm>_seed<S>.pt
#   plots/dmcs/<task>_architectures.png
#   plots/dmcs/<task>_timestep_sweep.png
#
# Usage:
#   ./scripts/run_dmcs_pipeline.sh                  # default task: cartpole-swingup
#   ./scripts/run_dmcs_pipeline.sh walker-walk
#   ./scripts/run_dmcs_pipeline.sh cheetah-run
#
# Task slugs are passed straight to gym.make("dm_control/<task>-v0").
#
# Knobs (env vars):
#   TASK               positional arg overrides; default "cartpole-swingup"
#   SEEDS              architectures phase, default "42 43 44 45 46"
#   SAC_TIMESTEPS      default 300000
#   DQN_TIMESTEPS      default 300000   (uniform + zooming, architectures phase)
#   N_ACTIONS          architectures phase action budget per axis, default 16
#   INIT_DEPTH         zooming starting depth, default 1
#                      (= 2 bins per axis at start; max adaptive room)
#   TS_SEEDS           timestep sweep seeds (hardcoded budget arm), default "42"
#   TS_N_ACTIONS       timestep sweep action budget per axis, default 64
#   TS_TIMESTEPS       timestep sweep total steps, default 600000
#   PYTHON             interpreter, default "python"
#
# Per-run stdout is tee'd to logs/dmcs/<task>/<phase>/<label>.log.
# A failing run is logged and the pipeline continues.
#
set -uo pipefail

TASK="${1:-cartpole-swingup}"

SEEDS="${SEEDS:-42 43 44 45 46}"
SAC_TIMESTEPS="${SAC_TIMESTEPS:-300000}"
DQN_TIMESTEPS="${DQN_TIMESTEPS:-300000}"
N_ACTIONS="${N_ACTIONS:-16}"
INIT_DEPTH="${INIT_DEPTH:-1}"
TS_SEEDS="${TS_SEEDS:-42}"
TS_N_ACTIONS="${TS_N_ACTIONS:-64}"
TS_TIMESTEPS="${TS_TIMESTEPS:-600000}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

ARCH_CKPT_DIR="checkpoints/dmcs/${TASK}/architectures"
ARCH_LOG_DIR="logs/dmcs/${TASK}/architectures"
ARCH_PLOT_OUT="plots/dmcs/${TASK}_architectures.png"
TS_CKPT_DIR="checkpoints/dmcs/${TASK}/timestep_sweep"
TS_LOG_DIR="logs/dmcs/${TASK}/timestep_sweep"
TS_PLOT_OUT="plots/dmcs/${TASK}_timestep_sweep.png"
mkdir -p "$ARCH_CKPT_DIR" "$ARCH_LOG_DIR" "$TS_CKPT_DIR" "$TS_LOG_DIR" \
         "$(dirname "$ARCH_PLOT_OUT")"

failed=()
total=0
ok=0

run_one() {
    local label="$1"; shift
    local log_dir="$1"; shift
    local log="$log_dir/${label}.log"
    total=$((total + 1))
    echo
    echo "=== [$(date '+%F %T')] START  $label  ==="
    echo "    cmd: $*"
    echo "    log: $log"
    if "$@" 2>&1 | tee "$log"; then
        ok=$((ok + 1))
        echo "=== [$(date '+%F %T')] OK     $label  ==="
    else
        failed+=("$label")
        echo "=== [$(date '+%F %T')] FAIL   $label (continuing) ===" >&2
    fi
}

echo "dmcs pipeline"
echo "  task:              $TASK"
echo "  arch seeds:        $SEEDS"
echo "  sac timesteps:     $SAC_TIMESTEPS"
echo "  dqn timesteps:     $DQN_TIMESTEPS"
echo "  n_actions:         $N_ACTIONS"
echo "  init_depth:        $INIT_DEPTH"
echo "  ts seeds:          $TS_SEEDS"
echo "  ts n_actions:      $TS_N_ACTIONS"
echo "  ts timesteps:      $TS_TIMESTEPS"
echo "  python:            $PYTHON"
echo "  arch ckpts → $ARCH_CKPT_DIR"
echo "  arch plot  → $ARCH_PLOT_OUT"
echo "  ts ckpts   → $TS_CKPT_DIR"
echo "  ts plot    → $TS_PLOT_OUT"

# ------------------------------------------------------------------
# Phase 1: architectures (SAC vs Uniform vs Zooming)
# ------------------------------------------------------------------
echo
echo "=== phase 1: architectures ==="
for seed in $SEEDS; do
    run_one "sac_seed${seed}" "$ARCH_LOG_DIR" \
        "$PYTHON" src/dmcs/run_sac.py \
            --task "$TASK" --seed "$seed" \
            --total_timesteps "$SAC_TIMESTEPS" \
            --output "$ARCH_CKPT_DIR/sac_seed${seed}.pt"

    run_one "uniform_n${N_ACTIONS}_seed${seed}" "$ARCH_LOG_DIR" \
        "$PYTHON" src/dmcs/run_uniform.py \
            --task "$TASK" --seed "$seed" --n_actions "$N_ACTIONS" \
            --total_timesteps "$DQN_TIMESTEPS" \
            --output "$ARCH_CKPT_DIR/uniform_n${N_ACTIONS}_seed${seed}.pt"

    run_one "zooming_n${N_ACTIONS}_seed${seed}" "$ARCH_LOG_DIR" \
        "$PYTHON" src/dmcs/run_zooming.py \
            --task "$TASK" --seed "$seed" \
            --init_depth "$INIT_DEPTH" --n_actions "$N_ACTIONS" \
            --total_timesteps "$DQN_TIMESTEPS" \
            --output "$ARCH_CKPT_DIR/zooming_n${N_ACTIONS}_seed${seed}.pt"
done

echo
echo "=== running architectures compare.py ==="
"$PYTHON" src/dmcs/compare.py \
    --task "$TASK" \
    --checkpoints-dir "$ARCH_CKPT_DIR" \
    --output "$ARCH_PLOT_OUT" \
    --title "Architectures — dm_control/${TASK}"

# ------------------------------------------------------------------
# Phase 2: timestep sweep (uniform vs zooming at large N, long training)
# ------------------------------------------------------------------
echo
echo "=== phase 2: timestep sweep (uniform vs zooming, N=$TS_N_ACTIONS, ${TS_TIMESTEPS} steps) ==="
for seed in $TS_SEEDS; do
    run_one "uniform_n${TS_N_ACTIONS}_seed${seed}" "$TS_LOG_DIR" \
        "$PYTHON" src/dmcs/run_uniform.py \
            --task "$TASK" --seed "$seed" --n_actions "$TS_N_ACTIONS" \
            --total_timesteps "$TS_TIMESTEPS" \
            --output "$TS_CKPT_DIR/uniform_n${TS_N_ACTIONS}_seed${seed}.pt"

    run_one "zooming_n${TS_N_ACTIONS}_seed${seed}" "$TS_LOG_DIR" \
        "$PYTHON" src/dmcs/run_zooming.py \
            --task "$TASK" --seed "$seed" \
            --init_depth "$INIT_DEPTH" --n_actions "$TS_N_ACTIONS" \
            --total_timesteps "$TS_TIMESTEPS" \
            --output "$TS_CKPT_DIR/zooming_n${TS_N_ACTIONS}_seed${seed}.pt"
done

echo
echo "=== running timestep_sweep compare.py ==="
"$PYTHON" src/dmcs/compare.py \
    --task "$TASK" \
    --checkpoints-dir "$TS_CKPT_DIR" \
    --output "$TS_PLOT_OUT" \
    --title "Timestep sweep — dm_control/${TASK}  (N=${TS_N_ACTIONS})"

echo
echo "=== runs complete: ${ok}/${total} succeeded ==="
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed runs:"
    printf "  %s\n" "${failed[@]}"
fi

echo
echo "=== pipeline done ==="
