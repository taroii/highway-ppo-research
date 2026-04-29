#!/usr/bin/env bash
#
# Architectures experiment — SAC vs Uniform vs Zooming on racetrack-v0.
# Runs all three across multiple seeds, then plots.
#
# Outputs:
#   checkpoints/highway/architectures/<arm>_seed<S>.pt
#   plots/highway/architectures.png
#
# This is a different experiment from src/highway/run_scarcity_sweep.py.
# The pipeline tests whether *adaptive zooming* beats *uniform discretization*
# at a single matched action budget; the scarcity sweep tests how that gap
# scales with action budget N.
#
# Knobs (env vars):
#   SEEDS              space-separated list, default "42 43 44 45 46"
#   SAC_TIMESTEPS      default 50000
#   DQN_TIMESTEPS      default 150000   (uniform + zooming)
#   N_ACTIONS          action budget per axis, default 16.  Uniform uses
#                      this as bins per axis; zooming uses it as the
#                      total-cells budget (n_actions * da).
#   INIT_DEPTH         zooming starting depth (2^INIT_DEPTH bins per axis
#                      at start), default 3.
#   PYTHON             interpreter, default "python" — override on Windows:
#                        PYTHON="/c/Users/Polar/miniconda3/envs/highway/python.exe" ./run_highway_pipeline.sh
#
# Per-run stdout is tee'd to logs/highway/architectures/<label>.log.
# A failing run is logged and the pipeline continues.
#
set -uo pipefail

SEEDS="${SEEDS:-42 43 44 45 46}"
SAC_TIMESTEPS="${SAC_TIMESTEPS:-150000}"
DQN_TIMESTEPS="${DQN_TIMESTEPS:-150000}"
N_ACTIONS="${N_ACTIONS:-16}"
INIT_DEPTH="${INIT_DEPTH:-3}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CKPT_DIR="checkpoints/highway/architectures"
LOG_DIR="logs/highway/architectures"
PLOT_OUT="plots/highway/architectures.png"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$(dirname "$PLOT_OUT")"

failed=()
total=0
ok=0

run_one() {
    local label="$1"; shift
    local log="$LOG_DIR/${label}.log"
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

echo "highway architectures pipeline"
echo "  seeds:          $SEEDS"
echo "  sac timesteps:  $SAC_TIMESTEPS"
echo "  dqn timesteps:  $DQN_TIMESTEPS"
echo "  n_actions:      $N_ACTIONS"
echo "  init_depth:     $INIT_DEPTH"
echo "  python:         $PYTHON"
echo "  ckpts → $CKPT_DIR"
echo "  plot  → $PLOT_OUT"

for seed in $SEEDS; do
    run_one "sac_seed${seed}" \
        "$PYTHON" src/highway/run_sac.py \
            --seed "$seed" --total_timesteps "$SAC_TIMESTEPS" \
            --output "$CKPT_DIR/sac_seed${seed}.pt"

    run_one "uniform_n${N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/highway/run_uniform.py \
            --seed "$seed" --n_actions "$N_ACTIONS" \
            --total_timesteps "$DQN_TIMESTEPS" \
            --output "$CKPT_DIR/uniform_n${N_ACTIONS}_seed${seed}.pt"

    run_one "zooming_n${N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/highway/run_zooming.py \
            --seed "$seed" \
            --init_depth "$INIT_DEPTH" --n_actions "$N_ACTIONS" \
            --total_timesteps "$DQN_TIMESTEPS" \
            --output "$CKPT_DIR/zooming_n${N_ACTIONS}_seed${seed}.pt"
done

echo
echo "=== runs complete: ${ok}/${total} succeeded ==="
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed runs:"
    printf "  %s\n" "${failed[@]}"
fi

echo
echo "=== running compare.py ==="
"$PYTHON" src/highway/compare.py \
    --checkpoints-dir "$CKPT_DIR" \
    --output "$PLOT_OUT" \
    --title "Architectures — racetrack-v0"

echo
echo "=== running timestep sweep (uniform vs zooming, N=64, 600k steps) ==="
echo "    config hardcoded in src/highway/run_timestep_sweep.py"
"$PYTHON" src/highway/run_timestep_sweep.py --run

echo
echo "=== pipeline done ==="
