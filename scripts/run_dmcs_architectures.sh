#!/usr/bin/env bash
#
# DMCS architectures: SAC vs (factored) Uniform vs (factored) Zooming
# on a chosen task at a fixed action budget, several seeds.
#
# Outputs:
#   checkpoints/dmcs/<task>/architectures/<arm>_seed<S>.pt
#   plots/dmcs/<task>_architectures.png
#
# Usage:
#   ./scripts/run_dmcs_architectures.sh                  # cartpole-swingup (default)
#   ./scripts/run_dmcs_architectures.sh walker-walk
#   ./scripts/run_dmcs_architectures.sh cheetah-run
#   N_ACTIONS=32 ./scripts/run_dmcs_architectures.sh walker-walk
#
# Knobs (env vars):
#   SEEDS              space-separated list, default "42 43 44 45 46"
#   SAC_TIMESTEPS      per-task default (cartpole-swingup: 150000,
#                      walker-walk: 300000, cheetah-run: 500000)
#   DQN_TIMESTEPS      same per-task default (uniform + zooming)
#   N_ACTIONS          action budget per axis, default 32
#   INIT_DEPTH         zooming starting depth, default 1
#                      (= 2 bins per axis at start; max adaptive room)
#   PYTHON             interpreter, default "python"
#
# Per-run stdout is tee'd to logs/dmcs/<task>/architectures/<label>.log.
# A failing run is logged and the script continues.
#
set -uo pipefail

TASK="${1:-cartpole-swingup}"

# Per-task training budget. Cartpole saturates by ~150k; walker-walk
# wants ~300k at N=32; cheetah-run is harder and benefits from 500k.
case "$TASK" in
    cartpole-swingup) DEFAULT_TIMESTEPS=150000 ;;
    walker-walk)      DEFAULT_TIMESTEPS=300000 ;;
    cheetah-run)      DEFAULT_TIMESTEPS=500000 ;;
    *)                DEFAULT_TIMESTEPS=300000 ;;
esac

SEEDS="${SEEDS:-42 43 44 45 46}"
SAC_TIMESTEPS="${SAC_TIMESTEPS:-$DEFAULT_TIMESTEPS}"
DQN_TIMESTEPS="${DQN_TIMESTEPS:-$DEFAULT_TIMESTEPS}"
N_ACTIONS="${N_ACTIONS:-32}"
INIT_DEPTH="${INIT_DEPTH:-1}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CKPT_DIR="checkpoints/dmcs/${TASK}/architectures"
LOG_DIR="logs/dmcs/${TASK}/architectures"
PLOT_OUT="plots/dmcs/${TASK}_architectures.png"
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

echo "dmcs architectures"
echo "  task:           $TASK"
echo "  seeds:          $SEEDS"
echo "  sac timesteps:  $SAC_TIMESTEPS"
echo "  dqn timesteps:  $DQN_TIMESTEPS"
echo "  n_actions:      $N_ACTIONS"
echo "  init_depth:     $INIT_DEPTH"
echo "  python:         $PYTHON"
echo "  ckpts -> $CKPT_DIR"
echo "  plot  -> $PLOT_OUT"

for seed in $SEEDS; do
    run_one "sac_seed${seed}" \
        "$PYTHON" src/dmcs/run_sac.py \
            --task "$TASK" --seed "$seed" \
            --total_timesteps "$SAC_TIMESTEPS" \
            --output "$CKPT_DIR/sac_seed${seed}.pt"

    run_one "uniform_n${N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/dmcs/run_uniform.py \
            --task "$TASK" --seed "$seed" --n_actions "$N_ACTIONS" \
            --total_timesteps "$DQN_TIMESTEPS" \
            --output "$CKPT_DIR/uniform_n${N_ACTIONS}_seed${seed}.pt"

    run_one "zooming_n${N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/dmcs/run_zooming.py \
            --task "$TASK" --seed "$seed" \
            --init_depth "$INIT_DEPTH" --n_actions "$N_ACTIONS" \
            --total_timesteps "$DQN_TIMESTEPS" \
            --output "$CKPT_DIR/zooming_n${N_ACTIONS}_seed${seed}.pt"
done

echo
echo "=== running compare.py ==="
"$PYTHON" src/dmcs/compare.py \
    --task "$TASK" \
    --checkpoints-dir "$CKPT_DIR" \
    --n_actions "$N_ACTIONS" \
    --output "$PLOT_OUT" \
    --title "Architectures -- dm_control/${TASK}  (N=${N_ACTIONS})"

echo
echo "=== runs complete: ${ok}/${total} succeeded ==="
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed runs:"
    printf "  %s\n" "${failed[@]}"
fi

echo
echo "=== architectures done ==="
