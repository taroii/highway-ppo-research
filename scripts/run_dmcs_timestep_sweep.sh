#!/usr/bin/env bash
#
# DMCS timestep sweep: long-horizon learning curves for uniform vs
# zooming at a chosen action budget.
#
# Pick TS_N_ACTIONS by inspecting plots/dmcs/<task>_action_sweep.png
# (from scripts/run_dmcs_pipeline.sh) -- typically the largest N at
# which both arms still train, where the uniform-vs-zooming gap is
# expected to widen with more compute.
#
# Outputs:
#   checkpoints/dmcs/<task>/timestep_sweep/<arm>_n{N}_seed<S>.pt
#   plots/dmcs/<task>_timestep_sweep.png
#
# Usage:
#   TS_N_ACTIONS=32 ./scripts/run_dmcs_timestep_sweep.sh walker-walk
#   TS_N_ACTIONS=64 TS_TIMESTEPS=1000000 ./scripts/run_dmcs_timestep_sweep.sh cheetah-run
#
# Required env var:
#   TS_N_ACTIONS  bins per axis for both arms (no default -- must pick from
#                 the action sweep plot).
#
# Optional env vars:
#   TS_SEEDS      space-separated, default "42"
#   TS_TIMESTEPS  default 600000
#   INIT_DEPTH    zooming starting depth, default 1
#   PYTHON        default "python"
#
# Per-run stdout is tee'd to logs/dmcs/<task>/timestep_sweep/<label>.log.
# A failing run is logged and the script continues.
#
set -uo pipefail

TASK="${1:-cartpole-swingup}"

if [ -z "${TS_N_ACTIONS:-}" ]; then
    echo "Error: TS_N_ACTIONS is required.  Inspect plots/dmcs/${TASK}_action_sweep.png" >&2
    echo "       (produced by scripts/run_dmcs_pipeline.sh) and re-run with TS_N_ACTIONS=<N>." >&2
    exit 1
fi

TS_SEEDS="${TS_SEEDS:-42}"
TS_TIMESTEPS="${TS_TIMESTEPS:-600000}"
INIT_DEPTH="${INIT_DEPTH:-1}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

TS_CKPT_DIR="checkpoints/dmcs/${TASK}/timestep_sweep"
TS_LOG_DIR="logs/dmcs/${TASK}/timestep_sweep"
TS_PLOT_OUT="plots/dmcs/${TASK}_timestep_sweep.png"
mkdir -p "$TS_CKPT_DIR" "$TS_LOG_DIR" "$(dirname "$TS_PLOT_OUT")"

failed=()
total=0
ok=0

run_one() {
    local label="$1"; shift
    local log="$TS_LOG_DIR/${label}.log"
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

echo "dmcs timestep sweep"
echo "  task:           $TASK"
echo "  ts seeds:       $TS_SEEDS"
echo "  ts n_actions:   $TS_N_ACTIONS"
echo "  ts timesteps:   $TS_TIMESTEPS"
echo "  init_depth:     $INIT_DEPTH"
echo "  python:         $PYTHON"
echo "  ckpts -> $TS_CKPT_DIR"
echo "  plot  -> $TS_PLOT_OUT"

for seed in $TS_SEEDS; do
    run_one "uniform_n${TS_N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/dmcs/run_uniform.py \
            --task "$TASK" --seed "$seed" --n_actions "$TS_N_ACTIONS" \
            --total_timesteps "$TS_TIMESTEPS" \
            --output "$TS_CKPT_DIR/uniform_n${TS_N_ACTIONS}_seed${seed}.pt"

    run_one "zooming_n${TS_N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/dmcs/run_zooming.py \
            --task "$TASK" --seed "$seed" \
            --init_depth "$INIT_DEPTH" --n_actions "$TS_N_ACTIONS" \
            --total_timesteps "$TS_TIMESTEPS" \
            --output "$TS_CKPT_DIR/zooming_n${TS_N_ACTIONS}_seed${seed}.pt"
done

echo
echo "=== running compare.py ==="
"$PYTHON" src/dmcs/compare.py \
    --task "$TASK" \
    --checkpoints-dir "$TS_CKPT_DIR" \
    --n_actions "$TS_N_ACTIONS" \
    --output "$TS_PLOT_OUT" \
    --title "Timestep sweep -- dm_control/${TASK}  (N=${TS_N_ACTIONS})"

echo
echo "=== runs complete: ${ok}/${total} succeeded ==="
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed runs:"
    printf "  %s\n" "${failed[@]}"
fi

echo
echo "=== timestep sweep done ==="
