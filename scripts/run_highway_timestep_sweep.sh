#!/usr/bin/env bash
#
# Highway timestep sweep: long-horizon learning curves for uniform vs
# zooming on racetrack-v0 at a chosen action budget.
#
# Pick TS_N_ACTIONS by inspecting plots/highway/action_sweep.png (from
# scripts/run_highway_pipeline.sh) — typically the largest N at which
# both arms still train, where the uniform-vs-zooming gap is expected
# to widen with more compute.
#
# Outputs:
#   checkpoints/highway/timestep_sweep/<arm>_n{N}_seed<S>.pt
#   plots/highway/timestep_sweep.png
#
# Usage:
#   TS_N_ACTIONS=64 ./scripts/run_highway_timestep_sweep.sh
#   TS_N_ACTIONS=32 TS_TIMESTEPS=1000000 ./scripts/run_highway_timestep_sweep.sh
#
# Required env var:
#   TS_N_ACTIONS  bins per axis for both arms (no default — must pick from
#                 the action sweep plot).
#
# Optional env vars:
#   TS_SEEDS      space-separated, default "42"
#   TS_TIMESTEPS  default 600000
#   INIT_DEPTH    zooming starting depth, default 3
#   PYTHON        default "python"
#
# Per-run stdout is tee'd to logs/highway/timestep_sweep/<label>.log.
# A failing run is logged and the script continues.
#
set -uo pipefail

if [ -z "${TS_N_ACTIONS:-}" ]; then
    echo "Error: TS_N_ACTIONS is required.  Inspect plots/highway/action_sweep.png" >&2
    echo "       (produced by scripts/run_highway_pipeline.sh) and re-run with TS_N_ACTIONS=<N>." >&2
    exit 1
fi

TS_SEEDS="${TS_SEEDS:-42}"
TS_TIMESTEPS="${TS_TIMESTEPS:-600000}"
INIT_DEPTH="${INIT_DEPTH:-3}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

TS_CKPT_DIR="checkpoints/highway/timestep_sweep"
TS_LOG_DIR="logs/highway/timestep_sweep"
TS_PLOT_OUT="plots/highway/timestep_sweep.png"
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

echo "highway timestep sweep"
echo "  ts seeds:       $TS_SEEDS"
echo "  ts n_actions:   $TS_N_ACTIONS"
echo "  ts timesteps:   $TS_TIMESTEPS"
echo "  init_depth:     $INIT_DEPTH"
echo "  python:         $PYTHON"
echo "  ckpts → $TS_CKPT_DIR"
echo "  plot  → $TS_PLOT_OUT"

for seed in $TS_SEEDS; do
    run_one "uniform_n${TS_N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/highway/run_uniform.py \
            --seed "$seed" --n_actions "$TS_N_ACTIONS" \
            --total_timesteps "$TS_TIMESTEPS" \
            --output "$TS_CKPT_DIR/uniform_n${TS_N_ACTIONS}_seed${seed}.pt"

    run_one "zooming_n${TS_N_ACTIONS}_seed${seed}" \
        "$PYTHON" src/highway/run_zooming.py \
            --seed "$seed" \
            --init_depth "$INIT_DEPTH" --n_actions "$TS_N_ACTIONS" \
            --total_timesteps "$TS_TIMESTEPS" \
            --output "$TS_CKPT_DIR/zooming_n${TS_N_ACTIONS}_seed${seed}.pt"
done

echo
echo "=== running compare_timestep_sweep.py ==="
"$PYTHON" src/highway/compare_timestep_sweep.py \
    --checkpoints-dir "$TS_CKPT_DIR" \
    --output "$TS_PLOT_OUT" \
    --title "Timestep sweep — racetrack-v0  (N=${TS_N_ACTIONS})"

echo
echo "=== runs complete: ${ok}/${total} succeeded ==="
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed runs:"
    printf "  %s\n" "${failed[@]}"
fi

echo
echo "=== timestep sweep done ==="
