#!/usr/bin/env bash
#
# Highway (racetrack-v0) experiment driver (new framing).  ONE script, a
# phase selector picks the experiment.  Everything reads off the
# deterministic-eval curves persisted in each checkpoint.
#
#   scripts/run_highway.sh [phase]
#
# phase (default: efficiency):
#   efficiency         Headline.  uniform + zooming across N + SAC ceiling
#                      band (merges old action-sweep + architectures).
#   robustness         Auto-tuning claim.  zooming knob variants
#                      (buffer_period, init_depth) at REF_N vs uniform-across-N
#                      spread.  Reuses the efficiency checkpoints.
#   sample-efficiency  Repurposed timestep sweep.  Long runs at 1-2 N for
#                      uniform + zooming + SAC; reward vs env-steps.
#   all                efficiency -> robustness -> sample-efficiency.
#
# Note: racetrack is da=1 (steering only), so highway is a supporting
# result -- the multi-axis story lives in DMCS.  It IS the clean matched
# arm now (both arms constant-c UCB, gamma 0.99), so it's a good sanity
# task to run `efficiency` on first.
#
# Common env overrides:
#   SEEDS "42 43 44" | N_VALUES "8 16 32 64" | REF_N 16
#   BUFFER_PERIODS "8 16 32" | INIT_DEPTHS "2 3" | INIT_DEPTH 3
#   SE_N "32" | SE_SEEDS "42 43 44" | SE_TIMESTEPS 600000
#   DQN_TIMESTEPS 150000 | SAC_TIMESTEPS 150000
#   DRY_RUN=1 | PYTHON "python"
#
set -uo pipefail

PHASE="${1:-efficiency}"

SEEDS="${SEEDS:-42 43 44}"
N_VALUES="${N_VALUES:-8 16 32 64}"
REF_N="${REF_N:-16}"
# Knob grids EXCLUDE the center (bp=16, init_depth=3): that config is the
# efficiency zooming_n${REF_N} run, which plot_robustness already pools in.
BUFFER_PERIODS="${BUFFER_PERIODS:-8 32}"
INIT_DEPTHS="${INIT_DEPTHS:-2}"
SE_N="${SE_N:-32}"
SE_SEEDS="${SE_SEEDS:-42 43 44}"
DQN_TIMESTEPS="${DQN_TIMESTEPS:-150000}"
SAC_TIMESTEPS="${SAC_TIMESTEPS:-150000}"
SE_TIMESTEPS="${SE_TIMESTEPS:-600000}"
INIT_DEPTH="${INIT_DEPTH:-3}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CK="checkpoints/highway"
LG="logs/highway"
PL="plots/highway"
mkdir -p "$PL"

failed=(); total=0; ok=0; skipped=0
SKIP_EXISTING="${SKIP_EXISTING:-0}"

run_one() {   # run_one <log-subdir> <label> <cmd...>
    local sub="$1" label="$2"; shift 2
    local logdir="$LG/$sub"; mkdir -p "$logdir"
    local log="$logdir/${label}.log"
    # extract the --output <path> from the command (for resume/skip)
    local outpath="" prev=""
    for a in "$@"; do [ "$prev" = "--output" ] && outpath="$a"; prev="$a"; done
    total=$((total + 1))
    if [ "$SKIP_EXISTING" = "1" ] && [ -n "$outpath" ] && [ -s "$outpath" ]; then
        skipped=$((skipped + 1)); ok=$((ok + 1))
        echo "=== SKIP   $sub/$label (checkpoint exists)"; return
    fi
    echo; echo "=== [$(date '+%F %T')] START  $sub/$label"
    echo "    cmd: $*"
    if [ "$DRY_RUN" = "1" ]; then echo "    (dry-run)"; ok=$((ok + 1)); return; fi
    if "$@" 2>&1 | tee "$log"; then
        ok=$((ok + 1)); echo "=== OK     $sub/$label"
    else
        failed+=("$sub/$label"); echo "=== FAIL   $sub/$label (continuing) ===" >&2
    fi
}

zoom() {  # zoom <outdir> <label> <n_actions> <seed> <timesteps> <init_depth> <buffer_period>
    run_one "$(basename "$1")" "$2" \
        "$PYTHON" src/highway/run_zooming.py --seed "$4" \
        --init_depth "$6" --n_actions "$3" --buffer_period "$7" \
        --total_timesteps "$5" --output "$1/$2.pt"
}
unif() {  # unif <outdir> <label> <n_actions> <seed> <timesteps>
    run_one "$(basename "$1")" "$2" \
        "$PYTHON" src/highway/run_uniform.py --seed "$4" \
        --n_actions "$3" --total_timesteps "$5" --output "$1/$2.pt"
}
sac() {   # sac <outdir> <label> <seed> <timesteps>
    run_one "$(basename "$1")" "$2" \
        "$PYTHON" src/highway/run_sac.py --seed "$3" \
        --total_timesteps "$4" --output "$1/$2.pt"
}

phase_efficiency() {
    local d="$CK/efficiency"; mkdir -p "$d"
    echo "### efficiency: N in {$N_VALUES}, seeds {$SEEDS}, +SAC ceiling"
    for n in $N_VALUES; do for s in $SEEDS; do
        unif "$d" "uniform_n${n}_seed${s}" "$n" "$s" "$DQN_TIMESTEPS"
        zoom "$d" "zooming_n${n}_seed${s}" "$n" "$s" "$DQN_TIMESTEPS" "$INIT_DEPTH" 16
    done; done
    for s in $SEEDS; do sac "$d" "sac_seed${s}" "$s" "$SAC_TIMESTEPS"; done
    [ "$DRY_RUN" = "1" ] || "$PYTHON" src/highway/plot_efficiency.py \
        --checkpoints-dir "$d" --output "$PL/efficiency.png" \
        --title "Efficiency -- racetrack-v0"
}

phase_robustness() {
    local d="$CK/robustness"; mkdir -p "$d"
    echo "### robustness: zooming knob grid at N=$REF_N (bp {$BUFFER_PERIODS}, id {$INIT_DEPTHS})"
    for bp in $BUFFER_PERIODS; do for s in $SEEDS; do
        zoom "$d" "zooming_bp${bp}_seed${s}" "$REF_N" "$s" "$DQN_TIMESTEPS" "$INIT_DEPTH" "$bp"
    done; done
    for id in $INIT_DEPTHS; do for s in $SEEDS; do
        zoom "$d" "zooming_id${id}_seed${s}" "$REF_N" "$s" "$DQN_TIMESTEPS" "$id" 16
    done; done
    [ "$DRY_RUN" = "1" ] || "$PYTHON" src/highway/plot_robustness.py \
        --dir "$CK/efficiency" --dir "$d" \
        --output "$PL/robustness.png" \
        --title "Robustness / auto-tuning -- racetrack-v0"
}

phase_sample_efficiency() {
    local d="$CK/sample_efficiency"; mkdir -p "$d"
    echo "### sample-efficiency: N in {$SE_N}, seeds {$SE_SEEDS}, ${SE_TIMESTEPS} steps, +SAC"
    for n in $SE_N; do for s in $SE_SEEDS; do
        unif "$d" "uniform_n${n}_seed${s}" "$n" "$s" "$SE_TIMESTEPS"
        zoom "$d" "zooming_n${n}_seed${s}" "$n" "$s" "$SE_TIMESTEPS" "$INIT_DEPTH" 16
    done; done
    for s in $SE_SEEDS; do sac "$d" "sac_seed${s}" "$s" "$SE_TIMESTEPS"; done
    [ "$DRY_RUN" = "1" ] || "$PYTHON" src/highway/compare.py \
        --checkpoints-dir "$d" --output "$PL/sample_efficiency.png" \
        --title "Sample efficiency -- racetrack-v0"
}

echo "highway driver | phase=$PHASE python=$PYTHON dry_run=$DRY_RUN"
case "$PHASE" in
    efficiency)         phase_efficiency ;;
    robustness)         phase_robustness ;;
    sample-efficiency)  phase_sample_efficiency ;;
    all)                phase_efficiency; phase_robustness; phase_sample_efficiency ;;
    *) echo "unknown phase '$PHASE' (efficiency|robustness|sample-efficiency|all)" >&2; exit 2 ;;
esac

echo; echo "=== $PHASE done: ${ok}/${total} runs ok (${skipped} skipped as already-present) ==="
if [ "${#failed[@]}" -gt 0 ]; then echo "failed:"; printf "  %s\n" "${failed[@]}"; fi
