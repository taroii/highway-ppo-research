#!/usr/bin/env bash
#
# DMCS experiment driver (new framing).  ONE script per task; a phase
# selector picks which experiment to run.  Everything reads off the
# deterministic-eval curves persisted in each checkpoint.
#
#   scripts/run_dmcs.sh <task> [phase]
#
# phase (default: efficiency):
#   efficiency         Headline figure.  uniform + zooming across N, plus
#                      SAC as a flat ceiling band.  Merges the old
#                      action-sweep + architectures phases into one
#                      performance-vs-N plot.
#   robustness         Auto-tuning claim.  Adds zooming knob variants
#                      (buffer_period, init_depth) at REF_N and compares
#                      the spread of zooming-across-knobs to uniform-across-N.
#                      REUSES the efficiency checkpoints (needs efficiency
#                      to have run first).
#   sample-efficiency  Repurposed timestep sweep.  Long (1M-step) runs at
#                      1-2 diagnostic N for uniform + zooming + SAC; the
#                      plot is reward vs env-steps (eval curve over steps).
#   all                efficiency -> robustness -> sample-efficiency.
#
# Task roles (planning decision -- change via env vars if you disagree):
#   cartpole-swingup, walker-walk : the "discretization works here" tasks;
#                                   worth the full efficiency + robustness story.
#   cheetah-run                    : boundary case (action dim 6); SAC is
#                                   expected to win.  Default phase is
#                                   efficiency-only -- one honest figure, not
#                                   the full machinery.
#
# Recommended workflow: run ONE task's `efficiency` end-to-end, eyeball the
# curve, THEN fan out.  Don't launch every task x phase on faith after a refactor.
#
# Common env overrides:
#   SEEDS            efficiency/robustness seeds       (default "42 43 44")
#   N_VALUES         action budgets to sweep           (default "16 32 64 128")
#   REF_N            reference N for robustness knobs   (default 32)
#   BUFFER_PERIODS   zooming buffer_period grid         (default "8 16 32")
#   INIT_DEPTHS      zooming init_depth grid            (default "1 2")
#   SE_N             sample-efficiency N values (1-2)   (default "32")
#   SE_SEEDS         sample-efficiency seeds            (default "42 43 44")
#   DQN_TIMESTEPS / SAC_TIMESTEPS   per-task defaults below
#   SE_TIMESTEPS     sample-efficiency budget           (default 1000000)
#   INIT_DEPTH       zooming default depth              (default 1)
#   DRY_RUN=1        print commands, don't execute
#   PYTHON           interpreter                        (default "python")
#
set -uo pipefail

TASK="${1:-cartpole-swingup}"
PHASE="${2:-}"

case "$TASK" in
    cartpole-swingup) DEFAULT_TS=150000 ;;
    walker-walk)      DEFAULT_TS=300000 ;;
    cheetah-run)      DEFAULT_TS=500000 ;;
    *)                DEFAULT_TS=300000 ;;
esac
# cheetah is the boundary case: default to the single efficiency figure.
if [ -z "$PHASE" ]; then
    case "$TASK" in
        cheetah-run) PHASE=efficiency ;;
        *)           PHASE=efficiency ;;
    esac
fi

SEEDS="${SEEDS:-42 43 44}"
N_VALUES="${N_VALUES:-16 32 64 128}"
REF_N="${REF_N:-32}"
# Knob grids EXCLUDE the center (bp=16, init_depth=1): that config is the
# efficiency zooming_n${REF_N} run, which plot_robustness already pools in.
BUFFER_PERIODS="${BUFFER_PERIODS:-8 32}"
INIT_DEPTHS="${INIT_DEPTHS:-2}"
SE_N="${SE_N:-32}"
SE_SEEDS="${SE_SEEDS:-42 43 44}"
DQN_TIMESTEPS="${DQN_TIMESTEPS:-$DEFAULT_TS}"
SAC_TIMESTEPS="${SAC_TIMESTEPS:-$DEFAULT_TS}"
SE_TIMESTEPS="${SE_TIMESTEPS:-1000000}"
INIT_DEPTH="${INIT_DEPTH:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CK="checkpoints/dmcs/${TASK}"
LG="logs/dmcs/${TASK}"
PL="plots/dmcs"
mkdir -p "$PL"

failed=(); total=0; ok=0

run_one() {   # run_one <log-subdir> <label> <cmd...>
    local sub="$1" label="$2"; shift 2
    local logdir="$LG/$sub"; mkdir -p "$logdir"
    local log="$logdir/${label}.log"
    total=$((total + 1))
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
        "$PYTHON" src/dmcs/run_zooming.py --task "$TASK" --seed "$4" \
        --init_depth "$6" --n_actions "$3" --buffer_period "$7" \
        --total_timesteps "$5" --output "$1/$2.pt"
}
unif() {  # unif <outdir> <label> <n_actions> <seed> <timesteps>
    run_one "$(basename "$1")" "$2" \
        "$PYTHON" src/dmcs/run_uniform.py --task "$TASK" --seed "$4" \
        --n_actions "$3" --total_timesteps "$5" --output "$1/$2.pt"
}
sac() {   # sac <outdir> <label> <seed> <timesteps>
    run_one "$(basename "$1")" "$2" \
        "$PYTHON" src/dmcs/run_sac.py --task "$TASK" --seed "$3" \
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
        --checkpoints-dir "$d" --output "$PL/${TASK}_efficiency.png" \
        --title "Efficiency -- dm_control/${TASK}"
}

phase_robustness() {
    local d="$CK/robustness"; mkdir -p "$d"
    echo "### robustness: zooming knob grid at N=$REF_N (bp {$BUFFER_PERIODS}, id {$INIT_DEPTHS})"
    echo "    (uniform-over-N + zooming-over-N reused from efficiency/)"
    for bp in $BUFFER_PERIODS; do for s in $SEEDS; do
        zoom "$d" "zooming_bp${bp}_seed${s}" "$REF_N" "$s" "$DQN_TIMESTEPS" "$INIT_DEPTH" "$bp"
    done; done
    for id in $INIT_DEPTHS; do for s in $SEEDS; do
        zoom "$d" "zooming_id${id}_seed${s}" "$REF_N" "$s" "$DQN_TIMESTEPS" "$id" 16
    done; done
    [ "$DRY_RUN" = "1" ] || "$PYTHON" src/highway/plot_robustness.py \
        --dir "$CK/efficiency" --dir "$d" \
        --output "$PL/${TASK}_robustness.png" \
        --title "Robustness / auto-tuning -- dm_control/${TASK}"
}

phase_sample_efficiency() {
    local d="$CK/sample_efficiency"; mkdir -p "$d"
    echo "### sample-efficiency: N in {$SE_N}, seeds {$SE_SEEDS}, ${SE_TIMESTEPS} steps, +SAC"
    for n in $SE_N; do for s in $SE_SEEDS; do
        unif "$d" "uniform_n${n}_seed${s}" "$n" "$s" "$SE_TIMESTEPS"
        zoom "$d" "zooming_n${n}_seed${s}" "$n" "$s" "$SE_TIMESTEPS" "$INIT_DEPTH" 16
    done; done
    for s in $SE_SEEDS; do sac "$d" "sac_seed${s}" "$s" "$SE_TIMESTEPS"; done
    [ "$DRY_RUN" = "1" ] || "$PYTHON" src/dmcs/compare.py --task "$TASK" \
        --checkpoints-dir "$d" --output "$PL/${TASK}_sample_efficiency.png" \
        --title "Sample efficiency -- dm_control/${TASK}"
}

echo "dmcs driver | task=$TASK phase=$PHASE python=$PYTHON dry_run=$DRY_RUN"
case "$PHASE" in
    efficiency)         phase_efficiency ;;
    robustness)         phase_robustness ;;
    sample-efficiency)  phase_sample_efficiency ;;
    all)                phase_efficiency; phase_robustness; phase_sample_efficiency ;;
    *) echo "unknown phase '$PHASE' (efficiency|robustness|sample-efficiency|all)" >&2; exit 2 ;;
esac

echo; echo "=== $PHASE done: ${ok}/${total} runs ok ==="
if [ "${#failed[@]}" -gt 0 ]; then echo "failed:"; printf "  %s\n" "${failed[@]}"; fi
