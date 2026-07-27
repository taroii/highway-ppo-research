#!/usr/bin/env bash
#
# Run the ENTIRE experiment suite (all tasks, all phases) in sequence with
# a single command and a configurable seed count. Wraps run_dmcs.sh and
# run_highway.sh.
#
#   scripts/run_all.sh --seeds 20
#   scripts/run_all.sh --seeds 50 --se-seeds 10
#   scripts/run_all.sh --seeds 20 --tasks walker-walk,highway --phases efficiency
#   scripts/run_all.sh --seeds 20 --dry-run          # preview the run list + counts
#
# Options:
#   --seeds N        number of seeds (default 3). Uses seeds 42, 43, ..., 42+N-1.
#   --se-seeds M     seeds for the sample-efficiency phase only (default = --seeds).
#                    These runs are 1M steps each and dominate wall-clock, so you
#                    will usually want M < N (e.g. --seeds 20 --se-seeds 5).
#   --phases LIST    subset of {efficiency,robustness,sample-efficiency} or "all"
#                    (default all). Comma- or space-separated.
#   --tasks LIST     subset of {cartpole-swingup,walker-walk,cheetah-run,highway}
#                    or "all" (default all). Comma- or space-separated.
#   --python PATH    interpreter (default "python").
#   --no-resume      re-run everything; by default finished checkpoints are SKIPPED
#                    so an interrupted run resumes where it left off.
#   --dry-run        print the plan and per-phase run counts, execute nothing.
#   -h, --help       this message.
#
# Ordering: phases are run OUTER, tasks INNER -- i.e. all efficiency runs first
# (across every task), then all robustness, then all sample-efficiency. This
# front-loads the headline efficiency results, so if you run out of time before
# the rebuilds you still have complete efficiency figures for every task.
#
# Resumability: safe to Ctrl-C and re-launch (or run under nohup/tmux). Existing
# checkpoints are skipped unless --no-resume is given. Per-run logs land under
# logs/<family>/<task>/<phase>/.
#
set -uo pipefail

SEEDS_N=3
SE_SEEDS_N=""            # default: equals SEEDS_N
PHASES="efficiency robustness sample-efficiency"
TASKS="cartpole-swingup walker-walk cheetah-run highway"
PYTHON="python"
RESUME=1
DRY_RUN=0

norm_list() { echo "${1//,/ }"; }   # commas -> spaces

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)     SEEDS_N="$2"; shift 2 ;;
        --se-seeds)  SE_SEEDS_N="$2"; shift 2 ;;
        --phases)    PHASES="$(norm_list "$2")"; shift 2 ;;
        --tasks)     TASKS="$(norm_list "$2")"; shift 2 ;;
        --python)    PYTHON="$2"; shift 2 ;;
        --no-resume) RESUME=0; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

[ "$PHASES" = "all" ] && PHASES="efficiency robustness sample-efficiency"
[ "$TASKS" = "all" ]  && TASKS="cartpole-swingup walker-walk cheetah-run highway"
[ -z "$SE_SEEDS_N" ] && SE_SEEDS_N="$SEEDS_N"

# Build seed lists: 42, 43, ..., 42+N-1
SEEDS_LIST="$(seq -s ' ' 42 $((42 + SEEDS_N - 1)))"
SE_SEEDS_LIST="$(seq -s ' ' 42 $((42 + SE_SEEDS_N - 1)))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# ---- estimate run counts (uses the sub-scripts' DEFAULT grid sizes) ----
# efficiency:  |N|(4) x seeds x 2 arms + seeds (SAC)          = 9 * S
# robustness:  (|BP|(2) + |ID|(1)) x seeds                    = 3 * S
# sample-eff:  |SE_N|(1) x se_seeds x 2 arms + se_seeds (SAC) = 3 * SE
n_tasks=0; for t in $TASKS; do n_tasks=$((n_tasks + 1)); done
tot=0
for ph in $PHASES; do
    case "$ph" in
        efficiency)        per=$((9 * SEEDS_N)) ;;
        robustness)        per=$((3 * SEEDS_N)) ;;
        sample-efficiency) per=$((3 * SE_SEEDS_N)) ;;
        *) echo "unknown phase '$ph'" >&2; exit 2 ;;
    esac
    tot=$((tot + per * n_tasks))
done

echo "=============================================================="
echo " run_all  |  $(date '+%F %T')"
echo "   seeds:      $SEEDS_N   ->  [$SEEDS_LIST]"
echo "   se-seeds:   $SE_SEEDS_N   ->  [$SE_SEEDS_LIST]   (sample-efficiency only, 1M steps each)"
echo "   phases:     $PHASES"
echo "   tasks:      $TASKS"
echo "   python:     $PYTHON      resume: $RESUME      dry-run: $DRY_RUN"
echo "   approx total training runs (default grids): ~$tot"
echo "   NOTE: sample-efficiency runs are 1M steps and dominate wall-clock;"
echo "         use --se-seeds < --seeds to keep them tractable."
echo "=============================================================="

start_all=$(date +%s)

for phase in $PHASES; do
    for task in $TASKS; do
        echo; echo ">>>>> phase=$phase  task=$task  [$(date '+%F %T')]"
        t0=$(date +%s)
        if [ "$task" = "highway" ]; then
            SEEDS="$SEEDS_LIST" SE_SEEDS="$SE_SEEDS_LIST" \
              SKIP_EXISTING="$RESUME" PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" \
              bash scripts/run_highway.sh "$phase"
        else
            SEEDS="$SEEDS_LIST" SE_SEEDS="$SE_SEEDS_LIST" \
              SKIP_EXISTING="$RESUME" PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" \
              bash scripts/run_dmcs.sh "$task" "$phase"
        fi
        echo ">>>>> phase=$phase task=$task done in $(( ($(date +%s) - t0) / 60 )) min"
    done
done

echo; echo "=============================================================="
echo " run_all COMPLETE in $(( ($(date +%s) - start_all) / 60 )) min  |  $(date '+%F %T')"
echo " plots refreshed under plots/dmcs/ and plots/highway/"
echo "=============================================================="
