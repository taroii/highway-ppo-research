#!/usr/bin/env bash
#
# DMCS action-budget sweep: uniform vs zooming on a chosen task across
# N \in {16, 32, 64, 128} at fixed training budget.  Thin wrapper around
# src/dmcs/run_action_sweep.py -- the sweep grid (N values, seeds) is
# hardcoded there.  Per-task default timesteps mirror the architectures
# script so all phases of the pipeline see the same per-task budget.
#
# Outputs:
#   checkpoints/dmcs/<task>/action_sweep/<arm>_n{N}_seed<S>.pt
#   plots/dmcs/<task>_action_sweep.png
#
# Usage:
#   ./scripts/run_dmcs_action_sweep.sh                   # cartpole-swingup (default)
#   ./scripts/run_dmcs_action_sweep.sh walker-walk
#   ./scripts/run_dmcs_action_sweep.sh cheetah-run
#   DQN_TIMESTEPS=200000 ./scripts/run_dmcs_action_sweep.sh cartpole-swingup
#
# Env vars:
#   DQN_TIMESTEPS   per-task default (cartpole-swingup: 150000,
#                   walker-walk: 300000, cheetah-run: 500000)
#   PYTHON          default "python"
#
set -uo pipefail

TASK="${1:-cartpole-swingup}"
PYTHON="${PYTHON:-python}"

case "$TASK" in
    cartpole-swingup) DEFAULT_TIMESTEPS=150000 ;;
    walker-walk)      DEFAULT_TIMESTEPS=300000 ;;
    cheetah-run)      DEFAULT_TIMESTEPS=500000 ;;
    *)                DEFAULT_TIMESTEPS=300000 ;;
esac
DQN_TIMESTEPS="${DQN_TIMESTEPS:-$DEFAULT_TIMESTEPS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

exec "$PYTHON" src/dmcs/run_action_sweep.py \
    --task "$TASK" \
    --total_timesteps "$DQN_TIMESTEPS" \
    --run
