#!/usr/bin/env bash
#
# DMCS action-budget sweep: uniform vs zooming on a chosen task across
# N ∈ {8, 16, 32, 64} at fixed training budget.  Thin wrapper around
# src/dmcs/run_action_sweep.py — the sweep config (N values, seeds,
# timesteps) is hardcoded there.
#
# Outputs:
#   checkpoints/dmcs/<task>/action_sweep/<arm>_n{N}_seed<S>.pt
#   plots/dmcs/<task>_action_sweep.png
#
# Usage:
#   ./scripts/run_dmcs_action_sweep.sh                   # cartpole-swingup (default)
#   ./scripts/run_dmcs_action_sweep.sh walker-walk
#   ./scripts/run_dmcs_action_sweep.sh cheetah-run
#
# Env vars:
#   PYTHON   default "python"
#
set -uo pipefail

TASK="${1:-cartpole-swingup}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

exec "$PYTHON" src/dmcs/run_action_sweep.py --task "$TASK" --run
