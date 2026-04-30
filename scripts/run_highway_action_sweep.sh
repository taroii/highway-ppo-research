#!/usr/bin/env bash
#
# Highway action-budget sweep: uniform vs zooming on racetrack-v0
# across N \in {8, 16, 32, 64} at fixed training budget.  Thin wrapper
# around src/highway/run_action_sweep.py -- the sweep config (N values,
# seeds, timesteps) is hardcoded there.
#
# Outputs:
#   checkpoints/highway/action_sweep/<arm>_n{N}_seed<S>.pt
#   plots/highway/action_sweep.png
#
# Usage:
#   ./scripts/run_highway_action_sweep.sh
#
# Env vars:
#   PYTHON   default "python"
#
set -uo pipefail

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

exec "$PYTHON" src/highway/run_action_sweep.py --run
