r"""
Timestep sweep: at a fixed (large) action budget, run uniform and
zooming for a long training horizon and plot the full learning curves.

Why this experiment exists -- the action-budget sweep
(``run_action_sweep.py``) holds training timesteps fixed and shows
both arms degrading at large N.  That confounds two effects:

  1. *Sample-efficiency:* a 64-arm bandit needs more samples than a
     16-arm one, so absolute reward at fixed budget can drop just from
     undertraining, regardless of architecture.
  2. *Asymptotic quality:* zooming concentrates plays where the policy
     wants resolution, while uniform spreads them across all 64 bins.

This sweep separates them: at large N (default 64), we hand both arms
a *long* training budget and read the convergence rate from the
learning curve.  Zooming should pull ahead earlier and may also reach
a higher asymptote if uniform's exploration is the bottleneck.

Why a learning-curve plot, not a discrete sweep over total_timesteps:
each point on a single long run *is* the same as a separately trained
shorter run (no LR schedule depends on the total budget in our DQN).
One long run per seed gives the entire timestep-sweep curve for free.

Sweep dimensions:
  - n = 64 (configurable via N_ACTIONS).  Zooming starts at
    ``2^init_depth = 8`` bins (init_depth=3) and refines up to ``n``.
  - seed \in {42, 43, 44} by default; extend SEEDS for robustness,
    or cut to a single seed for a smoke test.
  - total_timesteps default 600_000 (~ 4x the architectures pipeline).

Outputs:
  - checkpoints/highway/timestep_sweep/<arm>_<config>_seed<S>.pt
  - logs/highway/timestep_sweep/<label>.log    (with --run)
  - plots/highway/timestep_sweep.png           (learning curves)

Usage:
    python src/highway/run_timestep_sweep.py            # print commands only
    python src/highway/run_timestep_sweep.py --run      # execute sequentially
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


N_ACTIONS = 64
SEEDS = [42, 43, 44]
TOTAL_TIMESTEPS = 600_000
PYTHON = sys.executable

CKPT_DIR = Path("checkpoints/highway/timestep_sweep")
LOG_DIR = Path("logs/highway/timestep_sweep")
PLOT_OUT = Path("plots/highway/timestep_sweep.png")


def commands() -> List[Tuple[str, List[str]]]:
    """Yield (label, argv-list) for every (arm, seed) cell."""
    out: List[Tuple[str, List[str]]] = []
    init_depth = min(3, int(math.log2(N_ACTIONS)))
    for seed in SEEDS:
        label_u = f"uniform_n{N_ACTIONS}_seed{seed}"
        out.append((
            label_u,
            [PYTHON, "src/highway/run_uniform.py",
             "--seed", str(seed),
             "--n_actions", str(N_ACTIONS),
             "--total_timesteps", str(TOTAL_TIMESTEPS),
             "--output", str(CKPT_DIR / f"{label_u}.pt")],
        ))
        label_z = f"zooming_n{N_ACTIONS}_seed{seed}"
        out.append((
            label_z,
            [PYTHON, "src/highway/run_zooming.py",
             "--seed", str(seed),
             "--init_depth", str(init_depth),
             "--n_actions", str(N_ACTIONS),
             "--total_timesteps", str(TOTAL_TIMESTEPS),
             "--output", str(CKPT_DIR / f"{label_z}.pt")],
        ))
    return out


def _print_commands(cmds: List[Tuple[str, List[str]]]) -> None:
    for _, argv in cmds:
        print(" ".join(shlex.quote(a) for a in argv))


def _execute_with_logs(cmds: List[Tuple[str, List[str]]]) -> int:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    failed: List[str] = []
    ok = 0
    for label, argv in cmds:
        log_path = LOG_DIR / f"{label}.log"
        line = " ".join(shlex.quote(a) for a in argv)
        print(f"\n=== START  {label}")
        print(f"    cmd: {line}")
        print(f"    log: {log_path}")
        try:
            with open(log_path, "wb") as logf:
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                assert proc.stdout is not None
                for chunk in iter(lambda: proc.stdout.read(4096), b""):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    logf.write(chunk)
                rc = proc.wait()
            if rc == 0:
                print(f"=== OK     {label}")
                ok += 1
            else:
                print(f"=== FAIL   {label} (rc={rc}, continuing)", file=sys.stderr)
                failed.append(label)
        except Exception as e:
            print(f"=== FAIL   {label} (exception: {e}, continuing)",
                  file=sys.stderr)
            failed.append(label)

    print(f"\n=== runs complete: {ok}/{len(cmds)} succeeded ===")
    if failed:
        print("failed runs:")
        for f in failed:
            print(f"  {f}")

    print(f"\n=== running compare_timestep_sweep.py ===")
    rc = subprocess.run(
        [PYTHON, "src/highway/compare_timestep_sweep.py",
         "--checkpoints-dir", str(CKPT_DIR),
         "--output", str(PLOT_OUT),
         "--title", f"Timestep sweep -- racetrack-v0 (N={N_ACTIONS})"],
    ).returncode

    print("\n=== sweep done ===")
    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true",
                   help="Execute each command sequentially. Default: print only.")
    args = p.parse_args()

    cmds = commands()
    print(f"# Timestep sweep -- {len(cmds)} runs total "
          f"(N={N_ACTIONS}, {len(SEEDS)} seeds x 2 arms)")
    print(f"# Total timesteps per run: {TOTAL_TIMESTEPS}")
    print(f"# Outputs: {CKPT_DIR}/, plot: {PLOT_OUT}")

    if not args.run:
        print()
        _print_commands(cmds)
        return 0
    return _execute_with_logs(cmds)


if __name__ == "__main__":
    sys.exit(main())
