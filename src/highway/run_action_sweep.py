r"""
Action-budget sweep: matched A/B between uniform and zooming across
increasing action counts at a fixed training budget.

The hypothesis: zooming's advantage over uniform grows as the action
budget grows, because uniform pays the full ``N``-output Q-learning cost
from step 1 while zooming bootstraps from a coarser grid (init=8) and
only refines to N where the policy concentrates plays.

Caveat -- this sweep does *not* control for compute: a 64-arm bandit
needs more samples to converge than an 8-arm bandit, so very large N
may look bad here purely because the training budget is fixed. The
companion ``run_timestep_sweep.py`` separates that confound by holding
N=64 and varying training timesteps.

Sweep dimensions:
  - n \in {8, 16, 32, 64} -- action budget (per axis; for racetrack
    da=1 this is total cells).  Zooming starts at ``2^init_depth``
    bins (init_depth = min(3, log2(n))) and refines up to ``n``.
  - seed \in {42, 43, 44} by default (extend SEEDS for robustness;
    cut to a single seed for a smoke test).

Outputs:
  - checkpoints/highway/action_sweep/<arm>_<config>_seed<S>.pt
  - logs/highway/action_sweep/<label>.log    (with --run)
  - plots/highway/action_sweep.png           (final-reward vs N curve)

This is a different experiment from run_highway_pipeline.sh: the
pipeline tests three architectures at a fixed N; this sweep tests how
the uniform-vs-zooming gap scales with N.

Usage:
    python src/highway/run_action_sweep.py            # print commands only
    python src/highway/run_action_sweep.py --run      # execute sequentially
    python src/highway/run_action_sweep.py --print | xargs -P 4 -I{} sh -c '{}'
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


N_VALUES = [8, 16, 32, 64]
SEEDS = [42, 43, 44]
TOTAL_TIMESTEPS = 150_000
PYTHON = sys.executable

CKPT_DIR = Path("checkpoints/highway/action_sweep")
LOG_DIR = Path("logs/highway/action_sweep")
PLOT_OUT = Path("plots/highway/action_sweep.png")


def commands() -> List[Tuple[str, List[str]]]:
    """Yield (label, argv-list) for every (arm, n, seed) cell in the sweep."""
    out: List[Tuple[str, List[str]]] = []
    for seed in SEEDS:
        for n in N_VALUES:
            label_u = f"uniform_n{n}_seed{seed}"
            out.append((
                label_u,
                [PYTHON, "src/highway/run_uniform.py",
                 "--seed", str(seed),
                 "--n_actions", str(n),
                 "--total_timesteps", str(TOTAL_TIMESTEPS),
                 "--output", str(CKPT_DIR / f"{label_u}.pt")],
            ))
            init_depth = min(3, int(math.log2(n)))
            label_z = f"zooming_n{n}_seed{seed}"
            out.append((
                label_z,
                [PYTHON, "src/highway/run_zooming.py",
                 "--seed", str(seed),
                 "--init_depth", str(init_depth),
                 "--n_actions", str(n),
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

    print(f"\n=== running compare_action_sweep.py ===")
    rc = subprocess.run(
        [PYTHON, "src/highway/compare_action_sweep.py",
         "--checkpoints-dir", str(CKPT_DIR),
         "--output", str(PLOT_OUT)],
    ).returncode

    print("\n=== sweep done ===")
    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true",
                   help="Execute each command sequentially. Default: print only.")
    args = p.parse_args()

    cmds = commands()
    print(f"# Action-budget sweep -- {len(cmds)} runs total "
          f"({len(N_VALUES)} action counts x {len(SEEDS)} seeds x 2 arms)")
    print(f"# Total timesteps per run: {TOTAL_TIMESTEPS}")
    print(f"# Outputs: {CKPT_DIR}/, plot: {PLOT_OUT}")

    if not args.run:
        print()
        _print_commands(cmds)
        return 0
    return _execute_with_logs(cmds)


if __name__ == "__main__":
    sys.exit(main())
