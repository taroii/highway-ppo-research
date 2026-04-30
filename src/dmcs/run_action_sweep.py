r"""
Action-budget sweep on a DMCS task: matched A/B between uniform and
zooming across increasing action counts, at fixed training budget.

The hypothesis: zooming's advantage over uniform grows as the action
budget grows, because uniform pays the full ``n * da``-output Q-learning
cost from step 1 while zooming bootstraps from a coarser grid (init=2
bins per axis at ``init_depth=1``) and only refines where the policy
concentrates plays.

Caveat -- this sweep does *not* control for compute: a 64-arm bandit
needs more samples to converge than an 8-arm bandit, so very large N
may look bad here purely because the training budget is fixed.  The
companion timestep sweep (in ``run_dmcs_pipeline.sh``'s phase 2)
separates that confound by holding N and varying training timesteps.

Sweep dimensions:
  - n \in {8, 16, 32, 64} -- bins per action axis.  Total cells per arm =
    ``n * da``; for da=6 (walker, cheetah) N=64 means 384 total cells.
  - seed \in {42} by default (extend SEEDS for robustness).
  - task: passed via --task, default ``cartpole-swingup``.

Outputs (per task):
  - checkpoints/dmcs/<task>/action_sweep/<arm>_n{N}_seed<S>.pt
  - logs/dmcs/<task>/action_sweep/<label>.log    (with --run)
  - plots/dmcs/<task>_action_sweep.png           (final-reward vs N curve)

Usage:
    python src/dmcs/run_action_sweep.py                         # print only
    python src/dmcs/run_action_sweep.py --run                   # execute
    python src/dmcs/run_action_sweep.py --task walker-walk --run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


N_VALUES = [8, 16, 32, 64]
SEEDS = [42]
TOTAL_TIMESTEPS = 300_000
INIT_DEPTH = 1
PYTHON = sys.executable


def commands(task: str, ckpt_dir: Path) -> List[Tuple[str, List[str]]]:
    """Yield (label, argv-list) for every (arm, n, seed) cell in the sweep."""
    out: List[Tuple[str, List[str]]] = []
    for seed in SEEDS:
        for n in N_VALUES:
            label_u = f"uniform_n{n}_seed{seed}"
            out.append((
                label_u,
                [PYTHON, "src/dmcs/run_uniform.py",
                 "--task", task,
                 "--seed", str(seed),
                 "--n_actions", str(n),
                 "--total_timesteps", str(TOTAL_TIMESTEPS),
                 "--output", str(ckpt_dir / f"{label_u}.pt")],
            ))
            label_z = f"zooming_n{n}_seed{seed}"
            out.append((
                label_z,
                [PYTHON, "src/dmcs/run_zooming.py",
                 "--task", task,
                 "--seed", str(seed),
                 "--init_depth", str(INIT_DEPTH),
                 "--n_actions", str(n),
                 "--total_timesteps", str(TOTAL_TIMESTEPS),
                 "--output", str(ckpt_dir / f"{label_z}.pt")],
            ))
    return out


def _print_commands(cmds: List[Tuple[str, List[str]]]) -> None:
    for _, argv in cmds:
        print(" ".join(shlex.quote(a) for a in argv))


def _execute_with_logs(cmds: List[Tuple[str, List[str]]],
                       task: str,
                       ckpt_dir: Path,
                       log_dir: Path,
                       plot_out: Path) -> int:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    failed: List[str] = []
    ok = 0
    for label, argv in cmds:
        log_path = log_dir / f"{label}.log"
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
        [PYTHON, "src/dmcs/compare_action_sweep.py",
         "--task", task,
         "--checkpoints-dir", str(ckpt_dir),
         "--output", str(plot_out)],
    ).returncode

    print("\n=== sweep done ===")
    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cartpole-swingup",
                   help="DMCS task slug.")
    p.add_argument("--run", action="store_true",
                   help="Execute each command sequentially. Default: print only.")
    args = p.parse_args()

    ckpt_dir = Path(f"checkpoints/dmcs/{args.task}/action_sweep")
    log_dir = Path(f"logs/dmcs/{args.task}/action_sweep")
    plot_out = Path(f"plots/dmcs/{args.task}_action_sweep.png")

    cmds = commands(args.task, ckpt_dir)
    print(f"# Action-budget sweep on dm_control/{args.task} -- "
          f"{len(cmds)} runs total "
          f"({len(N_VALUES)} action counts x {len(SEEDS)} seeds x 2 arms)")
    print(f"# Total timesteps per run: {TOTAL_TIMESTEPS}")
    print(f"# Outputs: {ckpt_dir}/, plot: {plot_out}")

    if not args.run:
        print()
        _print_commands(cmds)
        return 0
    return _execute_with_logs(cmds, args.task, ckpt_dir, log_dir, plot_out)


if __name__ == "__main__":
    sys.exit(main())
