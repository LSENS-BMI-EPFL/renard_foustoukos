"""Run the analysis pipeline and/or regenerate the figures.

Stages:
    pipeline  NWB files -> processed data (pipeline/NN_*.py, in order)
    figures   processed data -> figure panels, source data and statistics
              (figures/*/*.py)

Each step runs in its own process with a non-interactive matplotlib backend;
its output goes to <output_root>/logs/<step>.log. Paths come from config.yaml
(see config.example.yaml).

Examples:
    python run_all.py --list                    # show the steps
    python run_all.py --stage figures           # all figures from processed data
    python run_all.py --only figure_4 supp_4    # one or more figure folders
    python run_all.py --stage pipeline --from 06_decoder
    python run_all.py --stage all --skip 04_learning_curves
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# (name, script, extra arguments, run by default). Steps run in this order.
PIPELINE = [
    # Regenerates the stop-flag / trial-index YAMLs into <processed_dir>/stop_flags;
    # the published ones are part of the dataset, so this is off by default.
    ('01_session_flags', 'pipeline/01_session_flags.py', [], False),
    # Writes tensors to <processed_dir>/mice; point tensor_dir there to use them.
    ('02_tensors', 'pipeline/02_tensors.py', [], True),
    ('03_behavior_tables', 'pipeline/03_behavior_tables.py', [], True),
    ('04_learning_curves', 'pipeline/04_learning_curves.py', [], True),  # ~3 h
    ('05_lmi', 'pipeline/05_lmi.py', [], True),
    ('06_decoder', 'pipeline/06_decoder.py', [], True),
    ('07_reactivations', 'pipeline/07_reactivations.py', [], True),
    ('07_reactivations_nolick', 'pipeline/07_reactivations.py', ['--nolick'], True),
    ('08_participation', 'pipeline/08_participation.py', [], True),
    ('09_pairwise_correlations', 'pipeline/09_pairwise_correlations.py', [], True),
]

FIGURE_DIRS = [
    'figure_1',
    'supp_1',
    'figure_2',
    'figure_3',
    'supp_2',
    'supp_3',
    'figure_4',
    'supp_4',
    'revisions',
]
# Revision analyses with a slow computation of their own; --recompute forces it.
RECOMPUTE = {'figure_3f_LMIshuffles', 'figure_4h_shuffle_control'}


def figure_steps():
    steps = []
    for d in FIGURE_DIRS:
        for script in sorted((REPO / 'figures' / d).glob('*.py')):
            steps.append((script.stem, str(script.relative_to(REPO)), [], True, d))
    return steps


def select(steps, args):
    """Filter steps by --only / --skip / --from and the default flag."""
    names = [s[0] for s in steps]
    if args.from_step:
        if args.from_step not in names:
            sys.exit(f"--from: unknown step '{args.from_step}'")
        steps = steps[names.index(args.from_step) :]
    out = []
    for step in steps:
        name, group = step[0], (step[4] if len(step) > 4 else None)
        if args.only:
            if name not in args.only and group not in args.only:
                continue
        elif not step[3]:
            continue
        if name in args.skip or group in args.skip:
            continue
        out.append(step)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n', 1)[1],
    )
    parser.add_argument('--stage', choices=['pipeline', 'figures', 'all'], default='all')
    parser.add_argument(
        '--only',
        nargs='+',
        default=[],
        metavar='NAME',
        help='steps or figure folders to run (includes steps off by default)',
    )
    parser.add_argument(
        '--skip', nargs='+', default=[], metavar='NAME', help='steps or figure folders to leave out'
    )
    parser.add_argument(
        '--from', dest='from_step', metavar='NAME', help='start at this step (within the selected stage)'
    )
    parser.add_argument(
        '--recompute', action='store_true', help='force the slow computations of the revision analyses'
    )
    parser.add_argument('--keep-going', action='store_true', help='continue after a failed step')
    parser.add_argument('--list', action='store_true', help='list the selected steps and exit')
    parser.add_argument('--dry-run', action='store_true', help='print the commands only')
    args = parser.parse_args()

    steps = []
    if args.stage in ('pipeline', 'all'):
        steps += select(PIPELINE, args)
    if args.stage in ('figures', 'all'):
        steps += select(figure_steps(), args)
    if not steps:
        sys.exit('No step selected (see --list).')

    if args.list:
        for step in steps:
            print(f"{step[0]:32s} {step[1]} {' '.join(step[2])}")
        return

    from fast_learning import paths  # reads config.yaml

    log_dir = Path(paths.output_root) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg')

    failed = []
    t_start = time.time()
    for i, (name, script, extra, *_rest) in enumerate(steps, 1):
        cmd = [sys.executable, script, *extra]
        if args.recompute and name in RECOMPUTE:
            cmd.append('--recompute')
        print(f"[{i}/{len(steps)}] {name:32s}", end=' ', flush=True)
        if args.dry_run:
            print(' '.join(cmd))
            continue
        t0 = time.time()
        with open(log_dir / f'{name}.log', 'w') as log:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'ok' if rc == 0 else f'FAILED (exit {rc})'}  {time.time() - t0:.0f} s")
        if rc != 0:
            failed.append(name)
            print(f"    see {log_dir / (name + '.log')}")
            if not args.keep_going:
                break

    if not args.dry_run:
        print(
            f"\nDone in {(time.time() - t_start) / 60:.1f} min; "
            f"{'all steps succeeded' if not failed else 'failed: ' + ', '.join(failed)}. Logs: {log_dir}"
        )
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
