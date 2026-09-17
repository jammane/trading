#!/usr/bin/env python3
"""
check_determinism.py — does --seed actually give reproducible runs?

Runs the C++ trainer twice with an identical --seed and compares
`training_log.csv` cell by cell. Answers, programmatically and repeatably, the
question that otherwise gets settled by eyeballing a diff:

    can an A/B comparison distinguish a code change from run-to-run noise?

If the answer is no, every A/B result in this project is uninterpretable —
including the way we would validate a refactor of the scoring path, and the
control-vs-trained comparison itself.

Compares the CSV rather than the log because the log is written by N worker
threads with no ordering guarantee, so its line ORDER varies even when every
number is identical. That is a false positive, and it is exactly what made the
first hand-run comparison look like a regression.

Why the default day range spans day 25: MT1 activates at `actual_day >= 25`
(MT1_START_DAY), and shared MT1 state across workers is the prime suspect for
any nondeterminism. A range that stops short of it would report a clean PASS
and prove nothing.

Usage:
  python check_determinism.py --binary ./build/training_v4_cpp
  python check_determinism.py --binary ./build/training_v4_cpp --workers 1
  python check_determinism.py --binary ./build/training_v4_cpp --start-day 16 --stop-day 34

Exit code 0 = deterministic, 1 = not, 2 = a run failed.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile


def run_once(binary, out_dir, start_day, stop_day, seed, workers, cwd):
    """One trainer run. Returns the path to its training_log.csv."""
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(out_dir + '.nosave', ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        binary,
        '--output', out_dir,
        '--start-day', str(start_day),
        '--stop-day', str(stop_day),
        '--passes', '1',
        '--seed', str(seed),
        '--workers', str(workers),
        '--no-save',
    ]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'  RUN FAILED (exit {proc.returncode}):')
        print('  ' + '\n  '.join((proc.stdout + proc.stderr).strip().split('\n')[-12:]))
        return None
    # --no-save redirects the MODEL dir to <out>.nosave and deletes it at exit, but log_dir is
    # fixed before that redirect, so the CSV survives in out_dir.
    path = os.path.join(out_dir, 'training_log.csv')
    return path if os.path.exists(path) else None


def load_rows(path):
    with open(path) as f:
        return list(csv.reader(f))


def compare(path_a, path_b):
    """Returns (identical, list of (row_idx, day, column_name, a, b))."""
    a, b = load_rows(path_a), load_rows(path_b)
    if not a or not b:
        return False, [(0, '?', '<empty csv>', str(len(a)), str(len(b)))]
    if a[0] != b[0]:
        return False, [(0, '?', '<header>', ','.join(a[0])[:60], ','.join(b[0])[:60])]
    header = a[0]
    diffs = []
    if len(a) != len(b):
        diffs.append((0, '?', '<row count>', str(len(a) - 1), str(len(b) - 1)))
    for i in range(1, min(len(a), len(b))):
        ra, rb = a[i], b[i]
        day = ra[1] if len(ra) > 1 else '?'
        for j in range(min(len(ra), len(rb))):
            if ra[j].strip() != rb[j].strip():
                col = header[j] if j < len(header) else f'col{j}'
                diffs.append((i, day, col, ra[j].strip(), rb[j].strip()))
    return len(diffs) == 0, diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', default='./build/training_v4_cpp')
    ap.add_argument('--cwd', default='.', help='run from here (needs universe.json + stock_data)')
    ap.add_argument('--start-day', type=int, default=16)
    ap.add_argument('--stop-day', type=int, default=28,
                    help='must span day 25 (MT1_START_DAY) to be meaningful')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--tmp', default=None, help='scratch dir (default: a temp dir on real disk)')
    args = ap.parse_args()

    if args.stop_day <= 25:
        print(f'WARNING: --stop-day {args.stop_day} does not reach day 25 (MT1_START_DAY).')
        print('         MT1 shared state is the prime suspect; this range cannot see it.')

    base = args.tmp or tempfile.mkdtemp(prefix='determinism_', dir='/root'
                                        if os.path.isdir('/root') else None)
    os.makedirs(base, exist_ok=True)
    binary = os.path.abspath(os.path.join(args.cwd, args.binary)) \
        if not os.path.isabs(args.binary) else args.binary

    print(f'Determinism check: {binary}')
    print(f'  days {args.start_day}-{args.stop_day}, seed {args.seed}, workers {args.workers}')
    print(f'  (day 25 = MT1_START_DAY is '
          f'{"INSIDE" if args.stop_day > 25 else "OUTSIDE"} this range)')

    paths = []
    for n in ('run1', 'run2'):
        print(f'  running {n}...', flush=True)
        p = run_once(binary, os.path.join(base, n), args.start_day, args.stop_day,
                     args.seed, args.workers, args.cwd)
        if p is None:
            print('FAIL: run did not produce a CSV')
            return 2
        paths.append(p)

    identical, diffs = compare(*paths)
    print()
    if identical:
        print(f'RESULT: DETERMINISTIC at --workers {args.workers}')
        print('  Same seed reproduces every cell. A/B comparison is valid at this setting.')
        return 0

    print(f'RESULT: NOT DETERMINISTIC at --workers {args.workers}')
    print(f'  {len(diffs)} differing cell(s). A/B comparison at this setting CANNOT')
    print('  distinguish a code change from run-to-run noise.')
    first_day = min((int(d[1]) for d in diffs if d[1].isdigit()), default=None)
    if first_day is not None:
        print(f'  First divergence at day {first_day}'
              + ('  <-- MT1_START_DAY' if first_day == 25 else ''))
    cols = {}
    for _, _, col, _, _ in diffs:
        cols[col] = cols.get(col, 0) + 1
    print('  Columns affected (top 6):')
    for col, n in sorted(cols.items(), key=lambda kv: -kv[1])[:6]:
        print(f'    {col}: {n}')
    print('  Sample:')
    for _, day, col, va, vb in diffs[:5]:
        print(f'    day {day} {col}: {va} vs {vb}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
