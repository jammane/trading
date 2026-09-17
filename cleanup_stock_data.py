#!/usr/bin/env python3
"""
cleanup_stock_data.py — Remove stock_data/*.json files for symbols no longer
active in any account's universe or open positions.

A symbol is kept if it appears in ANY environment's universe (dev / paper / prod, read from
their branches) OR has a non-zero holding in any models/acct*/paper|prod/state.json.

The three environments may legitimately hold different universes — a staged rollout means prod
trades a proven set while dev tests a new one — but they SHARE stock_data/, so a cleanup that
sees only one of them deletes the others' symbols. The second check
prevents purging a symbol that was swapped out of the universe but still has an
open Alpaca position pending liquidation.

Run weekly (e.g. Sundays 4 AM). Safe to run at any time.

Usage:
  python cleanup_stock_data.py           # live removal
  python cleanup_stock_data.py --dry-run # preview only
"""
import argparse
import json
import os

from universe import union_across_environments

STOCK_DATA_DIR = 'stock_data'


def _active_symbols(union: set) -> set:
    """Union of every environment's universe plus anything currently held."""
    active = set(union)
    models_root = 'models'
    if not os.path.isdir(models_root):
        return active
    for acct in sorted(os.listdir(models_root)):
        acct_dir = os.path.join(models_root, acct)
        if not os.path.isdir(acct_dir):
            continue
        for subtype in ('paper', 'prod'):
            state_path = os.path.join(acct_dir, subtype, 'state.json')
            if not os.path.exists(state_path):
                continue
            try:
                with open(state_path) as f:
                    state = json.load(f)
                for sym, qty in state.get('holdings', {}).items():
                    if qty and float(qty) > 0:
                        active.add(sym)
            except Exception as e:
                print(f'  [warn] could not read {state_path}: {e}')
    return active


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Remove stock_data files for symbols not in any active universe or held position')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be removed without deleting anything')
    args = parser.parse_args()

    if not os.path.isdir(STOCK_DATA_DIR):
        print(f'{STOCK_DATA_DIR}/ not found — nothing to clean.')
        return

    # dev / paper / prod may legitimately hold DIFFERENT universes — a staged rollout means prod
    # trades a proven set while dev tests a new one. They SHARE stock_data, so deleting on one
    # environment's view destroys another's data. Union across all of them.
    union, per_env, unreadable = union_across_environments()
    print('Universes by environment:')
    for env in ('prod', 'paper', 'dev', 'working'):
        syms = per_env.get(env)
        print(f'  {env:<8} {"UNREADABLE" if syms is None else str(len(syms)) + " symbols"}')
    only = {env: sorted(s - set().union(*(v for e, v in per_env.items() if e != env and v)))
            for env, s in per_env.items() if s}
    for env, syms in only.items():
        if syms:
            print(f'  only in {env}: {len(syms)} — {", ".join(syms[:8])}'
                  f'{"..." if len(syms) > 8 else ""}')

    if unreadable:
        # Refuse rather than delete on a partial view. That partial view is the whole bug: on
        # 2026-09-17 a cleanup from the paper worktree would have deleted 102 symbols that only
        # dev's universe knew about, mid-training-run.
        print(f'\nREFUSING TO DELETE: could not read the universe for {", ".join(unreadable)}.')
        print('stock_data is shared across environments; removing files without seeing every')
        print('universe risks deleting another environment\'s symbols. Fix the branch(es) and')
        print('re-run, or use --dry-run to preview.')
        if not args.dry_run:
            raise SystemExit(2)

    active    = _active_symbols(union)
    all_files = sorted(f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.json'))
    stale     = [f for f in all_files if f[:-5] not in active]

    print(f'\nActive symbols (union): {len(active)}  |  Files on disk: {len(all_files)}  |  Stale: {len(stale)}')

    if not stale:
        print('Nothing to remove.')
        return

    tag = '[DRY RUN] ' if args.dry_run else ''
    for fname in stale:
        print(f'  {tag}removing {fname}')
        if not args.dry_run:
            os.remove(os.path.join(STOCK_DATA_DIR, fname))

    if args.dry_run:
        print(f'\nDry run complete — {len(stale)} file(s) would be removed.')
    else:
        print(f'\nRemoved {len(stale)} stale file(s).')


if __name__ == '__main__':
    main()
