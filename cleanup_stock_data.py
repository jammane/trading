#!/usr/bin/env python3
"""
cleanup_stock_data.py — Remove stock_data/*.json files for symbols no longer
active in any account's universe or open positions.

A symbol is kept if it appears in the current universe (universe.py) OR has a
non-zero holding in any models/acct*/paper|prod/state.json. The second check
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

from universe import ALL_SYMBOLS

STOCK_DATA_DIR = 'stock_data'


def _active_symbols() -> set:
    """Union of universe symbols and any currently held across all accounts."""
    active = set(ALL_SYMBOLS)
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

    active    = _active_symbols()
    all_files = sorted(f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.json'))
    stale     = [f for f in all_files if f[:-5] not in active]

    print(f'Active symbols: {len(active)}  |  Files on disk: {len(all_files)}  |  Stale: {len(stale)}')

    if not stale:
        print('Nothing to remove.')
        return

    tag = '[DRY RUN] ' if args.dry_run else ''
    for fname in stale:
        sym = fname[:-5]
        print(f'  {tag}removing {fname}')
        if not args.dry_run:
            os.remove(os.path.join(STOCK_DATA_DIR, fname))

    if args.dry_run:
        print(f'\nDry run complete — {len(stale)} file(s) would be removed.')
    else:
        print(f'\nRemoved {len(stale)} stale file(s).')


if __name__ == '__main__':
    main()
