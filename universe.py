"""
universe.py — Aggregator: discovers all universe_acct*.py files and exposes
the union of their symbols as INDUSTRIES, ALL_SYMBOLS, and INDUSTRY_NAMES.

All existing `from universe import INDUSTRIES` imports continue to work.

To add a new account universe, create universe_acct1.py with the same
INDUSTRIES dict structure as universe_acct0.py. This file auto-discovers it.
For download_daily.py and cleanup_stock_data.py the union covers all accounts.
Per-account scripts (production_v2.py, training) use their own file directly.
"""
import glob
import json
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

INDUSTRIES: dict[str, list[str]] = {}

for _path in sorted(glob.glob(os.path.join(_HERE, 'universe_acct*.py'))):
    _spec = importlib.util.spec_from_file_location('_acct_universe', _path)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _ind, _syms in _mod.INDUSTRIES.items():
        if _ind not in INDUSTRIES:
            INDUSTRIES[_ind] = list(_syms)
        else:
            _seen = set(INDUSTRIES[_ind])
            INDUSTRIES[_ind].extend(s for s in _syms if s not in _seen)

ALL_SYMBOLS:    list[str] = [sym for syms in INDUSTRIES.values() for sym in syms]
INDUSTRY_NAMES: list[str] = list(INDUSTRIES.keys())

# ── Cross-environment universes ────────────────────────────────────────────────
#
# dev, paper and prod may legitimately hold DIFFERENT universes — that is what a staged rollout
# looks like: prod trades a proven set while dev tests a new one and paper validates in between.
# They nonetheless SHARE stock_data/ (the droplet symlinks trading-ht's to trading's).
#
# So anything that deletes from stock_data must consider all three. A cleanup run from one
# worktree sees only that worktree's universe: on 2026-09-17, dev had been re-normalized to 144
# new symbols while paper still held the old ones, and the weekly cron — which runs from the paper
# worktree — would have deleted all 102 freshly downloaded symbols mid-training-run.
#
# Each environment's universe is read from its BRANCH, not from a worktree path, so this works
# regardless of which worktree it runs in and whether the others are checked out at all.

ENVIRONMENT_BRANCHES = {'prod': 'main', 'paper': 'paper', 'dev': 'dev'}


def _universe_from_branch(branch: str):
    """Symbols in `branch`'s universe.json, or None if it cannot be read."""
    import subprocess
    try:
        r = subprocess.run(['git', 'show', f'{branch}:universe.json'],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        inds = data['industries'] if isinstance(data, dict) and 'industries' in data else data
        return {s for syms in inds.values() for s in syms}
    except Exception:
        return None


def environment_universes() -> dict:
    """{environment: set(symbols) or None}. None means that environment could not be read.

    The working tree is included as 'working' because it may be ahead of every branch.
    """
    out = {env: _universe_from_branch(br) for env, br in ENVIRONMENT_BRANCHES.items()}
    out['working'] = set(ALL_SYMBOLS)
    return out


def union_across_environments():
    """(union_of_symbols, per_environment_dict, unreadable_environments).

    Callers that DELETE must refuse to act when `unreadable` is non-empty: removing shared data
    on a partial view is precisely the failure this exists to prevent.
    """
    per = environment_universes()
    unreadable = sorted(e for e, v in per.items() if v is None)
    union = set()
    for v in per.values():
        if v:
            union |= v
    return union, per, unreadable
