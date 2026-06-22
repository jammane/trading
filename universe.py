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
