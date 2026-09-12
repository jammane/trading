"""Import smoke tests — every module must at least be importable.

Why this file exists: `production_v2.py` sat unimportable on the mt1-heads-tails branch for many
versions (`ImportError: cannot import name 'load_mt2_norm_stats' from 'upkeep'` — the function was
deleted in eb70bea/v0.2.0.0 but its import and call sites were left behind). Nothing caught it
because no test imported the module. These tests close that gap for every module at once: they are
cheap, and they catch the entire class of "a symbol was deleted and a stale reference was left".

Excluded: `download_5y_data.py`, which is a script, not a module — it runs its download loops at
top level, so importing it would hit the network and rewrite `stock_data/`.
"""

import importlib

import pytest

# Every importable top-level module in the repo.
MODULES = [
    "cleanup_stock_data",
    "convert_weights",
    "download_daily",
    "fees",
    "inspect_trades",
    "models",
    "plot_training",
    "prepare_models",
    "production_v2",
    "read_mt_log",
    "swap_symbols",
    "training_lib",
    "universe",
    "universe_acct0",
    "upkeep",
    "version",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """The module imports without raising."""
    assert importlib.import_module(module_name) is not None


def test_download_5y_data_is_excluded_deliberately():
    """Guard the exclusion above: if this ever grows a __main__ guard, add it to MODULES."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "download_5y_data.py"
    tree = ast.parse(src.read_text())
    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "__name__"
        for node in tree.body
    )
    assert not has_main_guard, (
        "download_5y_data.py now has a __main__ guard, so it is import-safe — "
        "add it to MODULES and delete this test."
    )
