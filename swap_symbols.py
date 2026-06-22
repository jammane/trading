"""
swap_symbols.py — replace ticker symbols in a per-account universe file.

Usage:
    python swap_symbols.py '{"OLDTICKER": "NEWTICKER", "FOO": "BAR"}'
    python swap_symbols.py --account acct1 '{"OLD": "NEW"}'   # future accounts

The replacement is done as an exact quoted-string match inside Python list
literals (e.g. 'OLDTICKER' → 'NEWTICKER'), so partial-match false positives
(e.g. 'C' inside 'CRWD') are impossible.

Files edited:
    universe_ACCOUNT.py  (per-account symbol universe)
    universe.json        (auto-generated from universe_ACCOUNT.py; read by the C++ trainer)
"""

import json
import os
import re
import sys


def swap_in_file(path: str, symbol_map: dict[str, str]) -> int:
    """Replace quoted ticker symbols in *path* using *symbol_map*; returns replacement count."""
    with open(path) as f:
        original = f.read()

    text = original
    replacements = 0
    for old, new in symbol_map.items():
        # Match the symbol as a single-quoted Python string token.
        # Handles surrounding whitespace/commas in list literals safely.
        pattern = rf"(?<![A-Z0-9])'{re.escape(old)}'(?![A-Z0-9])"
        count = len(re.findall(pattern, text))
        if count:
            text = re.sub(pattern, f"'{new}'", text)
            replacements += count
            print(f"  {path}: '{old}' → '{new}' ({count} occurrence{'s' if count > 1 else ''})")

    if text != original:
        with open(path, 'w') as f:
            f.write(text)

    return replacements


def export_universe_json(script_dir: str, account: str) -> None:
    """Regenerate universe.json from universe_ACCOUNT.py (read by C++ trainer at runtime).

    universe.json is always the per-account view, not the multi-account union,
    because the C++ trainer operates on a single account at a time.
    """
    import importlib.util

    src = os.path.join(script_dir, f'universe_{account}.py')
    if not os.path.exists(src):
        print(f"  WARNING: {src} not found — universe.json not updated")
        return
    spec = importlib.util.spec_from_file_location('_universe_fresh', src)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = {k: list(v) for k, v in mod.INDUSTRIES.items()}
    dst  = os.path.join(script_dir, 'universe.json')
    with open(dst, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  universe.json: regenerated from {os.path.basename(src)} "
          f"({len(data)} industries, {sum(len(v) for v in data.values())} symbols)")


def main():
    """Parse optional --account flag plus JSON symbol-map from argv and apply."""
    args = sys.argv[1:]
    account = 'acct0'
    if args and args[0] == '--account':
        if len(args) < 3:
            print("Usage: python swap_symbols.py [--account ACCT] '{\"OLD\": \"NEW\", ...}'")
            sys.exit(1)
        account = args[1]
        args = args[2:]
    if len(args) != 1:
        print("Usage: python swap_symbols.py [--account ACCT] '{\"OLD\": \"NEW\", ...}'")
        sys.exit(1)

    try:
        symbol_map = json.loads(args[0])
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)

    if not isinstance(symbol_map, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in symbol_map.items()
    ):
        print("JSON must be a flat {\"OLD\": \"NEW\"} dictionary of strings.")
        sys.exit(1)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    universe_file = f'universe_{account}.py'
    path = os.path.join(script_dir, universe_file)
    if not os.path.exists(path):
        print(f"Error: {universe_file} not found. Create it first.")
        sys.exit(1)

    total = swap_in_file(path, symbol_map)

    if total == 0:
        print("No occurrences found — nothing changed.")
    else:
        print(f"\nDone. {total} total replacement{'s' if total > 1 else ''} made.")
        print("Remember to re-download data for new symbols and delete old JSON files from stock_data/.")

    export_universe_json(script_dir, account)


if __name__ == '__main__':
    main()
