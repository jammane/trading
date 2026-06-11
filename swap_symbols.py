"""
swap_symbols.py — replace ticker symbols across all project source files.

Usage:
    python swap_symbols.py '{"OLDTICKER": "NEWTICKER", "FOO": "BAR"}'

The replacement is done as an exact quoted-string match inside Python list
literals (e.g. 'OLDTICKER' → 'NEWTICKER'), so partial-match false positives
(e.g. 'C' inside 'CRWD') are impossible.

Files edited:
    universe.py   (single source of truth for all ticker symbols)
    universe.json (auto-generated from universe.py; read by the C++ trainer at runtime)

Note: run this script locally in the development working tree, not inside a
running container.  Container source files are baked into the image layer at
build time; edits made inside a container are discarded when the pod exits.
After swapping symbols, re-download data for the new ticker, remove the old
symbol's JSON from stock_data/, then rebuild and push the Docker image so the
updated universe.py is included in the next training or production run.
"""

import json
import os
import re
import sys

TARGET_FILES = [
    'universe.py',
]


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


def export_universe_json(script_dir: str) -> None:
    """Regenerate universe.json from the current universe.py (read by C++ trainer at runtime)."""
    import importlib.util

    path = os.path.join(script_dir, 'universe.py')
    spec = importlib.util.spec_from_file_location('_universe_fresh', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = {k: list(v) for k, v in mod.INDUSTRIES.items()}
    dst = os.path.join(script_dir, 'universe.json')
    with open(dst, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  universe.json: regenerated ({len(data)} industries, "
          f"{sum(len(v) for v in data.values())} symbols)")


def main():
    """Parse the JSON symbol-map from argv[1] and apply it to all TARGET_FILES."""
    if len(sys.argv) != 2:
        print("Usage: python swap_symbols.py '{\"OLD\": \"NEW\", ...}'")
        sys.exit(1)

    try:
        symbol_map = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)

    if not isinstance(symbol_map, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in symbol_map.items()
    ):
        print("JSON must be a flat {\"OLD\": \"NEW\"} dictionary of strings.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    total = 0
    for filename in TARGET_FILES:
        path = os.path.join(script_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue
        total += swap_in_file(path, symbol_map)

    if total == 0:
        print("No occurrences found — nothing changed.")
    else:
        print(f"\nDone. {total} total replacement{'s' if total > 1 else ''} made.")
        print("Remember to re-download data for new symbols and delete old JSON files from stock_data/.")

    export_universe_json(script_dir)


if __name__ == '__main__':
    main()
