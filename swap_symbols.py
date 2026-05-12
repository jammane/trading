"""
swap_symbols.py — replace ticker symbols across all project source files.

Usage:
    python swap_symbols.py '{"OLDTICKER": "NEWTICKER", "FOO": "BAR"}'

The replacement is done as an exact quoted-string match inside Python list
literals (e.g. 'OLDTICKER' → 'NEWTICKER'), so partial-match false positives
(e.g. 'C' inside 'CRWD') are impossible.

Files edited:
    download_5y_data.py
    training_v2.py
    training_v3.py
    production_v2.py
"""

import json
import re
import sys
import os

TARGET_FILES = [
    'download_5y_data.py',
    'training_v2.py',
    'training_v3.py',
    'production_v2.py',
]


def swap_in_file(path: str, symbol_map: dict[str, str]) -> int:
    with open(path, 'r') as f:
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


def main():
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


if __name__ == '__main__':
    main()
