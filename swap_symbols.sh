#!/usr/bin/env bash
#
# swap_symbols.sh — guided ticker-replacement workflow.
#
# Runs all steps required after changing a symbol in the trading universe:
#   1. Updates universe.py via swap_symbols.py
#   2. Removes stale stock_data/<OLD>.json files
#   3. Downloads historical data for the new symbols
#   4. Optionally rebuilds and pushes the Docker image
#
# Usage:
#   ./swap_symbols.sh '{"OLDTICKER": "NEWTICKER"}'
#   ./swap_symbols.sh '{"FOO": "BAR", "BAZ": "QUX"}'
#
# Run from the project root on the local development machine.
# Do not run inside a Docker container — source-file changes are discarded
# when the pod exits.  After this script completes, redeploy any running
# Kubernetes jobs so they pick up the updated image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Argument validation ────────────────────────────────────────────────────────
if [ $# -ne 1 ]; then
    echo "Usage: $0 '{\"OLD\": \"NEW\", ...}'"
    exit 1
fi

JSON_MAP="$1"

if ! python3 -c "
import json, sys
m = json.loads(sys.argv[1])
if not isinstance(m, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in m.items()):
    sys.exit(1)
" "$JSON_MAP" 2>/dev/null; then
    echo "Error: argument must be a valid flat JSON object, e.g. '{\"OLDTICKER\": \"NEWTICKER\"}'"
    exit 1
fi

# ── Activate virtual environment ───────────────────────────────────────────────
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f ".venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source .venv/bin/activate
    else
        echo "Error: virtual environment not found at .venv/. Run install_python.sh first."
        exit 1
    fi
fi

# ── Step 1: update universe.py ────────────────────────────────────────────────
echo ""
echo "==> Step 1/4: Updating universe.py ..."
python swap_symbols.py "$JSON_MAP"

# ── Step 2: remove stale stock data ───────────────────────────────────────────
echo ""
echo "==> Step 2/4: Removing stale stock data for replaced symbols ..."
OLD_TICKERS=$(python3 -c "import json, sys; m=json.loads(sys.argv[1]); print('\n'.join(m.keys()))" "$JSON_MAP")
while IFS= read -r ticker; do
    path="stock_data/${ticker}.json"
    if [ -f "$path" ]; then
        rm "$path"
        echo "  Removed $path"
    else
        echo "  $path not found — skipping"
    fi
done <<< "$OLD_TICKERS"

# ── Step 3: download data for new symbols ─────────────────────────────────────
echo ""
echo "==> Step 3/4: Downloading historical data for new symbols ..."
python download_5y_data.py

# ── Step 4: rebuild Docker image ──────────────────────────────────────────────
echo ""
echo "==> Step 4/4: Docker image"
echo "universe.py has been updated.  Rebuild the Docker image so the new symbol"
echo "set is baked into the next training or production container."
echo ""
read -rp "Rebuild and push jammane80/trading:latest now? [y/N] " REPLY
if [[ "${REPLY:-}" =~ ^[Yy]$ ]]; then
    docker build -t jammane80/trading:latest .
    docker push jammane80/trading:latest
    echo "Image rebuilt and pushed successfully."
else
    echo ""
    echo "Skipped.  Run when ready:"
    echo "  docker build -t jammane80/trading:latest ."
    echo "  docker push jammane80/trading:latest"
fi

echo ""
echo "Symbol swap complete."
