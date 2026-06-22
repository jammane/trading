#!/usr/bin/env bash
#
# swap_symbols.sh — guided ticker-replacement workflow.
#
# Runs all steps required after changing a symbol in the trading universe:
#   1. Updates universe.py and regenerates universe.json via swap_symbols.py
#   2. Removes stale stock_data/<OLD>.json files
#   3. Downloads historical data for the new symbols
#   4. Optionally rebuilds and pushes the Docker image
#   5. Prints optional commands to clean stale model files on the droplet
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

# ── Step 1: update universe_acct0.py ─────────────────────────────────────────
echo ""
echo "==> Step 1/5: Updating universe_acct0.py ..."
python swap_symbols.py "$JSON_MAP"

# ── Step 2: remove stale stock data ───────────────────────────────────────────
echo ""
echo "==> Step 2/5: Removing stale stock data for replaced symbols ..."
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
echo "==> Step 3/5: Downloading data for new symbols (incremental update for all others) ..."
python download_daily.py

# ── Step 4: rebuild Docker image ──────────────────────────────────────────────
echo ""
echo "==> Step 4/5: Docker image"
echo "universe_acct0.py has been updated.  Rebuild the Docker image so the new symbol"
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


# ── Step 5: optional stale model cleanup on the droplet ───────────────────────
echo ""
echo "==> Step 5/5: Droplet model cleanup (optional)"
echo "The C++ binary reads universe.json at startup — no recompile needed."
echo ""

AFFECTED_INDS=$(python3 - "$JSON_MAP" <<'PYEOF'
import json, sys
sys.path.insert(0, '.')
from universe import INDUSTRIES
sym_map = json.loads(sys.argv[1])
new_syms = set(sym_map.values())
for ind_name, syms in INDUSTRIES.items():
    if any(s in new_syms for s in syms):
        print(ind_name)
PYEOF
)

if [ -n "$AFFECTED_INDS" ]; then
    echo "Affected industries (weights trained on old symbols — optional cleanup):"
    while IFS= read -r ind; do echo "  $ind"; done <<< "$AFFECTED_INDS"
    echo ""
    echo "To reset these industries to random init on the droplet (recommended):"
    while IFS= read -r ind; do
        echo "  ssh root@DROPLET_IP 'rm -f /root/trading/models/training/${ind}_elite_{0..19}.bin /root/trading/models/training/${ind}_model_{0..19}.pt /root/trading/models/training/${ind}_best.pt'"
    done <<< "$AFFECTED_INDS"
    echo ""
else
    echo "No industries affected — no model cleanup needed."
fi

echo ""
echo "Symbol swap complete."
