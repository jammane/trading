# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

**Setup (Linux/Fedora):**
```bash
bash install_python.sh
source .venv/bin/activate
pip install keyring
```

**Setup (DigitalOcean droplet — includes Claude Code for full test suite):**
```bash
# Python environment
bash install_python.sh
source .venv/bin/activate

# Claude Code (Node.js 22 is in the Fedora repos directly)
dnf install -y nodejs npm
npm install -g @anthropic-ai/claude-code

# Store Anthropic API key in kubectl (consistent with Alpaca credentials — never written to disk)
kubectl create secret generic anthropic-credentials \
    --namespace trading \
    --from-literal=ANTHROPIC_API_KEY="sk-ant-..." \
    --dry-run=client -o yaml | kubectl apply -f -

# Export the key to the current shell before running claude
export ANTHROPIC_API_KEY=$(kubectl get secret anthropic-credentials \
    -n trading -o jsonpath='{.data.ANTHROPIC_API_KEY}' | base64 -d)

claude
```
All 75 pytest tests (including `test_models.py`) run on the droplet where torch is available.
The pre-commit hook runs the full suite automatically before every `git commit`.

**Lint:**
```bash
ruff check .
ruff check --fix .
```

**Run tests:**
```bash
.venv/bin/pytest tests/ -v
```

**Download training data:**
```bash
python download_5y_data.py
```

**Train (single-threaded):**
```bash
python training_v2.py --output models
python training_v2.py --output models --load-dir models          # resume from checkpoint
python training_v2.py --output models --start-day 16 --stop-day 35 --passes 1 --preserve-stock-data --no-save-master  # short diagnostic run
```

**Train (C++ binary — canonical; handles both industries and master):**
```bash
# Seed once from existing Python models (or after any convert_weights.py run):
python prepare_models.py --load-dir models/training --output models/training
# Full training run (canonical settings — industries + master):
./build/training_v4_cpp --output models/training --load-dir models/training \
  --passes 5 --sigma 0.008 --master-sigma 0.006 --sigma-decay 1.0 \
  --start-day 17 --stop-day 1255
# Retrain master only (freeze industries, use their slot-0 perf for ind_val_hist):
./build/training_v4_cpp --output models/training --load-dir models/training \
  --master-only --passes 5 --start-day 17 --stop-day 1255
# Short diagnostic (verifies master fires at day 30+):
./build/training_v4_cpp --output /root/diag --load-dir /root/diag \
  --start-day 16 --stop-day 35 --passes 1 --preserve-stock-data
# After training, convert back to .pt before inspect_trades.py or production_v2.py:
python convert_weights.py --models-dir models/training --output models/training
```
Master trains via tier-classification (444 features, FN/FP penalties) starting at day 30.
`convert_weights.py` is required after C++ training before using `inspect_trades.py` or `production_v2.py`.
Note: existing master `.bin` files are incompatible after the architecture change — regenerate with `prepare_models.py`.

**Train (parallel, 7 threads — requires ≥4 GB RAM):**
```bash
python training_v3.py --output models
```

**Inspect elite model trade decisions:**
```bash
python inspect_trades.py --industry energy --date 2024-01-10 --models-dir ./models
python inspect_trades.py --industry energy --day-index 17 --models-dir ./models --top-n 5
```

**Paper / live trading:**
```bash
python production_v2.py --paper --model-dir models
python production_v2.py --model-dir models         # live (requires ALPACA_API_KEY / ALPACA_SECRET_KEY)
```

**Replace ticker symbols (full guided workflow):**
```bash
./swap_symbols.sh '{"OLDTICKER": "NEWTICKER"}'
```
Runs all five steps: updates `universe.py` and regenerates `universe.json`, removes stale `stock_data/` JSON, downloads new symbol data, prompts to rebuild the Docker image, and prints optional model-cleanup commands for the droplet. The C++ binary reads `universe.json` at startup — no recompile needed after a symbol swap. Run locally — not inside a container.

## Shared modules

| Module | Contents |
|--------|----------|
| `models.py` | `StockNN`, `MasterNN` — single source of truth for both model classes |
| `universe.py` | `INDUSTRIES` dict, `ALL_SYMBOLS`, `INDUSTRY_NAMES` — 144-symbol universe |
| `universe.json` | Auto-generated from `universe.py`; read by the C++ trainer at runtime |
| `fees.py` | Fee constants (`BUY_FILL`, `SEC_FEE_RATE`, etc.) and `_sell_net()` helper |
| `prepare_models.py` | `.pt` → `.bin` for C++ trainer (run before first C++ training) |
| `convert_weights.py` | `.bin` → `.pt` + `_best.pt` for Python tools (run after C++ training) |

All training scripts (`training_v2.py`, `training_v3.py`), `production_v2.py`, and `inspect_trades.py` import from these modules. (`training_v4.py` was deleted — superseded by `training_v4_cpp` for all training.) `download_5y_data.py` imports from `universe.py`. To add or change a ticker, run `swap_symbols.sh` — it updates both `universe.py` and `universe.json` together.

## Tests

75 pytest tests across three files in `tests/`:
- `test_models.py` — output shapes, output constraints (ReLU/sigmoid/softmax), serialization roundtrip, inject-layer growth dimensions
- `test_universe.py` — industry count, symbols per industry, no duplicates, formatting
- `test_fees.py` — fee constant values, `_sell_net` calculations, FINRA cap boundary

A `PreToolUse` hook in `.claude/settings.json` runs the suite automatically before every `git commit` or `git push`. Failures are reported before the commit runs, so Claude can self-correct without creating a broken commit. The changelog hook remains `PostToolUse` (it needs the commit hash to exist before it can amend).

## Architecture

### Models

Two model classes are defined identically in `training_v2.py`, `training_v3.py`, and `production_v2.py` — they must stay in sync manually:

- **`StockNN`** — one instance per industry sector (12 sectors). FC injection architecture: seed day → 14 inject layers → today layer → 2 flat layers → funnel. Output is `(12, 4)` — one row per stock in the sector, columns are `[buy_qty, buy_price_frac, sell_all_price_frac, sell_qty]`.
- **`MasterNN`** — single cross-sector allocator. Same injection pattern with wider today vector (229 vs 208 features). Output is `(12, 3)` — per-industry `[allocation_weight, liquidation_depth, liquidation_trigger]`.

### Evolutionary pool (training)

Each industry maintains **200 model slots** on disk as `.pt` files. The slot layout after every selection step is fixed:
- Slots 0–16: direct elites (rank-ordered, slot 0 is the production model)
- Slots 17–19: weighted-average blends (top-5, top-10, top-15 weights)
- Slots 20–199: Gaussian-noise mutations (9 children per parent, deterministic assignment)

Each training day: all 200 slots reset to slot 0's portfolio → infer → simulate fills → score as `delta × invested_pct` → select + mutate. The `invested_pct` multiplier penalises cash-heavy winners.

`training_v4_cpp` (C++ binary) is the canonical trainer — handles both industry and master training with ~6× speedup over Python. Master training uses tier-classification (444 features: 18 delta lookbacks + polynomial regression over per-industry portfolio value history; FN/FP penalty scoring; 3-consecutive-zero liquidation). Master only activates at `actual_day >= 30`.

`training_v3.py` (parallel) differs from `training_v2.py` in: 7 worker threads, in-RAM model cache (`_model_cache`), no slippage on limit fills, and slot-level portfolio JSON persisted alongside weights.

### Fill simulation

Orders are placed end-of-day N and filled against next-day OHLCV with `SLIPPAGE_RATE = 0.001` (v2 only). Strict boundary: a buy limit at exactly `nd_low` does not fill.

### Diagnostics and data dump

When an elite holds ≥50% cash (industry) or ≥80% cash (master), an `UNDER_INVEST` soft flag writes JSON to `data_dump/day_N/<prefix>.json`. A single-day gain >12.5% of baseline raises a `HardFlagError`. Use `inspect_trades.py` to audit flagged days.

### Production cycle

`production_v2.py` runs once per trading day: fetch data from yfinance → run MasterNN to rebalance capital → run StockNN per active industry → submit limit/stop orders to Alpaca → perform one upkeep evolution step on yesterday's data. Alpaca credentials are read from `keyring` or environment variables.

### Changelog hook

A `PostToolUse` hook in `.claude/settings.json` auto-updates `CHANGELOG.md` and amends the commit whenever Claude makes a `git commit`. This is intentional — do not skip it.

## Key constants (defined at top of each training/production file)

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_SLOTS` | 200 | Total model slots per pool |
| `ELITE_COUNT` | 17 | Direct elite slots |
| `ELITE_POOL` | 20 | Elites + weighted-average slots |
| `IND_STARTING_CASH` | $25,000 | Per-industry starting capital |
| `MST_STARTING_CASH` | $300,000 | Master starting capital |
| `MAX_SINGLE_STOCK_PCT` | 0.60 | Max fraction of industry cash in one stock |

## Ignored directories

`models/`, `stock_data/`, and `data_dump/` are git-ignored (large binaries, data, and diagnostics).
