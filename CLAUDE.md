# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

**Setup (Linux/Fedora):**
```bash
bash install_python.sh
source .venv/bin/activate
pip install keyring
```

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
python training_v2.py --output models --start-day 16 --stop-day 21 --passes 1 --preserve-stock-data  # short diagnostic run
```

**Train (parallel, 2-process dynamic industry pool):**
```bash
python training_v4.py --output models
python training_v4.py --output models --load-dir models          # resume from checkpoint
python training_v4.py --output models --start-day 16 --stop-day 21 --passes 1 --preserve-stock-data  # short diagnostic run
```

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
Runs all four steps: updates `universe.py`, removes stale `stock_data/` JSON, downloads new symbol data, and prompts to rebuild the Docker image.  Run locally — not inside a container.

## Shared modules

| Module | Contents |
|--------|----------|
| `models.py` | `StockNN`, `MasterNN` — single source of truth for both model classes |
| `universe.py` | `INDUSTRIES` dict, `ALL_SYMBOLS`, `INDUSTRY_NAMES` — 144-symbol universe |
| `fees.py` | Fee constants (`BUY_FILL`, `SEC_FEE_RATE`, etc.) and `_sell_net()` helper |

All training scripts (`training_v2.py`, `training_v3.py`, `training_v4.py`), `production_v2.py`, and `inspect_trades.py` import from these modules. `download_5y_data.py` imports from `universe.py`. To add or change a ticker, edit `universe.py` only — or run `swap_symbols.sh` for the full guided workflow.

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

`training_v4.py` (parallel) differs from `training_v2.py` (single-threaded) in: 2 worker *processes* (`ProcessPoolExecutor`) replacing the sequential industry loop, bypassing the Python GIL for genuine parallel execution of the trade simulation. Portfolio and history state is pickled to each worker and the mutated copies are returned and reassigned in the main process after each day. Expected speedup: ~40% (~3 min/day vs ~5 min/day) on a 2-vCPU host.

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
