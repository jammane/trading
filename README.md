# trading

Neural network-based stock trading system. Trains an evolutionary population of LSTM models (one per industry sector + a master allocator) on historical daily OHLCV data, then runs paper or live orders through Alpaca.

## Architecture

### Industry sub-models (StockNN)

Each of 12 industry sectors has its own pool of 100 StockNN models. StockNN is a 4-layer LSTM (208→208→208→155) followed by 3 fully-connected layers (168→128→88→48). Its output is a 12×4 matrix — one row per stock:

| Column | Meaning | Activation |
|--------|---------|------------|
| buy_qty | Shares to buy | ReLU (0 = no order) |
| buy_price_frac | Buy limit price as fraction of day's range | Sigmoid → [low, high] |
| sell_all_price_frac | Sell-all limit price as fraction of day's range | Sigmoid |
| sell_qty | Shares to sell at next-day open (market) | ReLU |

Each industry portfolio starts at $1,666.67 (total $20,000 across 12 sectors).

### Master allocator (MasterNN)

MasterNN is a 4-layer LSTM (229→229→229→152) with 3 FC layers. It allocates the $20,000 master portfolio across 12 industry "units", controlling allocation weight, position depth, and entry trigger per industry.

### Evolutionary training

Every training day:
1. **Reset** — all 100 slots in a pool are reset to the 10th-ranked portfolio from the prior day, giving every model the same starting point (level playing field — only trading decisions are rewarded).
2. **Step** — each model runs its forward pass; limit/stop orders are simulated against next-day OHLCV (fills execute on the *next* trading day, not the decision day).
3. **Score** — portfolio value computed at fill-day close prices (same day for baseline and post-trade, so the delta measures only trading alpha, not overnight price movement).
4. **Selection scoring** — raw delta vs. baseline is adjusted by an `invested_pct` multiplier: slots with positive deltas are scaled down if they held large cash positions (rewarding deployed capital, not idle cash). Slots below a survival floor (`−10%` of baseline for industry, `0` for master) are excluded before selection.
5. **Select** — top 8 become elites; weighted-average blending and Gaussian-noise mutation produce 90 offspring. The bottom half of elites are replaced by diversity-injection blends when a streak of fully-filtered days is detected.

### Soft flags and diagnostics

When an elite candidate holds ≥ 50% cash (industry) or ≥ 80% cash (master), an `UNDER_INVEST` soft flag fires. The day's scores, prices, and fill data are written to `data_dump/day_N/` for offline analysis. Flags only fire for elite candidates — under-invested non-elite slots are silently ignored.

### End-of-training ranking

After all training passes complete, a sorted table is printed showing per-industry: start value, end value, delta, % return, pos:zero:neg day ratio, minimum 3-stock entry cost (most expensive + 2 cheapest at end-of-training prices), and top stock names/prices. Use the pos:zero:neg ratio and min-entry cost to prioritise which industries to activate first in production.

### Fill timing and valuation convention

Orders are placed at end-of-day N; fills simulate on day N+1:

- **Buy limit**: fills if `nd_low < buy_price < nd_high` (strict — no fill at exact extreme). Slippage: `fill_price = buy_price × 1.001`.
- **Sell-all limit**: fills if `nd_open ≥ sell_all_price` (at open) or `nd_low < sell_all_price < nd_high` (at slipped price `× 0.999`).
- **Partial sell**: executes at `nd_open` (market order, no slippage).
- **Stop-loss**: fills if `nd_low ≤ stop_price`; slipped to `stop_price × 0.999`.

Both baseline and post-trade portfolio values are computed at fill-day close so gains reflect only trading decisions, not overnight price moves.

## Quick start

### 1. Download data

```bash
source .venv/bin/activate
python download_5y_data.py
```

Saves ~5 years of daily OHLCV JSON for all 144 symbols under `stock_data/`.

### 2. Train models

Full run from the beginning:
```bash
python training_v2.py --output models
```

Continue from a checkpoint:
```bash
python training_v2.py --output models --load-dir models
```

Short diagnostic run (days 17–22 only, 1 pass):
```bash
python training_v2.py --output models --preserve-stock-data --start-day 16 --stop-day 21 --passes 1
```

### 3. Inspect trades

Examine what the elite models actually did on a specific day, with full fill-condition explanations, gain attribution, and cash utilisation metrics:

```bash
# By calendar date
python inspect_trades.py --industry energy --date 2024-01-10 --models-dir ./models

# By training day index (matches --start-day / --stop-day used in training_v2.py)
python inspect_trades.py --industry energy --day-index 17 --models-dir ./models --top-n 5
```

### 4. Paper trading

```bash
python production_v2.py --paper --model-dir models
```

### 5. Live trading

```bash
python production_v2.py --model-dir models
```

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in your environment.

### Production industry flag system

Before each trading loop, `production_v2.py` checks whether each industry's allocated capital meets the single-share floor for that industry's most expensive stock. Industries below the floor are skipped for new orders (liquidations still execute). A summary is printed at startup listing inactive industries with their current capital, the floor they need to clear, and the minimum 3-stock entry cost.

Industries accumulate capital over time via master rebalancing — no manual intervention is needed; they activate automatically once the floor is reached.

**Note:** Alpaca limit orders do not support fractional share quantities. The system uses limit orders to preserve price-signal alignment with training. Minimum capital per industry should be sized to afford at least one full share of the most expensive stock in that sector.

## Setup

### Linux / Fedora

```bash
bash install_python.sh
source .venv/bin/activate
pip install keyring
```

### Windows

```bat
install_python.bat
call env\Scripts\activate
pip install keyring
```

## File reference

| File | Purpose | Command |
|------|---------|---------|
| `install_python.sh` | Linux/Fedora: create `.venv` and install packages | `bash install_python.sh` |
| `install_python.bat` | Windows: create `env` and install packages | `install_python.bat` |
| `download_5y_data.py` | Download ~5 years of daily OHLCV into `stock_data/` | `python download_5y_data.py` |
| `training_v2.py` | Train evolutionary LSTM models | `python training_v2.py --output models` |
| `training_v2.py` | Continue training from checkpoint | `python training_v2.py --output models --load-dir models` |
| `training_v2.py` | Short diagnostic run (days 16–21, 1 pass) | `python training_v2.py --output models --preserve-stock-data --start-day 16 --stop-day 21 --passes 1` |
| `inspect_trades.py` | Audit elite model decisions for a given day | `python inspect_trades.py --industry energy --day-index 17 --models-dir ./models` |
| `production_v2.py` | Paper trading via Alpaca | `python production_v2.py --paper --model-dir models` |
| `production_v2.py` | Live order submission via Alpaca | `python production_v2.py --model-dir models` |
| `production_v2.py` | Request a withdrawal during a run | `python production_v2.py --paper --model-dir models --withdraw 500` |
| `update_shares.py` | Record an investment for an owner | `python update_shares.py --invest Alice 1000` |
| `update_shares.py` | Record a withdrawal for an owner | `python update_shares.py --withdrawal Alice 250` |
| `update_shares.py` | Export `owners/owners_value.csv` | `python update_shares.py --value` |

## training_v2.py flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | (required) | Directory to save trained `.pt` models and metadata |
| `--load-dir` | None | Load existing models from this directory before training |
| `--start-day` | 0 | First training day index (0-based, relative to `all_data` sorted dates) |
| `--stop-day` | end | Last training day index (exclusive) |
| `--passes` | 1 | Number of full training passes over the day range |
| `--sigma` | 0.01 | Initial Gaussian mutation standard deviation |
| `--sigma-decay` | 0.5 | Multiply sigma by this value after each pass |
| `--preserve-stock-data` | False | Do not trim `stock_data/` JSON files after training |

## Stock universe

144 symbols across 12 sectors (12 per sector). Selected for high beta and volatility. Notable replacements from original list:

| Sector | Removed | Replacement | Reason |
|--------|---------|-------------|--------|
| financials | SQ | XYZ (Block Inc) | SQ ticker renamed to XYZ in Jan 2025 |
| industrials | X (US Steel) | BTU (Peabody Energy) | X acquired by Nippon Steel 2025, delisted |
| utilities | NOVA (Sunnova) | ARRY (Array Technologies) | NOVA filed Chapter 11 Jun 2025, delisted |
| real_estate | RDFN (Redfin) | OPEN (Opendoor Technologies) | RDFN acquired by Rocket Companies Jul 2025, delisted |
| real_estate | NVR | LGIH (LGI Homes) | NVR at ~$6,800 was untradeable at small account sizes |

## Notes

- `models/`, `stock_data/`, and `data_dump/` are git-ignored (large binary / data / diagnostic files).
- `production_v2.py` imports `keyring` — install it before running.
- `install_python.sh` creates `.venv`; `install_python.bat` creates `env`.
- Use `--paper` before attempting live trading.
- `training_log.csv` records slot-0 (the production model's) pre-selection trading outcomes — not best-of-200.
