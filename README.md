# Trading — Neural Network Stock Trading System

A neural network-based algorithmic trading system that trains an evolutionary population of FC-injection models (one pool per industry sector, plus a master allocator) on historical daily OHLCV data, then executes paper or live orders through the Alpaca brokerage API.

---

> **Development Status**
>
> This project is under active development and is not yet production-ready. Core training functionality is operational; paper and live trading have not been fully tested and should be treated as experimental. Breaking changes to model architecture, configuration, and CLI flags may occur between versions. Use in live trading environments is at your own risk.

---

## Architecture

### Industry sub-models (StockNN)

Each of 12 industry sectors maintains an independent pool of 200 StockNN models. StockNN uses a fully-connected injection architecture — no recurrent layers:

- **Seed** (day 15 of history): 60-feature OHLCV vector → FC → 120
- **Inject** (×14 earlier days): concatenated [prior hidden, day features] → FC, growing 120→190
- **Today**: concatenated [final hidden (190), today features (208)] → FC → 300
- **Flat** (×2): 300 → 300
- **Funnel**: 300 → 237 → 174 → 111 → 48

The 48-element output is reshaped to a 12×4 matrix — one row per stock in the sector:

| Column | Meaning | Activation |
|--------|---------|------------|
| `buy_qty` | Shares to buy | ReLU (0 = no order) |
| `buy_price_frac` | Buy limit price as a fraction of the day's low–high range | Sigmoid → [low, high] |
| `sell_all_price_frac` | Sell-all limit price as a fraction of the day's range | Sigmoid |
| `sell_qty` | Shares to sell at next-day open (market order) | ReLU |

Each industry portfolio is initialised at $25,000 ($300,000 total across 12 sectors).

### Master allocator stack (MT1NN + MT2NN, with legacy MasterNN fallback)

Capital allocation uses a two-stage stack trained concurrently with the industry models:

**MT1NN** (per-industry preprocessor, one pool per sector, 3,399 params):
- Input: `(1, 37)` — one industry's slice of a 444-feature master vector (18 delta lookbacks + poly-2 coefs + 4× poly-3 coefs)
- FC: 37→37→29→20→12→3 (ReLU activations, raw logits out)
- Output interpretation: `sigmoid(out[0])` = P(positive return), `tanh(out[1]) × 0.05` = expected delta, `softplus(out[2])` = error half-width
- Activates at `actual_day ≥ 25`

**MT2NN** (cross-industry tier allocator, replaces MasterNN, 33,996 params):
- Input: `(1, 36)` — 3 MT1 outputs × 12 industries (delta/range normalized via Welford running stats)
- Parallel FC branch (36→36→36) + 2-layer LSTM (12 steps × 3 features, hidden=36) → concat 72 → taper 72→66→60→54→48
- Output: `(1, 48)` raw logits → reshape `(12, 4)` → per-industry softmax → argmax → tier ∈ {0,1,2,3}
- Tier 0 = expected net loss (no allocation). Tiers 1/2/3 = positive-return terciles (low→high). Activates at `actual_day ≥ 30`

**MasterNN** (legacy fallback, 599,028 params):
- Flat 5-layer FC: 444→444→444→312→180→48 (ReLU activations, raw tier logits out)
- Same `(12, 4)` → tier decoding as MT2NN. Used when `mt2_best.pt` is absent or norm stats unavailable.

Capital allocation from tiers: positive-tier industries are divided into terciles weighted 1:1.5:2.25 (tier 1:2:3). Tier-0 industries receive $0. Industries with 3+ consecutive tier-0 predictions are fully liquidated.

### Evolutionary training

Each training day proceeds as follows:

1. **Reset** — All 200 model slots in a pool are reset to slot 0's portfolio, providing an identical starting point. Only trading decisions contribute to the score; held positions from prior days are inherited equally by all slots.
2. **Infer** — Each model runs a forward pass using the shared input tensors for the current day.
3. **Trade** — Limit and stop orders are simulated against next-day OHLCV (orders placed end-of-day N, filled on day N+1).
4. **Score** — Portfolio value is computed at fill-day close prices. The baseline (no trades, slot 0's position) is valued on the same day, so the delta measures trading alpha only — not overnight price movement.
5. **Selection scoring** — For slots with a positive delta, the raw delta is multiplied by `invested_pct` (fraction of portfolio deployed). This penalises slots that earned gains while holding large cash reserves and rewards efficient capital deployment.
6. **Select + mutate** — The top 17 performers become direct elites (slots 0–16). Three weighted-average slots (top-5, top-10, top-15 blends) occupy slots 17–19. The remaining 180 slots are Gaussian-noise mutations of the 20 parent slots (9 mutations each). When a streak of fully-filtered days is detected, the bottom half of elites are replaced with halfway blends for diversity injection.

### Fill simulation

Orders placed at end-of-day N are filled against next-day OHLCV:

| Order type | Fill condition | Fill price |
|-----------|----------------|------------|
| Buy limit | `nd_open ≤ buy_price` | `nd_open` |
| Buy limit | `nd_low < buy_price < nd_high` | `buy_price × 1.001` (slippage) |
| Sell-all limit | `nd_open ≥ sell_all_price` | `nd_open` |
| Sell-all limit | `nd_low < sell_all_price < nd_high` | `sell_all_price × 0.999` (slippage) |
| Partial sell | always | `nd_open` (market order, no slippage) |
| Stop-loss | `nd_low ≤ stop_price` | `stop_price × 0.999` (slippage) |

Strict boundary conditions apply: a buy limit at exactly `nd_low` does not fill.

### Diagnostics and soft flags

When an elite candidate holds ≥ 50% cash (industry) or ≥ 80% cash (master), an `UNDER_INVEST` soft flag is emitted and the day's scores, prices, and fill data are written to `data_dump/day_N/` for offline analysis. Hard flags trigger when a single-day gain exceeds 12.5% of the baseline portfolio value.

### End-of-training ranking

After all training passes complete, a summary table is printed showing per-industry: start value, end value, delta, percentage return, positive/zero/negative day ratio, minimum 3-stock entry cost (most expensive stock + 2 cheapest at end-of-training prices), and top stock names. Use the pos:zero:neg ratio and minimum entry cost to prioritise which industries to activate first in production.

---

## Quick Start

### 1. Install dependencies

```bash
bash install_python.sh
source .venv/bin/activate
pip install keyring
```

### 2. Download historical data

```bash
source .venv/bin/activate
python download_5y_data.py
```

Saves approximately five years of daily OHLCV JSON for all 144 symbols under `stock_data/`.

### 3. Build and train models

Build the C++ trainer (one-time):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
```

Seed the C++ trainer from any existing Python `.pt` checkpoints (or skip on first run):
```bash
python prepare_models.py --load-dir models/training --output models/training
```

Full training run (canonical — industry + MT1 + MT2):
```bash
./build/training_v4_cpp --output models/training --load-dir models/training \
  --passes 5 --sigma 0.008 --master-sigma 0.006 --sigma-decay 1.0 \
  --start-day 17 --stop-day 1255
```

After training, convert `.bin` weights back to `.pt` for Python tools:
```bash
python convert_weights.py --models-dir models/training --output models/training
```

Short diagnostic run (verifies history accumulates, days 16–37 only):
```bash
mkdir -p /root/diag_logs
./build/training_v4_cpp --output /root/diag_logs --load-dir /root/diag_logs \
  --start-day 16 --stop-day 37 --passes 1 --preserve-stock-data --no-save
```

### 4. Inspect trade decisions

Examine what the elite models did on a specific day, including fill-condition explanations, gain attribution, and cash utilisation:

```bash
# By calendar date
python inspect_trades.py --industry energy --date 2024-01-10 --models-dir ./models

# By training day index (matches --start-day / --stop-day used during training)
python inspect_trades.py --industry energy --day-index 17 --models-dir ./models --top-n 5
```

### 5. Paper trading

```bash
python production_v2.py --paper --model-dir models
```

### 6. Live trading

```bash
python production_v2.py --model-dir models
```

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` set in the environment (or stored via `keyring`).

---

## CLI Reference

### `training_v4_cpp` (C++ binary)

The canonical trainer — handles industry, MT1, and MT2 evolution with ~6× speedup over Python.

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | *(required)* | Directory to write `.bin` model files and logs |
| `--load-dir` | None | Seed elite slots from `.bin` models in this directory |
| `--start-day` | 0 | First training day index (0-based, matches sorted stock_data dates) |
| `--stop-day` | end | Last training day index (exclusive) |
| `--passes` | 1 | Number of full passes over the day range |
| `--sigma` | 0.008 | Gaussian mutation sigma for industry models |
| `--master-sigma` | 0.006 | Mutation sigma for MT1/MT2 models |
| `--sigma-decay` | 1.0 | Multiply sigma by this value after each pass (1.0 = no decay) |
| `--workers` | nproc | Number of parallel industry-training threads |
| `--preserve-stock-data` | False | Do not trim `stock_data/` JSON files during training |
| `--no-save` | False | Suppress all model writes (diagnostic runs only) |
| `--master-only` | False | Freeze industry models; evolve MT1/MT2 only |

### `inspect_trades.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--industry` | *(required)* | Industry key (e.g. `energy`, `tech_hardware`) |
| `--date` | — | Calendar date to inspect (`YYYY-MM-DD`) |
| `--day-index` | — | Training log day number (`N` in `Day N/...`) — alternative to `--date` |
| `--models-dir` | *(required)* | Directory containing trained `.pt` files and `top10_meta.json` |
| `--top-n` | 3 | Number of elite models to report (max 10) |
| `--stock-data` | `./stock_data` | Path to historical data directory |
| `--starting-cash` | 1666.67 | Starting cash per portfolio for the audit simulation |

### `prepare_models.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--load-dir` | *(required)* | Directory containing `.pt` elite model files |
| `--output` | *(required)* | Directory to write `.bin` files for the C++ trainer |

### `convert_weights.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir` | *(required)* | Directory containing `.bin` elite model files from C++ training |
| `--output` | *(required)* | Directory to write `.pt` and `_best.pt` files for Python tools |

### `read_mt_log.py`

| Flag | Default | Description |
|------|---------|-------------|
| `log` | *(required)* | Path to `mt_training_log.bin` |
| `--pass` | all | Limit output to a single pass number |
| `--industry` | all | Substring filter for industry names (e.g. `energy`) |

### `production_v2.py`

| Flag | Description |
|------|-------------|
| `--paper` | Route all Alpaca API calls to the paper trading endpoint (orders are submitted to paper, not suppressed) |
| `--model-dir` | Directory containing trained `.pt` model files (default: `models`) |
| `--capital` | Cap total deployed capital regardless of Alpaca account balance |
| `--withdraw` | Request a cash withdrawal of the specified dollar amount |

---

## File Reference

### Core modules

| File | Purpose |
|------|---------|
| `models.py` | `StockNN`, `MasterNN`, `MT1NN`, `MT2NN` class definitions — single source of truth |
| `universe_acct0.py` | 144-symbol universe for acct0 (`INDUSTRIES`, `ALL_SYMBOLS`, `INDUSTRY_NAMES`) |
| `universe.py` | Aggregator: auto-discovers all `universe_acct*.py` and exposes their union |
| `fees.py` | Broker fee constants and `_sell_net()` helper |
| `training_lib.py` | Shared evolutionary functions (fill simulation, selection, history, I/O) imported by `upkeep.py` and `production_v2.py` |
| `upkeep.py` | Single-day evolution for MT1/MT2/StockNN pools — called by `production_v2.py` after each trading day |

### Training

| File | Purpose |
|------|---------|
| `training_v4.cpp` | **Canonical C++ trainer** — industry + MT1 + MT2 evolution, ~6× Python speedup. Build with CMake. |
| `prepare_models.py` | Convert Python `.pt` elite slots → flat float32 `.bin` for the C++ trainer (run before first C++ training) |
| `convert_weights.py` | Convert C++ `.bin` elite slots → `.pt` + `_best.pt` for Python tools (run after C++ training) |
| `read_mt_log.py` | Read and summarize `mt_training_log.bin` produced by the C++ trainer |

### Data and tooling

| File | Purpose |
|------|---------|
| `download_5y_data.py` | One-time bulk download of ~5 years of daily OHLCV data into `stock_data/` |
| `download_daily.py` | Incremental daily update (run at 4:30 PM ET): appends new OHLCV for all acct universes, trims to 1255 days, full fetch for new symbols |
| `cleanup_stock_data.py` | Weekly cleanup: purges `stock_data/` entries for symbols no longer in any `universe_acct*.py` or held in any account |
| `inspect_trades.py` | Audit elite model trade decisions for a given day and industry |
| `swap_symbols.sh` | Guided ticker-replacement: updates `universe_acct0.py` + `universe.json`, cleans old data, downloads new data |
| `swap_symbols.py` | Core replacement logic called by `swap_symbols.sh` |

### Production

| File | Purpose |
|------|---------|
| `production_v2.py` | Daily trading cycle: fetch data → MT1/MT2 allocation → StockNN orders → Alpaca → upkeep evolution |

### Setup

| File | Purpose |
|------|---------|
| `install_python.sh` | Create `.venv` and install all packages (Fedora/Linux) |
| `k8s/setup.sh` | First-time Kubernetes cluster setup: build image, apply manifests, create secret, optionally download data |

---

## Deployment (Docker / Kubernetes)

A single-node Kubernetes deployment (tested with k3s) is provided under `k8s/`.

### First-time cluster setup

`k8s/setup.sh` handles all infrastructure steps end-to-end:

1. Builds and pushes the Docker image
2. Applies the namespace and all persistent-volume / PVC manifests
3. Creates the Alpaca credentials secret (keys passed directly to `kubectl` — never written to disk)
4. Optionally submits the stock-data download job

```bash
./k8s/setup.sh
```

The script is idempotent — safe to re-run to rebuild the image, reconcile manifests, or rotate credentials.

### Run a training job

```bash
kubectl apply -f k8s/job-training.yaml
kubectl logs -n trading -f -l job-name=training-job
```

### Enable the daily production CronJob

```bash
kubectl apply -f k8s/cronjob-production.yaml
```

The CronJob schedule (`"15 20 * * 1-5"`) targets 4:15 pm ET (EDT, March–November).
Adjust to `"15 21 * * 1-5"` for EST (November–March).

---

## Stock Universe

144 symbols across 12 sectors (12 per sector), selected for high beta and volatility:

`tech_hardware`, `tech_software_ai`, `financials`, `consumer_discretionary`, `consumer_services`, `health_care`, `industrials`, `consumer_staples`, `energy`, `utilities`, `real_estate`, `materials`

Notable symbol changes from the original universe:

| Sector | Removed | Replacement | Reason |
|--------|---------|-------------|--------|
| `financials` | SQ | XYZ (Block Inc) | SQ ticker renamed to XYZ in January 2025 |
| `industrials` | X (US Steel) | BTU (Peabody Energy) | X acquired by Nippon Steel 2025, delisted |
| `utilities` | NOVA (Sunnova) | ARRY (Array Technologies) | NOVA filed Chapter 11 June 2025, delisted |
| `real_estate` | RDFN (Redfin) | OPEN (Opendoor Technologies) | RDFN acquired by Rocket Companies July 2025, delisted |
| `real_estate` | NVR | LGIH (LGI Homes) | NVR share price (~$6,800) is untradeable at small account sizes |

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
Commercial use requires a separate agreement — see [COMMERCIAL.md](COMMERCIAL.md) or contact the project owner.

## Contributing

Contributions are welcome. By submitting a pull request you agree to the terms of the
[Contributor License Agreement](CLA.md). See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the attribution log.

---

## Notes

- `models/`, `stock_data/`, and `data_dump/` are git-ignored (large binary, data, and diagnostic files).
- Alpaca limit orders do not support fractional share quantities. The system intentionally uses limit orders to preserve price-signal alignment with training. Minimum capital per industry should be sufficient to purchase at least one full share of the sector's most expensive stock.
- In production, industries whose allocated capital falls below the single-share floor for their most expensive stock are skipped for new buy orders (liquidation orders still execute). Industries activate automatically once the capital floor is met through master rebalancing.
- `training_log.csv` records slot 0's (the production model's) pre-selection trading outcomes — not the best-of-200 result.
- Always validate with `--paper` before enabling live order submission.
