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

### Master allocator (MasterNN)

MasterNN uses the same FC injection architecture with a 61-feature history input (60 OHLCV means + 1 flat-cosine regime signal per day) and a 229-feature today vector. Its 36-element output encodes, per industry:

- **Allocation weight** (softmax, sums to 1)
- **Liquidation depth** (sigmoid, 0 = hold, 1 = liquidate to floor)
- **Liquidation trigger** (sigmoid, > 0.5 = execute)

Capital is capped at 40% per industry with a 2% minimum floor enforced at decode time.

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

### 3. Train models

Full run from scratch (recommended — 2-process parallel):
```bash
python training_v4.py --output models
```

Continue from an existing checkpoint:
```bash
python training_v4.py --output models --load-dir models
```

Short diagnostic run (days 17–22 only):
```bash
python training_v4.py --output models --preserve-stock-data --start-day 16 --stop-day 21 --passes 1
```

For single-process training on memory-constrained hardware, use `training_v2.py` with the same flags.

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

### `training_v4.py` / `training_v2.py`

`training_v4.py` (parallel, 2-thread dynamic industry pool — recommended) and `training_v2.py`
(single-threaded) share identical CLI flags. The thread count in v4 is a source-level constant
(`NUM_THREADS = 2`) rather than a CLI argument.

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | *(required)* | Directory to save trained `.pt` models and metadata |
| `--load-dir` | None | Seed elite slots from models in this directory before training |
| `--start-day` | 0 | First training day index (0-based, relative to sorted `all_data` dates) |
| `--stop-day` | end | Last training day index (exclusive) |
| `--passes` | 1 | Number of full passes over the day range |
| `--sigma` | 0.01 | Initial Gaussian mutation standard deviation for industry models |
| `--master-sigma` | same as `--sigma` | Mutation standard deviation for the master model |
| `--sigma-decay` | 0.5 | Multiply sigma by this value after each pass |
| `--daily` | False | Run 4 burst-refinement passes per day after normal selection |
| `--preserve-stock-data` | False | Do not trim `stock_data/` JSON files after training |
| `--promote` | None | Comma-separated sibling directories to copy best models into after training (e.g. `uat,prod`) |
| `--master-only` | False | Freeze industry models; train master allocator only |

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

### `production_v2.py`

| Flag | Description |
|------|-------------|
| `--paper` | Paper trading mode (no orders submitted) |
| `--model-dir` | Directory containing trained `.pt` model files (default: `models`) |
| `--withdraw` | Request a cash withdrawal of the specified dollar amount |

---

## File Reference

### Core modules

| File | Purpose |
|------|---------|
| `models.py` | `StockNN` and `MasterNN` class definitions — single source of truth |
| `universe.py` | 144-symbol trading universe (`INDUSTRIES`, `ALL_SYMBOLS`, `INDUSTRY_NAMES`) |
| `fees.py` | Broker fee constants and `_sell_net()` helper |

### Training

| File | Purpose |
|------|---------|
| `training_v4.py` | **Recommended.** Parallel training: v2 per-slot loading + dynamic 2-process industry pool |
| `training_v2.py` | Single-process training — lower RAM requirement, identical logic to v4 |
| `training_v3.py` | 7-thread parallel variant using an in-RAM model cache (requires ≥4 GB RAM) |

### Data and tooling

| File | Purpose |
|------|---------|
| `download_5y_data.py` | Download ~5 years of daily OHLCV data into `stock_data/` |
| `inspect_trades.py` | Audit elite model trade decisions for a given day and industry |
| `swap_symbols.sh` | Guided ticker-replacement: updates `universe.py`, cleans old data, downloads new data, prompts to rebuild Docker image |
| `swap_symbols.py` | Core replacement logic called by `swap_symbols.sh` — exact quoted-string match in `universe.py` |

### Production

| File | Purpose |
|------|---------|
| `production_v2.py` | Daily trading cycle: fetch data, infer, submit orders, upkeep training |

### Setup

| File | Purpose |
|------|---------|
| `install_python.sh` | Create `.venv` and install all packages (Fedora/Linux) |

---

## Deployment (Docker / Kubernetes)

A single-node Kubernetes deployment (tested with k3s) is provided under `k8s/`.

### Build and push the Docker image

```bash
docker build -t jammane80/trading:latest .
docker push jammane80/trading:latest
```

### Apply infrastructure manifests

```bash
# Namespace, persistent volumes and claims
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pv-app-state.yaml   -f k8s/pvc-app-state.yaml
kubectl apply -f k8s/pv-stock-data.yaml  -f k8s/pvc-stock-data.yaml
kubectl apply -f k8s/pv-models-training.yaml -f k8s/pvc-models-training.yaml
kubectl apply -f k8s/pv-models-prod.yaml -f k8s/pvc-models-prod.yaml
```

### Create the Alpaca credentials secret

```bash
# Copy the template, fill in real values, apply — never commit the filled file
cp k8s/secret.yaml.template k8s/secret.yaml
# edit k8s/secret.yaml
kubectl apply -f k8s/secret.yaml
```

### Run a training job

```bash
kubectl apply -f k8s/job-training.yaml
kubectl logs -n trading -f -l job-name=training-job
```

### Run a one-off data download

```bash
kubectl apply -f k8s/job-download.yaml
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
