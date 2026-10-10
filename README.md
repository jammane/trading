# Trading — Neural Network Stock Trading System

A neural network-based algorithmic trading system. An evolutionary population of FC-injection
models (one pool per industry sector) is trained on historical daily OHLCV data, alongside a race
of candidate capital allocators, and the result is traded through the Alpaca brokerage API —
paper first, live later.

---

> **Development Status**
>
> This project is under active development and is not yet production-ready. Core training is
> operational; paper trading runs with **flat allocation** (an even 1/12 per industry) while the
> allocators are still being qualified, and live trading has not run. Breaking changes to model
> architecture, file formats and CLI flags happen between versions. Use in live trading
> environments is at your own risk.

---

## Highlights

- **A C++ trainer built for the job.** 12 sector pools of 200 networks evolve daily against a
  fill simulator, with hand-written forward and backward passes (no autodiff library), each
  backward pass checked against finite differences, and Python/C++ parity tests that hold the
  two implementations to the same arithmetic.
- **Decisions made by measurement.** Competing allocator designs are raced inside the same
  training loop on identical data, judged against a flat-allocation baseline, and narrowed by
  qualifier rounds whose rules were fixed before the deciding passes finished.
- **Controls throughout.** No-learning control runs, planted-signal and shuffled-target checks
  on every analysis, block-bootstrap error bars for overlapping samples, and pre-registered
  out-of-sample tests. Several early "results" in this project were overturned by exactly these,
  and the write-ups are kept.
- **Production discipline.** Per-account isolation for paper and live trading, whole-share fill
  semantics matching the broker, position limits, a lock that pauses training during the trading
  cycle, and Kubernetes deployment with secrets kept out of the repository.

Built with C++20 (OpenBLAS, CMake), Python 3 (PyTorch, NumPy, pytest, ruff), the Alpaca API,
and k3s on a DigitalOcean droplet.

## Architecture

### Industry sub-models (StockNN)

Each of 12 industry sectors maintains an independent pool of 200 StockNN models (928,825
parameters each). StockNN uses a fully-connected injection architecture — no recurrent layers:

- **Seed** (day 15 of history): 60-feature OHLCV vector → FC → 120
- **Inject** (×14 earlier days): concatenated [prior hidden, day features] → FC, growing 120→190
- **Today**: concatenated [final hidden (190), today features (232)] → FC 422 → 300
- **Flat** (×2): 300 → 300
- **Funnel**: 300 → 237 → 174 → 111 → 48

The `today` vector is 17 features per symbol × 12 symbols, plus 15 cross-symbol aggregates and 13
portfolio-state features. Two of the per-symbol features describe where the close sits in the
day's bar: `close_pos = (4C − 2O − H − L) / (7(H − L))` and
`close_vs_wap = (C − A)/A` with `A = (2O+3C+H+L)/7`.

The 48-element output is reshaped to a 12×4 matrix — one row per stock in the sector:

| Column | Meaning | Activation |
|--------|---------|------------|
| `buy_qty` | Shares to buy | ReLU (0 = no order) |
| `buy_price_frac` | Buy limit price as a fraction of the day's low–high range | Sigmoid → [low, high] |
| `sell_all_price_frac` | Sell-all limit price as a fraction of the day's range | Sigmoid |
| `sell_qty` | Shares to sell at next-day open (market order) | ReLU |

Each industry portfolio is initialised at $25,000 ($300,000 total across 12 sectors).

### Capital allocation: the MT1 race

Which industries get capital is an open question, and it is being settled by a **race** run
inside every training pass rather than by assumption. Every entry sees the same days and the same
200 StockNN order sets per industry, in the same order:

| Axis | Values |
|---|---|
| Architecture | `MT1` (MT1Net, 74 trailing features), `MT1C` (MT1CNet, 146 features including the order set's own intent), `MT1S` (per-symbol nets: summed, seed replicate, shared encoder), `MT1S-stk` (one net per stock), `TANH32` (146→32→8→1 tanh MLP) |
| Search | `evo` (a 200-slot evolutionary pool) or gradient descent (`grad`, `grdC`) |
| Loss | MSE, demeaned MSE (`gdmn`) or ListNet ranking (`grnk`) |
| Carry | `grdC` and `evo` keep their model across passes; `grad` restarts each pass |

Plus `ALLOC bayes`, a discounted Bayesian posterior on each industry's daily return that funds the
top 4 by posterior mean (or by a Thompson draw).

Each entry is scored two ways every pass:

- **Within-day rank** — Spearman correlation between its predictions and the realised P&L across
  the 200 order sets in one industry on one day: can it tell a good order set from a bad one?
- **Allocation vs flat** — fund the 4 industries it rates highest, equal weight, long-only, and
  compare with an even 1/12 book, in basis points per day. Also reported recency-weighted
  (0.995/day, half-life ~138 days) and in 100-day segments.

**Qualifier races.** Each race decides the field for the next one: an entry must beat flat in
**more than 50%** of its counted passes (`race_qualify.py`). A non-qualifier whose mean gap is above
the qualifiers' average is kept as a **contender**. Losers are switched off at runtime with
`--race-skip`; the roster in the code stays complete.

**Every entry's daily choices are logged.** `race_daily.csv` records, for the last 252 days of every
pass, each entry's 12 predictions, the 4 industries it funded and the result, plus every
industry's actual return. `committee_race.py` replays it to test whether a **committee** of entries
voting on the 4 industries beats any single one.

### MT1Net and MT2NN (production allocator path)

**MT1Net** (per-industry predictor, one 200-slot pool per sector, 3,501 params):
- Input: `(1, 74)` — one industry's slice of the 888-feature master vector,
  `[37 market-index ‖ 37 portfolio]`
- Shape: two 37→28 trunks (market ‖ portfolio) → concat 56 → 22 → 10 → 1
- Output: a single raw logit; `tanh(out) × $10,000` is the predicted **next-session book P&L for
  that industry, in dollars**, signed
- The d2 layer takes 23 inputs — 22 from d1 plus one RESERVED slot fed 0.0, held open for
  `(H−L)/A`. Inert by construction, so filling it later changes no dimension or file format.
- Activates at `actual_day ≥ 25`

**MT2NN** (cross-industry tier allocator, 32,844 params):
- Input: `(1, 12)` — one MT1 prediction per industry, signed and unnormalized
- Parallel FC branch (12→36→36) + 2-layer LSTM (hidden 36, walking the 12 industries) → concat 72
  → taper 72→66→60→54→48
- Output: `(12, 4)` logits → per-industry argmax → tier ∈ {0,1,2,3}. Tier 0 = no allocation;
  tiers 1/2/3 are positive-return terciles weighted 1 : 1.5 : 2.25. Three consecutive tier-0 calls
  liquidate an industry.
- Activates at `actual_day ≥ 30`

**MasterNN** (legacy fallback, 599,028 params) is used only when `mt2_best.pt` is absent.

> **Status.** No allocator has yet shown out-of-sample skill, so paper trading runs
> `--flat-allocation` (even 1/12). The race above is how one earns the right to replace it.

### Champions

At every pass boundary each industry's StockNN elites are judged against the standing champion on
slot 0's **recency-weighted mean daily book return** (the same 0.995/day weighting the race uses).
The winner is copied into `champion/` together with every race model trained in that same pass, so
an industry's StockNN and its allocators stay paired. `champion/` is what gets converted and
promoted to paper.

### Evolutionary training

Each training day proceeds as follows:

1. **Reset** — All 200 model slots in a pool are reset to slot 0's portfolio, providing an identical starting point. Only trading decisions contribute to the score; held positions from prior days are inherited equally by all slots.
2. **Infer** — Each model runs a forward pass using the shared input tensors for the current day.
3. **Trade** — Limit and stop orders are simulated against next-day OHLCV (orders placed end-of-day N, filled on day N+1).
4. **Score** — Portfolio value is computed at fill-day close prices.
5. **Selection scoring** — For slots with a positive delta, the raw delta is multiplied by `invested_pct` (fraction of portfolio deployed). This penalises slots that earned gains while holding large cash reserves.
6. **Select + mutate** — The top 17 performers become direct elites (slots 0–16). Three weighted-average slots (top-5, top-10, top-15 blends) occupy slots 17–19. The remaining 180 slots are Gaussian-noise mutations of the 20 parent slots (9 each). A 5-day history of past elites is re-scored daily and can win back an elite slot.

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
python download_5y_data.py
```

Saves approximately five years of daily OHLCV JSON for all 144 symbols under `stock_data/`.
`download_daily.py` (run by cron at 4:30 PM ET) keeps it current.

### 3. Build and train

Build the C++ trainer:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
```

Seed from existing Python `.pt` checkpoints (optional):
```bash
python prepare_models.py --account acct0
```

Full training run (canonical settings):
```bash
./build/training_v4_cpp --account acct0 \
  --passes 5 --sigma 0.008 --master-sigma 0.006 --sigma-decay 1.0 \
  --start-day 17 --stop-day 1255 2>&1 | tee logs/acct0/training.log
```

Experimental runs use `--output DIR` instead of `--account`, so they never overwrite the account's
models or logs. A run directory is ~3 GB; `prune_runs.sh` strips weights from old runs (a run
holding a `KEEP` file is never touched).

Short diagnostic run (`--no-save` trains into `<output>.nosave` and deletes it at exit):
```bash
mkdir -p /root/diag_logs
./build/training_v4_cpp --output /root/diag_logs \
  --start-day 16 --stop-day 37 --passes 1 --no-save
```

Convert the champions to `.pt` for the Python tools:
```bash
python convert_weights.py --account acct0 --industry-dir models/acct0/training/champion
```

### 4. Inspect results

```bash
python read_mt_log.py logs/acct0/training/mt_training_log.bin            # MT1/MT2 per pass
python race_watch.py /root/ht_race_v5/train.log --all --trend            # the race, per pass
python race_qualify.py V3_LOG V4_LOG                                     # who qualifies
python committee_race.py /root/ht_race_v5/race_daily.csv --out cr.csv    # committee voting
python inspect_trades.py --industry energy --date 2024-01-10 --account acct0
```

### 5. Paper trading

```bash
python production_v2.py --paper --account acct0 --flat-allocation
```

### 6. Live trading

```bash
python production_v2.py --account acct0
```

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in the environment (or stored via `keyring`).
On the droplet they come from per-account Kubernetes secrets — see CLAUDE.md.

---

## Testing

28 pytest modules and 6 C++ test binaries (`ctest`), covering model shapes and serialization,
Python/C++ parity, gradient correctness, the fill simulator, production safety limits, log
formats, and the analysis tooling. Analysis code is tested against synthetic data with a known
answer: a planted signal must be found, and noise must not look like one.

```bash
.venv/bin/pytest tests/ -v          # Python (torch-dependent tests need the full environment)
cmake --build build && (cd build && ctest)
ruff check .
```

---

## CLI Reference

### `training_v4_cpp` (C++ binary)

| Flag | Default | Description |
|------|---------|-------------|
| `--account` | — | Derives `models/ACCT/training` and `logs/ACCT/training` |
| `--output` | — | Write models and logs here instead (diagnostic / experimental runs) |
| `--load-dir` | None | Seed from `.bin` models here — only when the working store is empty |
| `--start-day` | 0 | First training day index |
| `--stop-day` | end | Last training day index (exclusive) |
| `--passes` | 1 | Number of full passes over the day range |
| `--sigma` | 0.01 | Mutation sigma for industry models (canonical runs use 0.008) |
| `--master-sigma` | `--sigma` | Mutation sigma for the allocator pools |
| `--mt1-sigma` / `--mt2-sigma` | master-sigma / master-sigma÷6 | Per-stage overrides |
| `--sigma-decay` | 0.5 | Multiply sigma by this after each pass (canonical: 1.0) |
| `--workers` | 2 | Parallel industry-training threads |
| `--master-only` | off | Freeze industry models; evolve the allocators only |
| `--no-save` | off | Train into `<output>.nosave` and delete it at exit |
| `--no-nn-race` | off | Skip every neural allocator (StockNN and the online models only) |
| `--race-skip "A,B"` | none | Switch race entries off at runtime (`ALLOC bayes` for the allocator) |
| `--seed N` | clock | Reproducible run; default draws from the clock, re-seeded every pass |
| `--trade-lock PATH` / `--no-trade-lock` | `/run/trading/trading_active.lock` | Pause between days while production holds the lock |
| `--control-untrained` | off | No-learning control: re-randomise StockNN daily |
| `--control-random` | off | Pass-1 paired control: score random models each day |
| `--drift-study` | off | Phase-3 drift calibration (`--load-dir` + `--drift-scratch`); not yet runtime-validated |
| `--dir-reps` | — | No effect since v0.8.0.0; still accepted so old launch lines run |

### `production_v2.py`

| Flag | Description |
|------|-------------|
| `--paper` | Use Alpaca's paper endpoint (real paper orders, real paper portfolio) |
| `--account` | Account id (default `acct0`); selects `models/ACCT/paper|prod` |
| `--flat-allocation` | Even 1/12 split, skipping MT1/MT2 (paper default for now) |
| `--no-orders` | Run the full cycle but submit and cancel nothing |
| `--capital` | Cap total deployed capital regardless of account balance |
| `--withdraw` | Request a cash withdrawal of this dollar amount |

### `convert_weights.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--account` | `acct0` | Derives `models/ACCT/training` |
| `--source-dir` | account path | Read `.bin` from here instead |
| `--industry-dir` | source dir | Override the source for StockNN elites only (point at `champion/`) |
| `--output-dir` | source dir | Write `.pt` here |

### `prepare_models.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--account` | `acct0` | Convert `models/ACCT/training` `.pt` → `.bin` for the C++ trainer |

### `inspect_trades.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--industry` | *(required)* | Industry key (e.g. `energy`, `tech_hardware`) |
| `--date` | — | Calendar date to inspect (`YYYY-MM-DD`) |
| `--day-index` | — | Training day number — alternative to `--date` |
| `--account` | `acct0` | Account whose models are inspected |
| `--top-n` | 3 | Number of elite models to report |
| `--stock-data` | `./stock_data` | Historical data directory |
| `--starting-cash` | 1666.67 | Starting cash per portfolio for the audit simulation |

### Race and analysis tools

| Command | Purpose |
|---------|---------|
| `read_mt_log.py LOG [--pass N] [--industry S] [--per-day]` | Summarize `mt_training_log.bin` |
| `read_mt1_dataset.py DATASET [--pass N]` | Read and verify `mt1_dataset.bin` |
| `mt1_analysis.py DATASET [--horizons ...]` | Horizon / formulation study with block-bootstrap error bars |
| `race_watch.py [LOG] [--all] [--trend]` | Per-pass race table (default: newest `/root/ht_race_v*`) |
| `race_qualify.py V3_LOG V4_LOG` | Apply the qualifier rule; prints `RACE_SKIP=` |
| `race_daily_check.py CSV ROWS BAYES` | Consistency gate on `race_daily.csv` |
| `committee_race.py CSV [--sizes 3,4,5,6] [--pool ...] [--watch]` | Committee voting race; `--watch` for the cross-pass watch list |
| `plot_training.py [--download] [--pass N]` | SVG training charts from the droplet's logs |

---

## File Reference

### Core modules

| File | Purpose |
|------|---------|
| `models.py` | `StockNN`, `MasterNN`, `MT1Net`, `MT1CNet`, `MT2INet`, `MT2NN` — single source of truth |
| `universe_acct0.py` | 144-symbol universe for acct0 |
| `universe.py` | Aggregator over all `universe_acct*.py` |
| `fees.py` | Broker fee constants and `_sell_net()` |
| `training_lib.py` | Shared evolutionary functions used by `upkeep.py` and `production_v2.py` |
| `upkeep.py` | Single-day evolution run by production after each session |
| `version.py` | Project version (mirrored as `TRAINER_VERSION` in the C++) |

### Training (C++)

| File | Purpose |
|------|---------|
| `training_v4.cpp` | **Canonical trainer** — StockNN, MT1/MT2, the race, the allocation backtest |
| `mt1_pool.h` | MT1 scoring and pool lifecycle |
| `mt1_backprop.h` | Hand-coded gradient MLP (TANH32 entries) |
| `mt1_grad.h` | Backward passes for the race architectures and the per-stock model |
| `pass_seeding.h` | Pass-boundary champion judging and seed blending |
| `bayes_online.h` | Online conditional models trained inside the pass |
| `sort_util.h` | Small fixed-buffer index sort |

### Training support (Python / shell)

| File | Purpose |
|------|---------|
| `prepare_models.py` | `.pt` → `.bin` for the C++ trainer |
| `convert_weights.py` | `.bin` → `.pt` + `_best.pt` after training |
| `read_mt_log.py`, `read_mt1_dataset.py` | Readers for the trainer's binary logs |
| `race_qualify.py`, `race_watch.py`, `race_daily_check.py`, `committee_race.py` | Race tooling (above) |
| `prune_runs.sh` | Strip StockNN weights from old run directories, keeping logs and `KEEP` runs |
| `check_determinism.py` | Does `--seed` reproduce a run exactly? |
| `plot_training.py`, `plot_png.py` | Training charts |

### Studies

One-shot analyses kept because each carries a control or correction that was expensive to find:
`mt1_analysis.py`, `net_calibration.py`, `race.py`, `oos_vol.py`, `pnl_conditional.py`,
`pnl_persistence.py`, `depth_sweep.py`, `dyn_tree.py`, `living_bn.py`, `stocknn_attribution.py`,
`fill_reconciliation.py`, and the scripts under `analysis/` (see `analysis/README.md`).

### Data and tooling

| File | Purpose |
|------|---------|
| `download_5y_data.py` | One-time bulk download into `stock_data/` |
| `download_daily.py` | Daily incremental update, trims to 1255 days |
| `cleanup_stock_data.py` | Weekly purge of symbols no longer in any universe or position |
| `inspect_trades.py` | Audit elite trade decisions for a day and industry |
| `swap_symbols.sh`, `swap_symbols.py` | Guided ticker replacement |

### Production and setup

| File | Purpose |
|------|---------|
| `production_v2.py` | Daily cycle: fetch → allocate → StockNN orders → Alpaca → upkeep |
| `install_python.sh` | Create `.venv` and install packages (Fedora/Linux) |
| `k8s/setup.sh` | First-time Kubernetes setup |

### Design notes

| File | Status |
|------|--------|
| `CLAUDE.md` | The detailed, current reference — read this before changing anything |
| `PASS_SEEDING.md` | Pass-boundary seeding design; judging metric superseded (see its header) |
| `MT1_RANGE_FEATURE.md` | `(H−L)/A` feature, staged for MT1's reserved input |
| `PHASE3_DRIFT_STUDY.md` | Drift-study design; implemented, not yet run |
| `HEADS_TAILS_PLAN.md` | Superseded (v0.8.0.0); kept for its reasoning |

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

The CronJob schedule (`"15 20 * * 1-5"`) is in UTC and targets 4:15 PM EDT; use
`"15 21 * * 1-5"` under EST. The droplet itself schedules paper and prod from the host crontab
(5:05 PM and 5:35 PM ET, timezone-aware) — see CLAUDE.md.

---

## Stock Universe

144 symbols across 12 sectors (12 per sector):

`tech_hardware`, `tech_software_ai`, `financials`, `consumer_discretionary`, `consumer_services`, `health_care`, `industrials`, `consumer_staples`, `energy`, `utilities`, `real_estate`, `materials`

The universe is **price-banded** (re-normalized 2026-09-17). Fills are simulated in whole shares,
as Alpaca executes them, so a symbol's price decides how many distinct position sizes it offers;
the original universe spanned 73× within a single industry, and its most expensive names could
not be bought at all at paper capital. Each industry now targets a mean price of ~$60 with every
symbol between $30 and $90 (at most 3× apart). Within the band, symbols were chosen for the
largest mean daily range — the strategy earns from intraday movement — with at least five years
of history and $10M/day median dollar volume, from 185 screened candidates.

Drift is tracked with each industry's price mean/median ratio, which is scale-free and so never
goes stale the way an absolute price floor does. `swap_symbols.sh` performs a replacement, and
production holds orders on any new symbol for its first five sessions.

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
Commercial use requires a separate agreement — see [COMMERCIAL.md](COMMERCIAL.md) or contact the project owner.

## Contributing

Contributions are welcome. By submitting a pull request you agree to the terms of the
[Contributor License Agreement](CLA.md). See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the attribution log.

---

## Notes

- `models/`, `logs/`, `stock_data/`, and `data_dump/` are git-ignored.
- Alpaca limit orders do not support fractional share quantities. The system intentionally uses limit orders to preserve price-signal alignment with training. Minimum capital per industry should be sufficient to purchase at least one full share of the sector's most expensive stock.
- In production, industries whose allocated capital falls below the single-share floor for their most expensive stock are skipped for new buy orders (liquidation orders still execute).
- `training_log.csv` records slot 0's (the production model's) pre-selection trading outcomes — not the best-of-200 result.
- Always validate with `--paper` (and `--no-orders` for a first run) before enabling live order submission.
