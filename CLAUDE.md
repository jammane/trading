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

# Store Alpaca credentials per account+mode (paper and live keys are separate Alpaca accounts)
kubectl create secret generic alpaca-credentials-acct0-paper \
    --namespace trading \
    --from-literal=ALPACA_API_KEY="PK..." \
    --from-literal=ALPACA_SECRET_KEY="..." \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic alpaca-credentials-acct0-prod \
    --namespace trading \
    --from-literal=ALPACA_API_KEY="AK..." \
    --from-literal=ALPACA_SECRET_KEY="..." \
    --dry-run=client -o yaml | kubectl apply -f -
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

**Download training data (initial / full reset):**
```bash
python download_5y_data.py
```

**Daily incremental update (run automatically at 4:30 PM ET via cron):**
```bash
python download_daily.py
```
Appends new trading days for all universe symbols; does a full 5-year fetch for any symbol whose file is missing (new symbol after a swap). Trims each file to the most recent 1255 days. Uses the same shared `stock_data/` directory as the C++ trainer — no truncation.

**Stock data cleanup (run automatically weekly via cron):**
```bash
python cleanup_stock_data.py           # live removal
python cleanup_stock_data.py --dry-run # preview only
```
Removes `stock_data/<SYM>.json` for any symbol not in the current universe AND not held in any open Alpaca position (checked via `models/acct*/*/state.json`). Safe to run at any time.

**Train (C++ binary — canonical; handles industries, MT1, and MT2):**
```bash
# Seed once from existing Python models (or after any convert_weights.py run):
python prepare_models.py --account acct0
# Full training run (canonical settings — industries + master):
./build/training_v4_cpp --account acct0 \
  --passes 5 --sigma 0.008 --master-sigma 0.006 --sigma-decay 1.0 \
  --start-day 17 --stop-day 1255 2>&1 | tee logs/acct0/training.log
# Retrain master only (freeze industries, use their slot-0 perf for ind_val_hist):
./build/training_v4_cpp --account acct0 \
  --master-only --passes 5 --start-day 17 --stop-day 1255 2>&1 | tee logs/acct0/training.log
# Short diagnostic (verifies history accumulates at day 5+, CSV has elite columns):
mkdir -p /root/diag_logs
./build/training_v4_cpp --output /root/diag_logs --load-dir /root/diag_logs \
  --start-day 16 --stop-day 37 --passes 1 --preserve-stock-data --no-save
# After training, convert back to .pt before inspect_trades.py or production_v2.py:
python convert_weights.py --account acct0
```
`--no-save` suppresses all model writes (industry elites, history, master, MT1, MT2).
Always use real disk paths (`models/acct0/training`, `logs/`, `/root/diag_logs`) — never `/tmp` which is a 978 MB RAM-backed tmpfs on the droplet. Training and production can run concurrently; both write to real disk only.
MT1 trains via direction/delta/range scoring starting at `actual_day >= 25`; MT2 trains via tier-classification starting at `actual_day >= 30`.
`convert_weights.py` is required after C++ training before using `inspect_trades.py` or `production_v2.py`.
Note: existing master `.bin` files are incompatible after the MT1/MT2 architecture change — regenerate with `prepare_models.py`.

**Inspect MT1/MT2 training log:**
```bash
python read_mt_log.py logs/acct0/training/mt_training_log.bin
python read_mt_log.py logs/acct0/training/mt_training_log.bin --pass 2
python read_mt_log.py logs/acct0/training/mt_training_log.bin --industry energy
```

**Inspect elite model trade decisions:**
```bash
python inspect_trades.py --industry energy --date 2024-01-10 --account acct0
python inspect_trades.py --industry energy --day-index 17 --account acct0 --top-n 5
```

**Paper / live trading (per-account isolation):**

Each Alpaca account gets its own directory tree: `models/acct#/[training|paper|prod]`.
`--account acct0` selects the account; `--paper` selects subtype (paper vs prod).
All per-run files (`state.json`, `owners.json`, `master_state.json`, `*.pt`) live under
`models/ACCOUNT/paper|prod/`; internal logs under `logs/ACCOUNT/paper|prod/`.

Current account is `acct0`. Up to 5 accounts planned; additional accounts will likely
require a droplet upgrade.

**Scheduling convention (user is US Eastern Time, EDT=UTC-4, EST=UTC-5):**
- Market close: 4:00 PM ET
- acct0 paper: 1h05m after close = 5:05 PM ET (21:05 UTC EDT / 22:05 UTC EST)
- acct0 prod:  30 min after paper = 5:35 PM ET
- acct1 paper: acct0 paper + 1h = 6:05 PM ET
- acct1 prod:  30 min after acct1 paper = 6:35 PM ET
- (each additional account adds 1 hour to paper time, prod is always 30 min after paper)
- **Cron times shift by 1 hour in November (EDT→EST) and March (EST→EDT)**

```
# crontab on droplet (times in UTC, summer/EDT — shift +1h in Nov, -1h in Mar)
# Stock data: shared across all accounts; download once, cleanup weekly
30 20 * * 1-5 cd /root/trading && mkdir -p logs/data && source .venv/bin/activate && python download_daily.py >> logs/data/download_daily.log 2>&1
0  4  * * 0   cd /root/trading && mkdir -p logs/data && source .venv/bin/activate && python cleanup_stock_data.py >> logs/data/cleanup_stock_data.log 2>&1
# acct0 paper trading: 21:05 UTC (5:05 PM EDT); prod: 21:35 UTC
5 21 * * 1-5  cd /root/trading && mkdir -p logs/acct0 && export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d) && export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d) && source .venv/bin/activate && python production_v2.py --paper --account acct0 >> logs/acct0/paper.log 2>&1
35 21 * * 1-5 cd /root/trading && mkdir -p logs/acct0 && export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d) && export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d) && source .venv/bin/activate && python production_v2.py --account acct0 >> logs/acct0/prod.log 2>&1
# acct1 (future): 30 21 download_daily if diff universe; 5 22 paper, 35 22 prod
# acct2 (future): 5 23 paper, 35 23 prod
```

```bash
# Manual run (paper)
export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d)
export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d)
python production_v2.py --paper --account acct0

# Manual run (live, future)
export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d)
export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d)
python production_v2.py --account acct0
```

Training output (`training_v4_cpp`) writes to `models/acct#/training`; after training run
`python convert_weights.py --account acct0`, then copy `_best.pt` files to `.../prod`.

**Output directory convention:** `output_type/acct#/[subtype/]files` — applies to both `models/` and `logs/`:
- `models/acct0/training/` — C++ trainer model weights (`.bin`, `.pt` after convert_weights.py)
- `models/acct0/paper/` — paper trading state (`state.json`, `owners.json`, `*.pt`)
- `models/acct0/prod/` — live trading state
- `logs/acct0/training/` — training CSV + MT binary log (`training_log.csv`, `mt_training_log.bin`)
- `logs/acct0/training.log` — training stdout (tee'd from tmux)
- `logs/acct0/paper.log` — production_v2.py stdout (crontab redirect)
- `logs/acct0/paper/` — internal logs (`request_ids.log`, `data_fetch_failures.log`, `data_dump/`) via `LOG_DIR` (set from `--account`+`--paper`)
- `logs/acct0/prod.log` / `logs/acct0/prod/` — same pattern for live

**Replace ticker symbols (full guided workflow):**
```bash
./swap_symbols.sh '{"OLDTICKER": "NEWTICKER"}'
```
Runs all five steps: updates `universe.py` and regenerates `universe.json`, removes the old symbol's `stock_data/` JSON, runs `download_daily.py` (full 5-year fetch for the new symbol, incremental for all others), prompts to rebuild the Docker image, and prints optional model-cleanup commands for the droplet. The C++ binary reads `universe.json` at startup — no recompile needed after a symbol swap. Run locally — not inside a container.

**Symbol swap thresholds (checked ~monthly — expect 1-2 swaps/month):**
- **$15 watch floor** — symbol goes into `universe_watchlist.json` `"watch"` section with a candidate. Do NOT download candidate data yet. Do NOT run `swap_symbols.sh`.
- **$10 swap floor** (or defunct/halted ticker) — perform the swap: run `swap_symbols.sh`, which downloads candidate data. The removed symbol's open positions are auto-liquidated via market order on the next `production_v2.py` run (orphaned-position logic). Update the watchlist accordingly.
- **5-day new-symbol hold** — after any swap, `production_v2.py` automatically detects the new symbol (compares universe to `state['known_symbols']`) and applies a 5-run hold: paper/prod orders are suppressed for that symbol for 5 trading days. Training (regular C++ and daily upkeep) still runs on the new symbol immediately so the model starts adapting.

## Shared modules

| Module | Contents |
|--------|----------|
| `models.py` | `StockNN`, `MasterNN`, `MT1NN`, `MT2NN` — single source of truth for all model classes |
| `universe.py` | `INDUSTRIES` dict, `ALL_SYMBOLS`, `INDUSTRY_NAMES` — 144-symbol universe |
| `universe.json` | Auto-generated from `universe.py`; read by the C++ trainer at runtime |
| `fees.py` | Fee constants (`BUY_FILL`, `SEC_FEE_RATE`, etc.) and `_sell_net()` helper |
| `training_lib.py` | Shared evolutionary functions (`step_industry`, `selection_and_mutation`, `build_master_features`, I/O helpers, constants) — imported by `upkeep.py` and `production_v2.py`; not a standalone training script |
| `prepare_models.py` | `.pt` → `.bin` for C++ trainer (run before first C++ training) |
| `convert_weights.py` | `.bin` → `.pt` + `_best.pt` for Python tools (run after C++ training) |
| `download_daily.py` | Incremental daily OHLCV update — appends new days, full fetch for new symbols, trims to 1255 days |
| `cleanup_stock_data.py` | Weekly purge of `stock_data/` files for symbols no longer in any universe or held position |

`production_v2.py`, `upkeep.py`, and `inspect_trades.py` import from these modules. (`training_v2.py` and `training_v3.py` were deleted — superseded by `training_v4_cpp` for all training and `upkeep.py` for daily evolution.) `download_5y_data.py` imports from `universe.py`. To add or change a ticker, run `swap_symbols.sh` — it updates both `universe.py` and `universe.json` together.

## Tests

95 pytest tests across three files in `tests/`:
- `test_models.py` — output shapes, output constraints (ReLU/sigmoid/softmax), serialization roundtrip, inject-layer growth dimensions; MT1NN/MT2NN shape + activation + forward tests
- `test_universe.py` — industry count, symbols per industry, no duplicates, formatting
- `test_fees.py` — fee constant values, `_sell_net` calculations, FINRA cap boundary

A `PreToolUse` hook in `.claude/settings.json` runs the suite automatically before every `git commit` or `git push`. Failures are reported before the commit runs, so Claude can self-correct without creating a broken commit. The changelog hook remains `PostToolUse` (it needs the commit hash to exist before it can amend).

## Architecture

### Models

All model classes are defined in `models.py` (single source of truth) and imported everywhere:

- **`StockNN`** — one instance per industry sector (12 sectors). FC injection architecture: seed day → 14 inject layers → today layer → 2 flat layers → funnel. Output is `(12, 4)` — one row per stock in the sector, columns are `[buy_qty, buy_price_frac, sell_all_price_frac, sell_qty]`.
- **`MasterNN`** — legacy single cross-sector allocator (444→48). Kept for backward compatibility; superseded by MT1+MT2 in production once MT2 models are available.
- **`MT1NN`** (37→3, 3,399 params per slot) — per-industry preprocessor. One independent 200-slot pool per industry. Input: one industry's 37-feature slice of the 444-feature master vector. Output (after activation): `sigmoid(out0)` = confidence P(positive return), `tanh(out1)*0.05` = expected delta, `softplus(out2)` = error range half-width. Activates at `actual_day >= 25`. Files: `mt1_{industry}_model_{n}.pt` / `mt1_{industry}_best.pt`.
- **`MT2NN`** (FC+LSTM→48, 33,708 params per slot) — cross-industry allocator. Replaces `MasterNN`. Input: 36 normalized MT1 slot0 outputs (3 per industry × 12 industries, sequence arranged for LSTM as 12 steps × 3 features). Parallel FC branch (36→36→36) + 2-layer LSTM (hidden=36) → concat 72 → taper (72→66→60→54→48). Activates at `actual_day >= 30`. Files: `mt2_model_{n}.pt` / `mt2_best.pt` / `mt2_norm_stats.json`.

### MT1 scoring formula

```
conf = sigmoid(out[0]);  delta = tanh(out[1])*0.05;  range_hw = softplus(out[2])
score_direction = 1.0 if (conf>=0.5) == (actual>=0) else 0.0
score_range     = exp(-range_hw/0.02) if |actual-delta|<=range_hw else 0.0
score_accuracy  = max(0, 1 - clip(|actual-delta|/(|actual|+1e-9), 0, 1))
mt1_score       = 0.50*score_direction + 0.33*score_range + 0.17*score_accuracy
```

### MT2 input normalization

Confidence stays as-is (∈[0,1]). Delta and range are normalized via running Welford mean/variance accumulated across all 12 industries per training day. Stats persisted as `mt2_norm_stats.json` (Python) / `mt2_norm_stats.bin` (C++, 4 doubles + 1 int = 36 bytes). `prepare_models.py` and `convert_weights.py` handle conversion in both directions.

### Production inference chain (when MT2 models available)

```
build_master_features() → today444
  → MT1 slot0 ×12 (slices today444[i*37:(i+1)*37]) → 36 raw outputs
  → normalize delta/range via mt2_norm_stats.json
  → MT2 slot0 forward pass → (12,4) logits → argmax per industry → tier map
  → allocation/liquidation (unchanged from MasterNN path)
```

`run_master_allocation()` in `production_v2.py` tries MT2 first (`mt2_best.pt` exists), falls back to MasterNN, then equal allocation.

### Daily upkeep (production_v2.py + upkeep.py)

`upkeep.py` handles single-step evolution in production (one day per run):
- `upkeep_industry()` — calls `step_industry()` with `daily_sigma=UPKEEP_SIGMA` (fixes the silent burst-skip bug in training_v2's upkeep path)
- `upkeep_mt1_industry()` — MT1 selection/mutation + 4 burst passes; bootstraps from `mt1_{ind}_best.pt` if no slot files exist
- `upkeep_mt2()` — MT2 selection/mutation; fires diversity injection when `best_pts < -1`

`UPKEEP_SIGMA = 0.004` (half of full-train 0.008). Four burst passes at sigma/2, /4, /8, /16.

### Evolutionary pool (training)

Each industry maintains **200 model slots** on disk as `.pt` files. The slot layout after every selection step is fixed:
- Slots 0–16: direct elites (rank-ordered, slot 0 is the production model)
- Slots 17–19: weighted-average blends (top-5, top-10, top-15 weights)
- Slots 20–199: Gaussian-noise mutations (9 children per parent, deterministic assignment)

Each training day: all 200 slots reset to slot 0's portfolio → infer → simulate fills → score as `delta × invested_pct` → select + mutate. The `invested_pct` multiplier penalises cash-heavy winners.

**5-day elite history:** After each selection step, the top-7 direct elites (slots 0–6) plus 3 wavg blends (slots 17–19) are saved into a 5-day circular buffer per industry. On each subsequent day these 10×N historical models are re-scored from scratch (same fill simulation, same reference portfolio) and made eligible for re-selection as direct elites — but NOT used as mutation parents unless they win a direct-elite slot. History resets at pass boundaries. Per-industry constants: `HIST_DAYS=5`, `HIST_PER_DAY=10`, `HIST_ELITE=7`, `HIST_WAVG=3`.

History files per industry:
- **C++**: `{ind}_hist.bin` — 2 ints (head, count) + `HIST_DAYS × HIST_PER_DAY × STOCKNN_PARAMS` floats
- **Python**: `{ind}_hist_{day_slot}_{pos}.pt` (up to 50 files) + `{ind}_hist_meta.json` `{"head": 0, "count": 0}`

**`mlock()`**: All hot model weight buffers in the C++ trainer are pinned in RAM via `mlock()` to prevent swap thrashing during inference. Pinned: `elite_buf`, `new_elites`, and `hist_buf` per worker (~315 MB/worker × 2 = 630 MB) plus MasterScratch buffers (~90 MB). Total locked ≈ 720 MB. `mlock()` failure is non-fatal (falls back to swappable). Not applicable to Python trainers.

**Swap file (droplet):** `/swapfile` (2 GB, btrfs-compatible via `chattr +C` + `dd`) is active on the DigitalOcean droplet alongside `/dev/zram0` (1.9 GB), giving ~3.9 GB total swap. To recreate after a rebuild: `truncate -s 0 /swapfile && chattr +C /swapfile && dd if=/dev/zero of=/swapfile bs=1M count=2048 && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab`.

`training_v4_cpp` (C++ binary) is the canonical trainer — handles industry, MT1, and MT2 training with ~6× speedup over Python. Writes `mt_training_log.bin` and `training_log.csv` to `logs/acct0/training/`; read with `read_mt_log.py`. MT2 tier-classification uses 444→12×37 per-industry feature slices fed through MT1, then normalized outputs fed to MT2 (FC + LSTM forward). MT1 activates at `actual_day >= 25`; MT2 at `actual_day >= 30`.

`training_v3.py` (parallel) differs from `training_v2.py` in: 7 worker threads, in-RAM model cache (`_model_cache`), no slippage on limit fills, and slot-level portfolio JSON persisted alongside weights. History candidates in v3 cause the model cache to be invalidated before `selection_and_mutation` (so virtual slot files load from disk); the cache is repopulated on the next day's `load_all_models` call.

### Fill simulation

Orders are placed end-of-day N and filled against next-day OHLCV with `SLIPPAGE_RATE = 0.001` (v2 only). Strict boundary: a buy limit at exactly `nd_low` does not fill.

### Diagnostics and data dump

When an elite holds ≥50% cash (industry) or ≥80% cash (master), an `UNDER_INVEST` soft flag writes JSON to `logs/acct0/paper|prod/data_dump/day_N/<prefix>.json` (production) or `data_dump/` (fallback). A single-day gain >12.5% of baseline raises a `HardFlagError`. Use `inspect_trades.py` to audit flagged days.

### Production cycle

`production_v2.py` runs once per trading day: fetch data from yfinance → run MT1×12 + MT2 (or MasterNN fallback) to rebalance capital → run StockNN per active industry → submit limit/stop orders to Alpaca → perform one upkeep evolution step on yesterday's data (via `upkeep.py`). Alpaca credentials are read from `keyring` or environment variables (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`). All per-account state files (`state.json`, `owners.json`, `master_state.json`, `mt2_norm_stats.json`) live under `--model-dir`, enabling multiple accounts to run independently with different `--model-dir` paths and credentials.

`--paper` routes all API calls to Alpaca's paper trading endpoint — orders are submitted and portfolio state is read from the paper account, giving real paper trading history without risking real money. Omit `--paper` for live trading.

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
| `HIST_DAYS` | 5 | Days of elite history kept per industry |
| `HIST_PER_DAY` | 10 | Models saved per day (HIST_ELITE + HIST_WAVG) |
| `HIST_ELITE` | 7 | Direct elite slots saved to history each day |
| `HIST_WAVG` | 3 | Wavg blend slots saved to history each day |

## Ignored directories

`models/`, `logs/`, `stock_data/`, and `data_dump/` are git-ignored (large binaries, runtime logs, data, and diagnostics).
