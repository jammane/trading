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
The full pytest suite (including `test_models.py`) runs on the droplet where torch is available.
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
  --start-day 16 --stop-day 37 --passes 1 --no-save
# After training, convert back to .pt before inspect_trades.py or production_v2.py:
python convert_weights.py --account acct0
```
**`--preserve-stock-data` was removed in v0.8.1.8.** It was parsed into a variable that
nothing ever read — a documented flag that did nothing. It surfaced as the lone
`-Wunused-but-set-variable` the moment `-Wall -Wextra` was turned on for the trainer target,
which until then carried no warning flags at all.

`--no-save` trains into `<output>.nosave` and **deletes it at exit** — the canonical model
directory is untouched, and no 3 GB run directory is left behind.

**This changed in v0.6.6.0, and the old behaviour was a silent correctness bug.** `--no-save` used
to suppress the writes themselves, which disabled StockNN training entirely: `step_industry`
reloads `elite_buf` and `hist_buf` from disk at the top of **every day**, so with nothing written
they were re-random-initialised daily from the same seed. Measured on the v0.6.0.0-A run:
**297,120 random-init lines = 20 slots × 12 industries × 1238 days**. Nothing learned; the
portfolio grew only from picking the best of 200 fresh random models each day. Any StockNN
"diagnostic" taken under the old `--no-save` measured that, not training — including the
v0.6.0.0-A results and the v0.5-vs-v0.6 comparison built on them.

Note `--no-save` now does the same disk I/O as a normal run, because that I/O *is* training. It is
no longer "free". A run directory is **~3.0 GB**, of which 70% is the StockNN
5-day elite history: `HIST_DAYS × HIST_PER_DAY = 50` full copies of a ~921K-parameter model per
industry (176 MB × 12 = 2.1 GB), plus 851 MB of elites. MT1 heads + tails + MT2 together are under
40 MB, so MT1 work is nearly free on disk — the cost is entirely the StockNN layer. Thirteen
accumulated run directories took the droplet's 58 GB disk to 92% full.
Analysis only needs the logs (`mt_training_log.bin`, `training_log.csv`, `train.log`, 3–5 MB);
weights are only needed to seed a run (`--load-dir`) or convert to `.pt`.

**`--load-dir` is a SEED, consulted only when the working store has nothing** (fixed v0.6.6.0).
It used to be checked *first, every day*, so a run seeded from a populated directory reloaded that
seed daily and never made progress — the same root cause as the `--no-save` bug: neither path had
any notion of "first day only". `/root/prune_runs.sh [KEEP]`
(default 2, `--dry-run` previews) strips weights from all but the N most recent `/root/ht_train*`
runs while preserving every log, and skips a run detected in flight. Run it before a full pass.
Always use real disk paths (`models/acct0/training`, `logs/`, `/root/diag_logs`) — never `/tmp` which is a 978 MB RAM-backed tmpfs on the droplet. Training and production can run concurrently; both write to real disk only.

**Training pauses for production (v0.6.5.0).** The droplet has 2 cores and a training run uses
~150% CPU, so the two compete: measured, training fell from 34 s/day to 1–3 min/day with other
jobs alongside it. Production is the time-sensitive side, so `production_v2.py` writes
`/run/trading/trading_active.lock` (PID on line 1) for the whole cycle — orders *and* upkeep — and
the trainer polls it **between training days**, sleeping 15 s at a time and logging `PAUSED` /
`RESUMED`. Worker threads are parked at that point, so the whole trainer idles.

The default path is hardcoded in both `production_v2.TRADING_LOCK_PATH` and `g_trade_lock` in
`training_v4.cpp` — **keep the two in sync**. It is absolute because the two processes run from
different worktrees (`/root/trading` vs `/root/trading-ht`), and under `/run` (tmpfs) so a reboot
cannot strand a lock. Override with `TRADING_LOCK_PATH=` and `--trade-lock PATH`; disable the
pause entirely with `--no-trade-lock`.

A stale lock is ignored if its **PID is dead** or it is **older than 30 minutes**, and the override
is logged loudly — a crashed production run must not idle training indefinitely. Failure to *write*
the lock never blocks a production run; the only cost is continued CPU competition.
MT1 trains one 200-slot pool per industry against the next session's StockNN P&L, starting at `actual_day >= 25`; MT2 trains via tier-classification starting at `actual_day >= 30`.
`convert_weights.py` is required after C++ training before using `inspect_trades.py` or `production_v2.py`.

**Choosing which models to convert (v0.6.4.1).** `--source-dir DIR` reads `.bin` from anywhere,
`--output-dir DIR` writes `.pt` anywhere, and `--industry-dir DIR` overrides the source for the
StockNN industry elites **only**. That last one exists for the champion store: with pass-boundary
seeding active (see `PASS_SEEDING.md`) the final pass is **not** necessarily the best, and
`models/acct#/training/champion/` holds the per-industry best — but *nothing else*, since master,
MT1 and MT2 are saved to the run root. So the deliverable is:

```bash
python convert_weights.py --account acct0 --industry-dir models/acct0/training/champion
```

**The champion judging metric is known to be too short (open, v0.8.1.0).** Champions are crowned on
slot-0's percent book change over the last `PASS_JUDGE_DAYS = 15` days of a pass. `PASS_SEEDING.md`
records that gate producing **13 dethronings against a null expectation of exactly 13.0** — i.e.
indistinguishable from choosing at random. The mechanism is now understood: daily book volatility
is ~1.5-2%, so a 15-day return has sd ~7% while a real skill difference between two model sets
might be 1.5% over that window — SNR ≈ 0.2, nearly all noise.

Two things that are NOT the problem, recorded because both were asserted and both were wrong: the
window sits at the *end* of a pass, which is the most-trained point, not inside the learning curve;
and market beta largely cancels because every pass is judged on the same calendar days.

One genuine contamination: a hard-floor reset inside the judging window jumps `baseline` from
~$22,400 to $25,000, a spurious **+11.6%** on a metric whose real spread is a few percent. At 83
resets per ~20,000 industry-days that hits roughly 6% of judgements.

The fix is to judge on the **sum of daily book P&L** over the window — exactly $0 on reset days
rather than jumping — and to lengthen the window. This matters beyond seeding: the champion store
is what decides which models get promoted to paper.

Pointing `--source-dir` at `champion/` instead would convert the industries and silently skip
master/MT1/MT2. The script errors out if the industry directory has no elite files at all, and
warns loudly if it holds fewer than 12 industries.
Note: existing master `.bin` files are incompatible after the MT1/MT2 architecture change — regenerate with `prepare_models.py`.
**v0.6.0.0 is BREAKING for StockNN too:** `STOCKNN_PARAMS` changed 921625 → 928825, so every
industry `.bin`/`.pt`, every elite pool and every `{ind}_hist.bin` ring from v0.5.0.0 or earlier is
unloadable. `load_bin` validates by exact element count and falls back to **random init silently**,
so a stale run directory looks like it works. Start from a clean `--output` directory and retrain.

**Mutation success rate (v0.6.7.0).** Each per-day industry line carries `mut_ok=NN%`, and
`training_log.csv` gains a `{ind}_mut_success` column: the fraction of the 180 mutations that beat
**their own parent**. Read-only — nothing selects on it.

It is Rechenberg's 1/5 statistic, and it answers two open questions cheaply:

- **Is sigma in the usable band on the current architecture?** The 0.0055–0.009 range was measured
  many versions and one breaking parameter change ago. Far above 1/5 means steps are too small and
  the pool is degenerate; far below means most mutations are damage and selection is mostly picking
  survivors of noise. The measured pool spread — `best-of-200 +604` vs `worst-of-200 +573`, ~5% of
  a ~$590 common move — is consistent with either, and this statistic separates them.
- **Does the regime-optimal sigma move?** Plot it against market volatility. If it spikes when the
  market shifts (existing weights suddenly wrong, so more mutations help) and falls in calm
  periods, a fixed sigma is leaving value on the table and a sigma ladder has a case. If it is
  stationary, one sigma suffices.

**`--control-untrained` (v0.6.9.0) — the no-learning control.** Re-randomises every industry's 20
StockNN elites at the start of **every day** instead of loading them, so the pool accumulates
nothing while the market data, the fill simulation, selection, scoring and logging stay byte-for-byte
the ordinary path. It measures the baseline that nothing in this repo has ever measured: how much of
a run's portfolio growth comes from **training** versus from the selection mechanism operating on
arbitrary models.

The question is not hypothetical. An accidental version of this ran for months as the `--no-save`
defect (fixed in v0.6.6.0), and it scored **+151.4%** against trained runs' **+127-142%** — the
control *beat* the real thing. Either that was luck in one run, or the scoring cannot distinguish
model quality at all, in which case every pass-level conclusion drawn from portfolio value is
unsupported. One control run per few real runs settles it.

Intended cadence is occasional, not routine — it costs a full run to produce a number that only
means anything next to a trained run over the *same* day range.

```bash
./build/training_v4_cpp --account acct0 --control-untrained \
  --passes 5 --sigma 0.008 --start-day 17 --stop-day 1255
```

Deliberately hard to mistake for a real run, because its output is otherwise indistinguishable:

- A banner at startup and the per-day lines unchanged, so read the banner.
- The CSV is **`training_log_CONTROL.csv`**, not `training_log.csv` — every plotting and analysis
  script reads the latter by exact name, so a control can never be picked up by accident.
- A `CONTROL_UNTRAINED` marker file is written into the output directory.
- Industry elites and history are **not saved** (tomorrow re-randomises rather than loads, so the
  write would only burn ~3 GB of I/O and leave a directory of weights that look trained).
- The pass-boundary champion gate is **skipped** — crowning a champion from random weights would
  write them into `champion/`, where the next real run would seed from them.

Seeding is per-day *and* per-industry (`day × 7919 + ind × 104729`), so the parents genuinely differ
day to day. The accidental `--no-save` version re-drew the *same* models every day, which left a
fixed 200-slot pool that selection could still exploit; this is the cleaner null. MT1/MT2 still
train normally, but on top of a random-StockNN portfolio, so their weights from a control run are
not usable either.

**Inspect MT1/MT2 training log:**

**Record V12 (904 B), header version 12** as of v0.8.0.0 — one record per day, per the rebuilt MT1.
Per industry it carries the prediction, the outcome it was made for, the baseline it had to beat,
the score floor, the pool's score distribution (slot-0 / mean / max / min), the lifecycle stats
(mature, culled, largest lineage, distinct lineages, mean retirement age) and the cumulative
retirement-age histogram.

The V1–V11 parsers are **gone** from both `read_mt_log.py` and `plot_training.py`. Every column
they decoded — four component pools × four stats, `mt1_slot0_act[12][4]`, the OOS activation twin
`mt1_oos_act`, the per-channel `mt1_skill`, `mt1_dir_injected`, `mt1_dir_stats`/`mt1_dir_life` —
names something the rebuild deleted, and no V1–V11 log survives (the old run directories were
cleared when the universe was re-normalized). Both readers now **refuse** an older version rather
than decode it into plausible wrong numbers.

The OOS instrument those versions carried is not replaced because it is no longer needed: every
prediction is parked before its outcome exists, so there is no in-sample twin to compare against.
What it measured before it was retired is worth remembering — 49.54% OOS against 61.09% in-sample,
negative skill in all 12 industries.

`read_mt_log.py` prints, per pass: a **Target** block (mean, sd, mean/sd, % negative, mean baseline
error, floor), an **MT1** block (score slot-0/mean/max/min, `corr`, % of days above 0.5, mature,
lineages), a **Pool lifecycle** block, and an **MT2** block. Read the MEAN column, not `best` —
`best` is max-of-200 and rises with pool size under a null. `corr` is the prediction on day *t*
against the P&L realised on *t+1*, and is the only column selection on noise cannot manufacture.

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

**Scheduling convention (user is US Eastern Time):**
- Market close: 4:00 PM ET
- acct0 paper: 1h05m after close = 5:05 PM ET
- acct0 prod:  30 min after paper = 5:35 PM ET
- acct1 paper: acct0 paper + 1h = 6:05 PM ET
- acct1 prod:  30 min after acct1 paper = 6:35 PM ET
- (each additional account adds 1 hour to paper time, prod is always 30 min after paper)
- DST transitions are handled automatically — droplet timezone is America/New_York

```
# crontab on droplet (times in Eastern Time — DST handled automatically by system timezone)
# Stock data: shared across all accounts; download once daily, cleanup weekly
30 16 * * 1-5 cd /root/trading && mkdir -p logs/data && source .venv/bin/activate && python download_daily.py >> logs/data/download_daily.log 2>&1
0  0  * * 0   cd /root/trading && mkdir -p logs/data && source .venv/bin/activate && python cleanup_stock_data.py >> logs/data/cleanup_stock_data.log 2>&1
# acct0 paper trading: 5:05 PM ET; prod: 5:35 PM ET
#5 17 * * 1-5 cd /root/trading && mkdir -p logs/acct0 && export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d) && export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d) && source .venv/bin/activate && python production_v2.py --paper --account acct0 --flat-allocation >> logs/acct0/paper.log 2>&1
#35 17 * * 1-5 cd /root/trading && mkdir -p logs/acct0 && export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d) && export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d) && source .venv/bin/activate && python production_v2.py --account acct0 >> logs/acct0/prod.log 2>&1
# acct1 (future): 30 16 download_daily if diff universe; 5 18 paper, 35 18 prod
# acct2 (future): 5 19 paper, 35 19 prod
```

**`--no-orders` (v0.6.8.0).** Runs the full cycle — fetch, allocate, decide, and the daily upkeep
training step — but submits nothing to Alpaca and cancels nothing. For catching the models up on
missed sessions without trading, which is step 3 of the paper rollout.

It guards **both** mutating paths: order submission *and* the existing-stop cancellation. Cancelling
a live stop while submitting no replacement would strip protection from a real position, which is
the one destructive thing this mode must not do.

Training is unaffected by the missing fills: `build_primed_portfolios` seeds from real Alpaca
positions, but `upkeep_industry` then **simulates** fills against `day_data`/`next_day_data`, so the
evolution step is driven by market data. The one real consequence is that positions stay flat, so
those runs train buy-side behaviour only and never exercise sell or stop-loss decisions — fine for a
few catch-up sessions, an argument against a long one.

```bash
# Manual run (paper)
export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d)
export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-paper -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d)
python production_v2.py --paper --account acct0 --flat-allocation

# Manual run (live, future)
export ALPACA_API_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_API_KEY}' | base64 -d)
export ALPACA_SECRET_KEY=$(kubectl get secret alpaca-credentials-acct0-prod -n trading -o jsonpath='{.data.ALPACA_SECRET_KEY}' | base64 -d)
python production_v2.py --account acct0
```

**`--flat-allocation` (paper only, v0.6.2.0).** Splits deployed capital evenly across all 12
industries and skips MT1/MT2 inference entirely. In force for paper while MT1/MT2 have no
demonstrated out-of-sample skill, so paper results measure the StockNN layer rather than an
unvalidated allocator. The pre-rebuild MT1 measured 52–53% direction OOS against 81–84% in-sample,
with negative per-channel skill in all 12 industries; the v0.8.0.0 rebuild has **no measurement
yet**, and the flag stays until it has one. **Remove the flag from the paper crontab line once MT1/MT2 are confirmed.**
Prod does not pass it and keeps the MT1/MT2 path.

It is a deliberate override, not a fallback — the "equal allocation" fallback in
`run_master_allocation`'s docstring allocates **$0.00** to every industry, because with no MT2 and
no MasterNN `tier_map` stays all-zero and `tiers_to_alloc` returns zeros (`n_pos == 0`). Nor can
`tiers_to_alloc` express it: it re-ranks positives into terciles weighted 1.0/1.5/2.25, so even a
uniform positive `tier_map` comes back unequal. The flat path also **clears** `zero_counts` — three
consecutive tier-0 readings liquidate an industry's holdings, so letting it accumulate would sell
the book off on day 3.

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
Runs all five steps: updates `universe_acct0.py` and regenerates `universe.json`, removes the old symbol's `stock_data/` JSON, runs `download_daily.py` (full 5-year fetch for the new symbol, incremental for all others), prompts to rebuild the Docker image, and prints optional model-cleanup commands for the droplet. The C++ binary reads `universe.json` at startup — no recompile needed after a symbol swap. Run locally — not inside a container.

**Symbol swap thresholds (checked ~monthly — expect 1-2 swaps/month):**
- **$15 watch floor** — symbol goes into `universe_watchlist.json` `"watch"` section with a candidate. Do NOT download candidate data yet. Do NOT run `swap_symbols.sh`.
- **$10 swap floor** (or defunct/halted ticker) — perform the swap: run `swap_symbols.sh`, which downloads candidate data. The removed symbol's open positions are auto-liquidated via market order on the next `production_v2.py` run (orphaned-position logic). Update the watchlist accordingly.
- **5-day new-symbol hold** — after any swap, `production_v2.py` automatically detects the new symbol (compares universe to `state['known_symbols']`) and applies a 5-run hold: paper/prod orders are suppressed for that symbol for 5 trading days. Training (regular C++ and daily upkeep) still runs on the new symbol immediately so the model starts adapting.

## Shared modules

| Module | Contents |
|--------|----------|
| `version.py` | `VERSION` string — single source of truth for the project version (mirrored as `TRAINER_VERSION` in `training_v4.cpp`) |
| `models.py` | `StockNN`, `MasterNN`, `MT1Net`, `MT2NN` — single source of truth for all model classes |
| `universe_acct0.py` | `INDUSTRIES` dict, `ALL_SYMBOLS`, `INDUSTRY_NAMES` for acct0 (144 symbols) |
| `universe.py` | Aggregator: discovers all `universe_acct*.py`, exposes union for download/cleanup scripts |
| `universe.json` | Auto-generated from `universe_acct0.py` by `swap_symbols.py`; read by the C++ trainer |
| `fees.py` | Fee constants (`BUY_FILL`, `SEC_FEE_RATE`, etc.) and `_sell_net()` helper |
| `training_lib.py` | Shared evolutionary functions (`step_industry`, `selection_and_mutation`, `build_master_features`, I/O helpers, constants) — imported by `upkeep.py` and `production_v2.py`; not a standalone training script |
| `prepare_models.py` | `.pt` → `.bin` for C++ trainer (run before first C++ training) |
| `convert_weights.py` | `.bin` → `.pt` + `_best.pt` for Python tools (run after C++ training) |
| `download_daily.py` | Incremental daily OHLCV update — appends new days, full fetch for new symbols, trims to 1255 days |
| `cleanup_stock_data.py` | Weekly purge of `stock_data/` files for symbols no longer in any universe or held position |

`production_v2.py`, `upkeep.py`, and `inspect_trades.py` import from `universe.py` (the aggregator). (`training_v2.py` and `training_v3.py` were deleted — superseded by `training_v4_cpp` for all training and `upkeep.py` for daily evolution.) To add or change a ticker for acct0, run `swap_symbols.sh` — it updates `universe_acct0.py` and regenerates `universe.json`. To add a new account, create `universe_acct1.py` with the same structure; `universe.py` and the download/cleanup scripts pick it up automatically.

## Tests

pytest tests across the files in `tests/`:
- `test_models.py` — output shapes, output constraints (ReLU/sigmoid/softmax), serialization roundtrip, inject-layer growth dimensions; MT2NN shape/dims/forward tests (MT1Net has its own files); `stock_close_pos` value/identity/scale-free/degenerate-bar tests and `TestTodayLayout` section-offset tests (both must mirror the C++ twins)
- `test_universe.py` — industry count, symbols per industry, no duplicates, formatting
- `test_fees.py` — fee constant values, `_sell_net` calculations, FINRA cap boundary
- `test_imports.py` — every module must import. Added after `production_v2.py` sat unimportable for
  many versions (`load_mt2_norm_stats`/`save_mt2_norm_stats` deleted from `upkeep` in eb70bea with
  the stale import left behind) with no test reaching it. That deferred fix landed in v0.6.2.0 —
  the norm-stats plumbing is gone, `production_v2` is in `MODULES` like every other module, and the
  xfail marker is retired. `download_5y_data` is excluded because it runs its download loops at
  import.
- `test_download_daily.py` — `find_stale_symbols` boundary cases: healthy cohort, the weekend
  false-positive a calendar-based check would produce, the real frozen-ticker shape, threshold
  exclusivity, ordering, empty input, and the single-symbol case that must not flag itself.
- `test_flat_allocation.py` — `--flat-allocation`: even split, positive tiers, `zero_counts`
  cleared (pre-seeded above the liquidation threshold), MasterNN never called, and the two
  contrast cases proving flat cannot be expressed as the no-models fallback or via `tiers_to_alloc`.
- `test_upkeep_mt1.py` — upkeep.py's MT1 path, the one production runs daily. `training_v4.cpp`
  and `upkeep.py` evolve the **same** pool files, so the parity tests compile `mt1_pool.h`'s own
  functions and compare numerically (the technique `test_mt1net_parity.py` uses for the forward
  pass) rather than comparing the two by reading them: every pool constant, `mt1_score`,
  `mt1_pred`, both register means, and the full pool ordering checked pairwise against
  `mt1_slot_better`. Plus the lifecycle (bootstrap, park-then-score-next-run, maturity gate, cull
  and lineage inheritance, deployed model published) and rolling-state persistence.
  Two real defects came out of writing it: the sort key had its two components in the opposite
  order from the C++ (recency-weighted mean is the PRIMARY key, plain mean only the tie-break —
  the swapped version ranks identically on most pairs, so it reads as correct), and
  `production_v2` never persisted the rolling state, which would have left every prediction parked
  and none ever scored. Both are pinned.
- `test_mt1_dataset.py` — `mt1_dataset.bin`: layout pinned against the C++ writer (magic, version,
  feature width, component order, record size derived from the `static_assert`), round-trip of
  every field, the decomposition identity plus a case that breaks it, and forward-window alignment
  — including a direct check that day *t* is never inside row *t*'s window. An off-by-one there
  would leak the present into the prediction and make every horizon look predictable.
- `test_mt1_analysis.py` — the A/B/C harness: no-look-ahead (spying on `ridge_fit`), power (a
  planted signal must be recovered, and a stronger one must read stronger), no-leak (shuffled
  targets give IC ≈ 0, and shuffling must kill a planted signal), and the block bootstrap tested
  against a series with a **known** dependence length — block=1 must recover the iid SE, and a
  block spanning the dependence must give an SE more than 2× larger.
- `test_zip_strict.py` — parallel-array guards for the `zip(..., strict=True)` conversion (ruff
  B905). Bare `zip` truncates silently, so a length mismatch yielded a plausible wrong number
  instead of an error; these assert `ValueError` on mismatch and pin the cross-module industry-list
  invariants. Also pins the known off-by-one in StockNN diversity injection (odd `ELITE_COUNT`
  leaves one inject slot unreplaced) so a deliberate fix surfaces as a failing test.

A `PreToolUse` hook in `.claude/settings.json` runs the suite automatically before every `git commit` or `git push`. Failures are reported before the commit runs, so Claude can self-correct without creating a broken commit. The changelog hook remains `PostToolUse` (it needs the commit hash to exist before it can amend).

## Architecture

### Models

All model classes are defined in `models.py` (single source of truth) and imported everywhere:

- **`StockNN`** — one instance per industry sector (12 sectors). FC injection architecture: seed day → 14 inject layers → today layer → 2 flat layers → funnel. Output is `(12, 4)` — one row per stock in the sector, columns are `[buy_qty, buy_price_frac, sell_all_price_frac, sell_qty]`. The `today` vector is **232** wide as of v0.6.0.0 (was 208): 17 features per symbol × 12 + 15 cross-sym aggs + 13 state. `STOCKNN_PARAMS = 928825` (was 921625; `fc_today` is 422→300).

  **`close_pos` (today feature 15, v0.6.0.0 — BREAKING).** `(4C − 2O − H − L) / (7(H − L))`, the
  intraday position of the close within today's bar; equivalently `2·(C−O)/(H−L) + CLV`. Defined in
  `models.stock_close_pos` and mirrored by `stock_close_pos()` in `training_v4.cpp` — **keep the two
  in sync**; returns 0.0 for a zero-range or invalid bar. Measured within-industry rank IC vs
  next-day targets (144 symbols × 1254 days, one observation per day): overnight gap C→O
  **−0.0371 (t −10.90)**; next intraday O→C **+0.0081 (t +2.54)**, rising to **+0.0129 (t +4.49)**
  after orthogonalizing against the prior-day return, which carries none of it (t −0.28). A strong
  close gaps down overnight and reverts during the next session; the O→C leg is the one the fill
  model can reach. It is additive because `today` already holds O/H/L/C as **raw dollars**, so
  `fc_today` can form the numerator in its first layer but cannot divide — the scale-free
  normalization is what lies outside its span, and nothing else in the normalized block
  (`base+10..14`) describes position within *today's* bar. The denominator is `(H−L)` and not `C`:
  with `/C` the tradeable O→C leg is not significant (t +0.93 vs t +2.54). Rejected alternatives
  (weighted price bases, range features as separate inputs, free-weighted two-input split) are
  recorded in the v0.6.0.0 changelog entry.

  **`close_vs_wap` (today feature 16, v0.6.1.0).** `(C − A)/A` where `A = (2O+3C+H+L)/7` is the
  weighted average price; equals `(4C − 2O − H − L)/(2O + 3C + H + L)`. It **shares its numerator
  with `close_pos`** and differs only in denominator — `7A` versus `7(H−L)`. Since `A` sits close
  to `C` this is effectively the `/C` normalization, which measured *weaker* on the tradeable
  next-intraday leg (within-industry t +0.93 vs +2.54). It is carried anyway because the two
  together encode `(H−L)/A`, the range as a fraction of price: their ratio is exactly `A/(H−L)`,
  a relationship no linear layer can form from either feature alone. Defined in
  `models.stock_close_vs_wap`, mirrored by `stock_close_vs_wap()` in `training_v4.cpp`.

  Slot 16 was held **reserved** (constant `0.0`) in v0.6.0.0 precisely so it could be filled
  later. Filling it in v0.6.1.0 changed **no dimension, no offset, no file format and no
  `STOCKNN_PARAMS`** — v0.6.0.0 and v0.6.1.0 models are binary-compatible, and the reservation is
  what bought that. `models.TODAY_RESERVED` remains as an alias for `TODAY_CLOSE_WAP`.

  Section offsets in `today_arr` are named constants (`STOCK_RESERVED`, `TODAY_PER_SYM`,
  `TODAY_AGG_OFF`, `TODAY_STATE_OFF`) guarded by a `static_assert` — they were bare literals
  (`180`, `195`, `196`) before v0.6.0.0 and are the thing that silently breaks if the per-symbol
  block changes width. The Python side mirrors them as module constants in `models.py`
  (`TODAY_PER_SYM`, `TODAY_CLOSE_POS`, `TODAY_CLOSE_WAP`, `TODAY_AGG_OFF`, `TODAY_STATE_OFF`,
  `TODAY_WIDTH`), pinned by `TestTodayLayout`. A mismatch between the two sides misaligns every
  feature past the first symbol block while staying in bounds — plausible wrong numbers, no crash.
- **`MasterNN`** — legacy single cross-sector allocator (444→48). Kept for backward compatibility; superseded by MT1+MT2 in production once MT2 models are available.
- **`MT1Net`** (74→1, **3,501 params**) — per-industry predictor. One instance per industry, one
  200-slot pool each. Input: that industry's 74-feature slice `[37 market ‖ 37 portfolio]` of the
  888-feature master vector. Output: a single raw logit; `tanh(out) × MT1_PRED_SCALE ($10,000)` is
  the predicted **next-session StockNN P&L in dollars**, signed. Shape: two 37→28 trunks (market ‖
  portfolio) → concat 56 → 22 → 10 → 1. The d2 layer takes **23** inputs — 22 from d1 plus one
  RESERVED slot fed 0.0, held open for `(H−L)/A`; inert by construction, so filling it later
  changes no dimension, offset, file format or `MT1NET_PARAMS`. Layer order is pinned by
  `MT1NET_LAYER_DEFS` in `models.py`, mirrored by the `NT_*` offsets in `mt1_pool.h`, and the two
  are held to the same arithmetic by `tests/test_mt1net_parity.py`, which compiles the C++ forward
  pass standalone and compares it against the torch module. Files: `mt1_{ind}_slot_{n}.bin` /
  `mt1_{ind}_model_{n}.pt`, deployed model `mt1_{ind}_best.pt`, sidecar `mt1_{ind}_meta.bin`.
  Activates at `actual_day >= MT1_START_DAY (25)`.
- **`MT2NN`** (FC+LSTM→48, **32,844 params** per slot) — cross-industry allocator. Replaces
  `MasterNN`. Input: **12** — one MT1 prediction per industry, signed, unnormalized (the dollar
  magnitude IS the allocation signal and the sign is the direction call). Parallel FC branch
  (12→36→36) + 2-layer LSTM (input=1, hidden=36, walking the 12 industries one scalar per step) →
  concat 72 → taper (72→66→60→54→48). Activates at `actual_day >= 30`. Files: `mt2_model_{n}.pt` /
  `mt2_best.pt`. **BREAKING:** the input was 48 (4 channels × 12) and `MT2NN_PARAMS` was 34,572, so
  every pre-v0.8.0.0 MT2 model is unloadable.

### MT1 target and scoring

**BREAKING in v0.8.0.0.** MT1 was one shared dual head plus four specialized tails (9,208 params),
five 200-slot pools per industry, four graded channels, a 10-day recency-weighted replay window,
and a 10-day-forward market target. It is now **one `MT1Net` (3,501 params), one output, one pool
of 200 persistent individuals per industry**, scored daily against the next session.

Every MT1 `.bin`/`.pt`, every pool file, every `mt_training_log.bin` and every MT2 model from
before this change is unloadable. `load_bin` validates by element count and falls back to random
init **silently**, so start from a clean output directory.

**The target (corrected in v0.8.1.0 — read this before trusting any older MT1 number).** MT1
predicts, for one industry, the **book P&L** that industry's StockNN will realise over the next
session, in dollars. Three marks on slot 0 decompose the day:

```
book_prev   = reference book (yesterday's holdings, NO trades today) at TODAY's close
baseline    = the same holdings at the NEXT day's close
slot0_score = the book AFTER today's trades, at the NEXT day's close

market move = baseline    − book_prev      prices moved, holdings fixed
trade delta = slot0_score − baseline       holdings moved, prices fixed
actual      = slot0_score − book_prev      = market + trade          <- the target
pred        = tanh(out) × MT1_PRED_SCALE   MT1_PRED_SCALE = $10,000
```

**The target used to be `slot0_score − baseline`, and that was wrong.** Both sides are marked at
the *same* next-day prices, so the market move on the held book **cancels exactly**, leaving only
the value added by the day's trades. Measured over a full pass that component is **−8% of the
book's gain**, zero-mean and 2.9× heavy-tailed, and directionally unpredictable (same-sign next day
51.9%). It is also not what "how profitable will this industry be" means. Any MT1 result from
before v0.8.1.0 was measured against it.

The same error ran through the **inputs**: `today_pf_val` compounded the trading ratio
(`slot0_score/baseline − 1`), so the portfolio-half features described a curve that ended a pass
near −7% while the book it was supposed to describe ended +80%. MT1 could see neither the book's
level nor the market move on it. It now compounds the book's return, on the same scale as the
market index.

Before that, the target was a 10-day-forward relative return — a proxy, and a poor one: a 10-day
window over a 10-day-forward target overlaps 9-of-10, so a "10-day scoring window" held roughly
**1.6 independent observations**.

**Hard-floor resets.** `step_industry` recapitalises all 200 portfolios to `IND_STARTING_CASH`
whenever the book falls below `IND_STARTING_CASH × 0.9` ($22,500). It fires often — 83 times in
1,666 day-steps of the v0.8.0.0 run. The reset path sets `book_prev == slot0_score`, so such a day
contributes a book P&L of **exactly $0** rather than a spurious +$2,600 jump to the reset level.
That is load-bearing: without it the target acquires a large fake positive tail, and any
reconstruction of the book from `holdings_log.csv` that ignores resets shows ~−0.4 daily
autocorrelation and 6% daily volatility, both impossible for a 12-stock book.

**The horizon is not yet decided.** The trainer's live target is the single-day atom (h=1).
Forward-h P&L is a sum over consecutive days, computed offline from `mt1_dataset.bin`, so the
horizon is a measurement rather than an assumption. See **Deciding the horizon** below.

**The score.**

```
err  = |actual − pred|
base = |actual − baseline|                    # baseline = trailing mean over MT1_BASELINE_DAYS = 20
d    = max(base, floor, 1e-6)
score = d / (err + d)                          # in (0, 1]; 0.5 = tied the trailing mean
floor = mean(|actual| over MT1_FLOOR_DAYS=10) × MT1_FLOOR_FRAC=0.5, and never below $1
```

Bounded, symmetric in the sign of the error, and anchored to a predictor that must actually be
beaten. The floor exists because `base` goes to zero on any day the trailing mean happens to land
on the outcome, which would otherwise make the score hypersensitive — p10 of |actual| is ~$86.

`1/(1+err)` was rejected: with err in dollars it saturates near 0 everywhere, and it weights quiet
days **19× more** than loud ones (p10 $86 vs p90 $1650), so a windowed mean would be dominated by
the days that matter least.

**The causality contract.** A prediction made on day *t* is scored at *t+1* against the realised
*t→t+1* P&L, using a baseline and floor computed from days ≤ *t*.
`mt1_windows_are_causal(scored_day, baseline_last_day, floor_last_day)` asserts it in both the C++
and the Python paths, and `tests/test_mt1_pool.cpp` pins it. One day of delay is the minimum
possible; the old design needed 10, and 20 for the volatility channel.

**Out-of-sample by construction.** Every individual's prediction is parked before the outcome
exists, so there is no in-sample path to correct for. The V10 log's OOS snapshot twin, the
`MT1_SKILL_DAYS=250` skill ring and the block-start freeze all existed to work around selection on
a window that contained the day being scored; all three are gone. What the instrument they
replaced measured is worth keeping in mind: **49.54% OOS against 61.09% in-sample**, negative skill
in all 12 industries.

**Read the MEAN, not the max.** `mt1_score_best` is max-of-200 and rises with pool size under a
pure null. `read_mt_log.py` prints the pool mean heavy and the max thin for that reason, plus
`corr` — the correlation between the prediction on day *t* and the P&L realised on *t+1*, over the
whole window. `corr` is the one column selection on noise cannot manufacture: a pool can hold a
score near 0.5 with corr 0.

**Why one output and not two.** The target is nearly pure noise — measured mean $24.9 against sd
$1121 (mean/sd = 0.022), 49.2% of sessions negative. A correct estimator therefore shrinks hard
toward the mean, which means **|prediction| already encodes conviction**: a model that is sure
predicts far from the baseline, and one that is not predicts near it. A separate confidence output
would be a computed function of the first, and MT2 can compute it. The old four channels made this
concrete: one (conf4) was ungraded and forwarded a constant, one was the magnitude with its sign
deliberately discarded, and one graded itself against another's residual.

**The pool.** 200 persistent individuals per industry, each carrying a rolling
`MT1_SCORE_HIST = 16` score register (`MT1SlotMeta` in `mt1_pool.h`):

```
primary   = mean score over the register
secondary = Σ wᵢ·scoreᵢ / Σ wᵢ      # MT1_RECENCY_W = 1.0,0.8,0.6,0.4 in blocks of 4, newest first
sort key  = (primary, secondary) descending; immature models sort last unconditionally
```

Partial registers normalise by the weight actually occupied, so an 8-prediction model is judged on
its 8. Lifecycle: `MT1_POOL_MIN_AGE = 8` predictions before a model may be culled **or** breed;
`MT1_POOL_CULL_PCT = 0.083` of **mature** models culled per day → ~60% mature, ~20-prediction
lifespan. Parents = the top `MT1_POOL_ELITE_PCT` (10%) of mature models whose lineage is not
barred. Backfill is **flat round-robin**, not the `kChildren` weighted table that handed slot 0
sixteen of 180 children.

Lineage: every slot carries an inherited id; children take the parent's. A lineage above
`MT1_POOL_LINEAGE_CAP` (12.5% = 25 of 200) is barred from breeding until it falls back under
`MT1_POOL_LINEAGE_RESUME` (10%). Logged per day as `lineage_max` / `lineage_n` — the direct read on
monoculture that previously had to be inferred from pool spread.

**No blocks.** `MT1_BLOCK_DAYS = 25` and the T1/H/T2 phase alternation are gone: there is one
network, so there is nothing to alternate between, and with a next-session target there is nothing
to buffer. `MT1_DAYS = 1` survives only as the extent of the per-day staging arrays the CSV and
binary-log writers index.

**Persistence.** All 200 individuals as `mt1_{ind}_slot_{n}.bin`, plus a metadata sidecar
`mt1_{ind}_meta.bin` holding each slot's register, age and lineage, the deployed slot, and the
trailing target window. The sidecar is a **separate file** because `save_bin`/`load_bin` are raw
headerless float arrays validated by exact element count — appending metadata to a weight file
makes the loader reject it and fall back to random init silently. A missing sidecar is handled: the
weights load and the registers start empty, costing `MT1_POOL_MIN_AGE` days of maturity.

### Deciding the horizon, and which layer needs a network

Two open questions are being settled by measurement rather than argument, from one artefact.

**`mt1_dataset.bin`** (v0.8.1.0, 3,752 B/day, ~4.6 MB/pass) is written every run alongside
`mt_training_log.bin`. Per day it carries, per industry, the exact 74 features handed to
`mt1_step_day` plus all four outcome marks (`book_prev`, `book_now`, `mkt_move`, `trade_delta`).
Forward-h P&L for **any** h is a sum over consecutive records, so the horizon is an offline sweep
instead of one ~10 h training run per candidate.

The components are logged separately on purpose: the market move carries the level (+$89/day
measured) and the trade delta is small and negative (−$4.57/day), so logging only the total would
average them and hide it. `read_mt1_dataset.verify_identity` asserts
`book_now − book_prev == mkt_move + trade_delta` on every row — a failure means the three marks in
`step_industry` have drifted apart, which nothing else would catch.

```bash
python read_mt1_dataset.py logs/acct0/training/mt1_dataset.bin
```

**`mt1_analysis.py`** answers three questions from that file:

- **A** — which formulation predicts at all: **per-industry** (12 models, 74→1, 3,501 params each,
  1,238 samples each), **pooled** (one model shared across all 12, 12× the data, no
  specialization), or **cross-sectional** (888→12, MT2's job done directly with no MT1 in between)
- **B** — the horizon h
- **C** — which of MT1/MT2 must be a network. Exactly one needs to be: if MT1 produces good
  per-industry estimates then ranking is `sorted()` and allocation is `tiers_to_alloc`, so MT2 is
  arithmetic; if MT2 maps 888 features to 12 tiers directly then MT1 is redundant. The prior
  favours MT1 on sample efficiency (0.35 samples/param vs 0.038), and MT1's calibration problem —
  twelve independently-miscalibrated estimates rank badly — is fixable programmatically by
  z-scoring each output against its own trailing distribution.

```bash
python mt1_analysis.py logs/acct0/training/mt1_dataset.bin --horizons 1,2,3,5,10,20
```

Three properties it is built around, **each because the opposite has already produced a wrong
answer in this project**:

- **No look-ahead.** A model predicting `[t+1, t+h]` may only be fitted on windows that *closed*
  before `t`, not started before it. A test spies on `ridge_fit` to assert the largest fit still
  stops `h` rows short of the test day at every horizon.
- **Overlap-correct error bars.** Consecutive daily h-day predictions share h−1 days and the 12
  industries move together, so treating T × 12 rows as independent overstates t by roughly
  `sqrt(12h)`. That produced a **t of −7.20 on a signal indistinguishable from zero**. All SEs come
  from a moving-block bootstrap over *time*, block = `max(2h, 20)`, resampling whole blocks with
  every industry attached.
- **Controls on every run.** A shuffled target must give IC ≈ 0 (else the protocol leaks) and a
  planted signal must be recovered (else the harness has no power and its nulls are
  uninformative). An IC of 0.02 means nothing without both.

**Do not read an IC without its block-bootstrap t.** This is the single most repeated mistake in
this project's history.

### Production inference chain (when MT2 models available)

```
build_master_features() → today888
  → MT1 best ×12 (slices today888[i*74:(i+1)*74]) → one raw logit each
  → in12[i] = tanh(logit) × $10,000     # predicted next-session P&L for industry i, signed
  → MT2 slot0 forward pass → (12,4) logits → argmax per industry → tier map
  → allocation/liquidation (unchanged)
```

`run_master_allocation()` in `production_v2.py` tries MT2 first (`mt2_best.pt` exists), falls back
to MasterNN, then equal allocation. The deployed MT1 for each industry is whichever individual its
pool ranked first, published to `mt1_{ind}_best.pt` on every upkeep run — not a fixed slot 0.

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

`training_v4_cpp` (C++ binary) is the canonical trainer — handles industry, MT1, and MT2 training with ~6× speedup over Python. Writes `mt_training_log.bin` and `training_log.csv` to `logs/acct0/training/`; read with `read_mt_log.py`. MT2 tier-classification takes the 12 MT1 predictions directly (FC + LSTM forward, no normalization). MT1 activates at `actual_day >= 25`; MT2 at `actual_day >= 30`.

`training_v3.py` (parallel) differs from `training_v2.py` in: 7 worker threads, in-RAM model cache (`_model_cache`), no slippage on limit fills, and slot-level portfolio JSON persisted alongside weights. History candidates in v3 cause the model cache to be invalidated before `selection_and_mutation` (so virtual slot files load from disk); the cache is repopulated on the next day's `load_all_models` call.

### Fill simulation

Orders are placed end-of-day N and filled against next-day OHLCV with `SLIPPAGE_RATE = 0.001` (v2 only). Strict boundary: a buy limit at exactly `nd_low` does not fill.

### Diagnostics and data dump

When an elite holds ≥50% cash (industry) or ≥80% cash (master), an `UNDER_INVEST` soft flag writes JSON to `logs/acct0/paper|prod/data_dump/day_N/<prefix>.json` (production) or `data_dump/` (fallback). A single-day gain >12.5% of baseline raises a `HardFlagError`. Use `inspect_trades.py` to audit flagged days.

### Production cycle

`production_v2.py` runs once per trading day: fetch data from yfinance → run MT1×12 + MT2 (or MasterNN fallback) to rebalance capital → run StockNN per active industry → submit limit/stop orders to Alpaca → perform one upkeep evolution step on yesterday's data (via `upkeep.py`). Alpaca credentials are read from `keyring` or environment variables (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`). All per-account state files (`state.json`, `owners.json`, `master_state.json`, `mt1_rolling_state.json`, `mt_prev_session.json`) live under `--model-dir`, enabling multiple accounts to run independently with different `--model-dir` paths and credentials.

`--paper` routes all API calls to Alpaca's paper trading endpoint — orders are submitted and portfolio state is read from the paper account, giving real paper trading history without risking real money. Omit `--paper` for live trading.

### Changelog hook

A `PostToolUse` hook in `.claude/settings.json` auto-updates `CHANGELOG.md` and amends the commit whenever Claude makes a `git commit`. This is intentional — do not skip it.

## Key constants (defined at top of each training/production file)

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_SLOTS` | 200 | Total model slots per pool |
| `ELITE_COUNT` | 17 | Direct elite slots (industry + MT2) |
| `ELITE_POOL` | 20 | Elites + weighted-average slots (industry + MT2) |
| `MT1_POOL_SLOTS` | 200 | Persistent individuals per industry (one pool, was five) |
| `MT1_PRED_SCALE` | $10,000 | `tanh(out) × scale` = predicted next-session P&L, in dollars |
| `MT1_BASELINE_DAYS` | 20 | Trailing mean of realised P&L — the predictor MT1 must beat |
| `MT1_FLOOR_DAYS` | 10 | Rolling window behind the score floor |
| `MT1_FLOOR_FRAC` | 0.5 | `floor = mean(\|actual\|) × frac`, per industry, never below $1 |
| `MT1_SCORE_HIST` | 16 | Rolling per-model score register (`MT1SlotMeta`) |
| `MT1_POOL_MIN_AGE` | 8 | Predictions before a model may be culled OR breed |
| `MT1_POOL_CULL_PCT` | 0.083 | Fraction of MATURE models culled per day (→ ~60% mature) |
| `MT1_POOL_ELITE_PCT` | 0.10 | Top fraction of mature models used as parents |
| `MT1_POOL_LINEAGE_CAP` | 0.125 | Lineage above this share of the pool stops breeding (25 of 200) |
| `MT1_POOL_LINEAGE_RESUME` | 0.10 | ...and resumes only below this (hysteresis) |
| `MT1_RECENCY_W` | 1.0/0.8/0.6/0.4 | Register weights, newest-first, in blocks of 4 |
| `MT1_DAYS` | 1 | MT1 steps once per session; vestigial staging-array extent |
| `DS_FEAT` | 74 | Features per industry in `mt1_dataset.bin`; must equal MT1Net's input width |
| `PASS_JUDGE_DAYS` | 15 | Champion judging window. **Known too short** — see the champion note under `convert_weights.py` |
| `--mt1-sigma` | master_sigma | One MT1 mutation sigma (was four per-channel sigmas) |
| `IND_STARTING_CASH` | $25,000 | Per-industry starting capital |
| `MST_STARTING_CASH` | $300,000 | Master starting capital |
| `MAX_SINGLE_STOCK_PCT` | 0.60 | Max fraction of industry cash in one stock |
| `HIST_DAYS` | 5 | Days of elite history kept per industry |
| `HIST_PER_DAY` | 10 | Models saved per day (HIST_ELITE + HIST_WAVG) |
| `HIST_ELITE` | 7 | Direct elite slots saved to history each day |
| `HIST_WAVG` | 3 | Wavg blend slots saved to history each day |

## Versioning

Version string is defined in `version.py` (`VERSION`) and mirrored as `TRAINER_VERSION` in `training_v4.cpp`.

**Scheme (pre-paper-trading):** `0.BREAKING.FEATURE.BUILD`
- `0.` prefix until paper trading performance is deemed acceptable; replaced by `1.` on first production promotion (version then becomes `MAJOR.FEATURE.BUILD`).
- `BREAKING` — increment when backward compatibility is lost (model file format, `state.json` schema, etc.); resets all trailing digits to 0.
- `FEATURE` — increment for any new capability or significant improvement; resets `BUILD` to 0.
- `BUILD` — increment for bug fixes and minor changes within a `FEATURE`.

Current version: **0.8.1.1**

To bump the version, edit `VERSION` in `version.py` and `TRAINER_VERSION` in `training_v4.cpp`, then rebuild the C++ binary.

## Ignored directories

`models/`, `logs/`, `stock_data/`, and `data_dump/` are git-ignored (large binaries, runtime logs, data, and diagnostics).
