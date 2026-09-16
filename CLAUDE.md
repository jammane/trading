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
All 176 pytest tests (including `test_models.py`) run on the droplet where torch is available.
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
MT1 trains via direction/delta/range scoring starting at `actual_day >= 25`; MT2 trains via tier-classification starting at `actual_day >= 30`.
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

Pointing `--source-dir` at `champion/` instead would convert the industries and silently skip
master/MT1/MT2. The script errors out if the industry directory has no elite files at all, and
warns loudly if it holds fewer than 12 industries.
Note: existing master `.bin` files are incompatible after the MT1/MT2 architecture change — regenerate with `prepare_models.py`.
**v0.6.0.0 is BREAKING for StockNN too:** `STOCKNN_PARAMS` changed 921625 → 928825, so every
industry `.bin`/`.pt`, every elite pool and every `{ind}_hist.bin` ring from v0.5.0.0 or earlier is
unloadable. `load_bin` validates by exact element count and falls back to **random init silently**,
so a stale run directory looks like it works. Start from a clean `--output` directory and retrain.

**Inspect MT1/MT2 training log:**

As of v0.4.1.0 the binary log is one record **per block-day** (was one per 25-day block — 50
points/pass, which is why the MT1 pool collapse was only visible in aggregate), and each record
carries `mt1_actual_d[12]`, the MT1 target, making a run re-gradable offline.

As of v0.4.3.0 it is **record V10 (1684 B), header version 9**, which adds the piece that makes those
grades *mean* anything:

- `mt1_oos_act[12][4]` — the same four activations as `mt1_slot0_act`, but produced by the
  head0/tail0 **snapshot taken at block start**, before the T1/H/T2 phases see any of the block's
  days. `mt1_slot0_act` is in-sample by construction — the model that emits day *d*'s activation was
  selected on a scoring window *containing* day *d* — so it cannot measure skill. Only the OOS twin
  can. On the v0.4.2.0 run the in-sample magnitude correlation was 0.543, above the 0.245 that
  perfect foresight of forward market volatility would give: proof of fit, not skill.
- `mt1_skill[12][4]` — per-channel skill (dir, acc, rng, cfd) over the trailing `MT1_SKILL_DAYS=250`
  OOS predictions: the fraction of a constant baseline's squared error removed. 1 = perfect,
  0 = no better than the constant, negative = worse. One common unit across all four channels.
- `mt1_dir_injected[12]` carries real values again — it was hardcoded to `0u` behind a stale
  "retired" comment while the constant-collapse detector was in fact firing ~560 times per run, so
  injection cadence was invisible to every offline tool.

`read_mt_log.py` prints an `MT1 OUT-OF-SAMPLE` block comparing OOS against in-sample side by side —
the gap between the two *is* the overfit. Both readers still parse V1–V9, and now prefer the header's
version stamp over size-sniffing (sizes are ambiguous: 421 V9 records divide evenly by the V10 size).

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
demonstrated out-of-sample skill (direction 52–53% OOS vs 81–84% in-sample, per-channel skill
negative in all 12 industries), so paper results measure the StockNN layer rather than an
unvalidated allocator. **Remove the flag from the paper crontab line once MT1/MT2 are confirmed.**
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
| `models.py` | `StockNN`, `MasterNN`, `MT1NN`, `MT2NN` — single source of truth for all model classes |
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

176 pytest tests across seven files in `tests/`:
- `test_models.py` — output shapes, output constraints (ReLU/sigmoid/softmax), serialization roundtrip, inject-layer growth dimensions; MT1NN/MT2NN shape + activation + forward tests; `stock_close_pos` value/identity/scale-free/degenerate-bar tests and `TestTodayLayout` section-offset tests (both must mirror the C++ twins)
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
- **`MT1NN`** (74→4, **9,208 params** composed) — per-industry preprocessor. Composed = one shared `MT1DualHead` (1,996 = two 37→28 trunks) + four `MT1Tail` (1,803 each). There is deliberately no single `MT1NN_PARAMS` constant: the C++ allocates and saves heads and tails separately (`HEADNN_PARAMS`, `TAILNN_PARAMS`), and a stale composed constant is exactly the kind of thing `load_bin` turns into a silent random-init. Pinned by `tests/test_models.py::TestMT1NN::test_param_count` and by `static_assert` in `training_v4.cpp`. **Five independent 200-slot pools per industry** (composite, direction, accuracy, range, confidence); see "MT1 pools" under MT1 scoring formulas. Input: one industry's 74-feature slice `[37 market ‖ 37 portfolio]` of the 888-feature master vector. Outputs (raw logits, activations applied at score time): `sigmoid(out[0])` = direction confidence P(positive return), `tanh(out[1])×$10K` = dollar P&L prediction, `softplus(out[2])` = range as fraction of effective delta, `sigmoid(out[3])` = calibrated confidence. Activates at `actual_day >= 25`; scored over a 10-day linear-weighted window. Files: `mt1_{industry}_model_{n}.pt` / `mt1_{industry}_best.pt`.
- **`MT2NN`** (FC+LSTM→48, ~34,572 params per slot) — cross-industry allocator. Replaces `MasterNN`. Input: 48 raw MT1 slot0 activations (4 per industry × 12 industries, no normalization — dollar magnitude IS the allocation signal). Parallel FC branch (48→36→36) + 2-layer LSTM (input=4, hidden=36) → concat 72 → taper (72→66→60→54→48). Activates at `actual_day >= 30`. Files: `mt2_model_{n}.pt` / `mt2_best.pt`.

### MT1 scoring formulas

Per-day raw outputs and target:
```
conf = sigmoid(out[0]);  delta_d = tanh(out[1]) × $10k
range_pct = softplus(out[2]);  conf4 = sigmoid(out[3])

# Market-based FORWARD target (v0.2.5+): cumulative relative return over the next MT1_FWD_DAYS=10 sessions.
# fwd_ret[i] = close[t+10]/close[t] − 1 (equal-weight per industry); daily magnitude is noise, but a
# 10-day-forward magnitude is predictable enough for MT2 to rank industries.
actual_d = (fwd_ret[i] − median_fwd_ret) × $10k     # forward relative return vs cross-sectional median
# Trainer skips the last 10 days (no forward data); single-day mkt_ret is kept ONLY for the
# backward-looking feature index (mkt_val_hist) + CSV. Production upkeep buffers predictions 10
# sessions (mt_fwd_buffer.json) and trains each when its forward window completes.
```

**Adaptive per-industry floors** (computed each day from 10-day rolling buffers, BEFORE scoring):
```
acc_floor     = mean(|actual_d| over last 10d) / 2            # cold-start $125 = MT1_FLOOR_COLD/2
range_ceiling = 4 × mean(|actual_d − comp0_δd| over last 10d)  # backward-looking clamp on r; none until buffer has data
eff_delta floor = acc_floor                                    # band anchored to the TARGET's scale
```
(The legacy flat `$250` floor is only the cold-start seed; after ~day 35 every floor is per-industry adaptive and sits *below* the typical signal.)

**Per-day component scores** (`compute_mt1_scores`):
```
err = |actual_d − delta_d|

# Direction (single-day, continuous):
sc_dir = conf if actual_d ≥ 0 else (1 − conf)

# Range (reward a tight band that still covers the error):
eff_delta = max(|delta_d|, acc_floor);  r = min(range_pct × eff_delta, range_ceiling)
m = err / r;  sc_rng = min(m, 1/m)                    # continuous, peaks at 1.0 when r == err
# Single-peaked with a gradient on BOTH sides. The pre-v0.4.1.0 form (m if m<1 else 0) rose as r
# shrank toward err and then fell off a cliff to 0, so the pool optimized itself over the edge into
# a flat all-zero landscape it could not mutate back out of — the range pool died in every run.

# Accuracy (scale-relative, smooth, always in (0,1]):
denom = max(|actual_d|, acc_floor)                    # accuracy POOL uses acc_floor/(err+acc_floor) directly
sc_acc = denom / (err + denom)

# Confidence (grade conf4 against range geometry ideal):
dor = err / r;  ideal = 1 / (1 + dor²)
sc_cfd = 1 − (conf4 − ideal)²                        # no outside-range compression (retired v0.4.1.0:
# with err > r always, it made conf4 = 0 the trivial optimum and collapsed the whole pool onto it)

# Composite (per day): equal-weight mean of the four components' SECONDARY [0,1] normalizations.
# Each secondary maps its component's naive baseline B → 0 and ideal → 1, so all four contribute
# equally (raw confidence ≈1 / range ≈0.85 would otherwise dominate; acc/dir sit near 0.5):
#   sec = clamp((raw − B)/(1 − B), 0, 1)
#   B_dir = 0.5;  B_acc = denom/(|actual_d|+denom);  B_rng = (err/acc_floor capped <1);
#   B_cfd = sc_cfd evaluated at conf4 = 0.5
composite = 0.25×(sec_dir + sec_acc + sec_rng + sec_cfd)
```

**Direction pool — forward accumulation (v0.5.0.0, BREAKING).** Direction left the replay regime
entirely. The v0.4.3.0 out-of-sample instrument measured it at **49.54% OOS against 61.09%
in-sample**, with a negative skill score in all 12 industries: the apparent skill was the pool
fitting its own scoring window. Two causes, both structural — a 10-day window over a 10-day-forward
target overlaps 9-of-10 (≈1.6 independent observations), and `step_mt1_pool` regenerates 183 of 200
slots daily, so nothing accumulates a record.

Direction now keeps **200 persistent individuals**, each carrying its own rolling 16-prediction
register (`DirSlotMeta` in `mt1_scoring.h`):
```
n = min(n_pred, 16)                       # bit 0 = most recent call
primary   = popcount(record, n) / n
secondary = Σ wᵢ·bitᵢ / Σ wᵢ              # w = 1.0,0.8,0.6,0.4 in blocks of 4 (full record Σw = 11.2)
tertiary  = 0.4·primary + 0.6·secondary
sort key  = (primary, secondary, tertiary) descending
```
Partial records normalize by the weight actually occupied, so an 8-prediction model is judged on its
8. The tertiary key is inert by construction — being a function of the other two it is equal whenever
both are equal — and is kept only because the blend distribution is worth logging.

Lifecycle: `MT1_DIR_MIN_AGE = 8` predictions before a model may be culled *or* breed;
`MT1_DIR_CULL_PCT = 0.083` of **mature** models culled per day. Steady state
`mature/N = 1/(1 + MIN_AGE×CULL_PCT)` → ~60% mature (120 of 200), ~10 births/day, ~20-prediction
lifespan. Parents = top `MT1_DIR_ELITE_PCT` (10%) of mature plus three **ephemeral** wavg blends of
top-5/10/15 — breeding templates, not pool residents, since a synthetic average has no record and
could never mature. Backfill is **flat round-robin**, not the `kChildren` table (which hands slot 0
sixteen of 180 children and drives the monoculture).

Lineage: every slot carries an inherited id; children take the parent's, blends and fresh inits get
a new one. A lineage above `MT1_DIR_LINEAGE_CAP` (25%) of the pool is barred from breeding until its
share falls back; both transitions log. This is the direct read on monoculture that previously had to
be inferred from pool spread.

Persistence: all 200 direction weights are saved (`mt1_{ind}_tail_dir_elite_{0..199}.bin`, was 20)
plus a metadata sidecar `mt1_{ind}_tail_dir_meta.bin`. The sidecar is separate because
`save_bin`/`load_bin` are raw headerless float arrays validated by exact element count — appending to
a weight file makes the loader reject it and silently fall back to random init. Direction has **no
weight-history ring**: persistent identity supersedes it, since a model worth recalling from history
is a model that was never culled.

`--dir-reps` (default 20) replays each block N times for the direction pool only. Each rep re-walks
the block from the start so inputs reset naturally, while models, records, ages and lineages carry
across. Reps restore the evolutionary throughput lost to the gentle cull — at the cost of N epochs
over the same 25 days, which is a real overfitting risk, hence a flag that can be swept and read off
the OOS instrument rather than assumed.

**Volatility channel (v0.5.0.0).** Channel 2 was a band width `r = range_pct × max(|delta|, acc_floor)`
graded by `min(m, 1/m)` against `err` — the delta head's own residual, i.e. the part of the target it
had just failed to predict. That is noise by construction, so the pool correctly converged to a
constant (range_pct 0.62, spread 2.2%). It now predicts **forward `MT1_VOL_DAYS = 20` realized
volatility of the industry's deployed portfolio**:

```
vol_pred = softplus(out[2]) × MT1_VOL_SCALE          # $500, NOT the $10,000 delta scale
sc_vol   = den / (|vol_pred − vol_actual| + den),  den = max(vol_actual, vol_floor, 1e-6)
vol_floor = mean(last 10 vol targets) / 2            # 2×floor = the naive predictor sec_vol grades against
```

`MT1_VOL_SCALE` is separate deliberately. The measured target averages **$242** (p90 $336), so at the
$10,000 delta scale a model must hold softplus in 0.005–0.06 — its flat tail, where Gaussian mutations
stop moving the output. That is the same squashing that pinned the magnitude head near zero. At $500
the useful range is softplus 0.1–1.2 and a fresh tail starts within ~1.5× of the target. Measured
effect on a 74-day smoke: vol slot-0 score **1.70 → 13.31** of a 15 max.

The benchmark it must beat out-of-sample is **r ≈ 0.41–0.46** (trailing vol → forward *portfolio* vol).
Sector vol is more predictable (0.55) but portfolio vol is what MT2 sizes on.

**conf4 (channel 3) is present but UNGRADED.** It graded itself against the band geometry that just
went away, and with `err > r` always its optimum was the degenerate `conf4 = 0` (89% of values below
0.01, pool spread 0.05%). The tail stays in the model so `MT1NN` and every `.bin`/`.pt` layout are
unchanged, but it is scored 0, dropped from the composite (now the mean of **three** secondaries), and
forwarded to MT2 as the constant `MT1_UNGRADED_FEED` — a frozen arbitrary function would be structured
noise MT2 could fit.

**Replay scoring window (acc/vol only, `MT1_DIR_DAYS = 10`):** each model is scored over the trailing 10 days,
summing its per-day score with a linear recency weight (oldest day in window = 1.0 → today = 2.0):
```
model_score = Σ_{d in window} weight(age_d) × per_day_score(d)
weight(age) = 2.0 − age/(MT1_DIR_DAYS−1)     # age 0 = today → 2.0;  age 9 → 1.0   (mt1_win_weight)
```
A longer, recency-weighted window reduces score-estimate variance → sharper selection, fewer lucky-model
promotions. Each pool sums its own per-day score (dir→sc_dir, acc→sc_acc, rng→sc_rng, cfd→sc_cfd,
composite→composite). The direction pool additionally tracks an integer correct-count (`dir_correct_dbl`,
today still counted ×2) as a secondary qualify/sort key separate from the continuous sum.

**Per-pool culls: NONE as of v0.4.1.0.** Both were removed:
- Range & confidence: the ceiling cull (`range_pct × eff_delta > range_ceiling`) is redundant now that
  `sc_rng` is self-limiting on the wide side, and its threshold was derived from *today's* target.
  `range_ceiling` survives as a plain clamp on `r`.
- Direction: the flip cull (sign-crossings `< market_flips/2`) culled precisely the constant predictor
  that the class-balanced day weights are DESIGNED to score at the no-skill baseline (dir_W/2 = 7.50).
  Removing the floor let the pool sit below it, and it did — ~35% balanced accuracy, systematically
  inverted, across all 5 passes of v0.4.0.0. Class balancing alone handles constant predictors.
  The **backfill gate** remains: freezes the pool when no live model gets ≥3 days' direction correct.

**Bounded activations (v0.4.1.0, `mt1_scoring.h`):** `conf`/`conf4` decode as
`sigmoid(MT1_LOGIT_CAP × tanh(raw / MT1_LOGIT_CAP))` → (0.018, 0.982), and `range_pct` softplus input is
clamped to ±20. A saturated sigmoid has zero derivative, so weight mutations stop changing the output
and the pool freezes genetically: by pass 5 of v0.4.0.0, 96.7% of `conf` values were *exactly* 0 or 1
and the 200-slot direction pool had a best−min spread of 0.35%. Decode-side only — model files unchanged.

**MT1 pools — one shared head + 4 tails, 200 slots each.** Two regimes as of v0.5.0.0:
- **Replay pools (acc/rng/cfd) and the head** — slot layout **0–16** direct elites · **17–19** wavg
  blends · **20–199** mutations (180). No injection slots; the shared head propagates cross-component
  learning. Mutations use a **weighted children table** (`kChildren`): slot 0 = 16 children, slots
  1–4 = 13/13/12/12, slots 5–16 = 8, wavg 17–19 = 6. Slot identity is positional — the pool ranks
  candidates and overwrites the buffer, so a slot means "whoever placed k-th today" and 183 of 200
  models are one day old. 5-day history ring (10/day) supplies extra candidates.
- **Direction pool** — 200 persistent individuals on forward accumulation; see the section above.
  Cull ~10/day, ~20-prediction lifespan, no history ring, lineage-tracked.
- Production model = composed head0 + tail0[4]; its 4 activations feed MT2. The direction tail0 is
  the best *mature* individual, copied into slot 0 by convention rather than by reordering the pool.

### Production inference chain (when MT2 models available)

```
build_master_features() → today444
  → MT1 slot0 ×12 (slices today444[i*37:(i+1)*37]) → 4 raw activations each
  → build in48: [conf, delta_t, range_pct, conf4] × 12 (no normalization). Source = direction-pool
    slot0 when MT2_FEED_DIRECTION=true (v0.2.5+, the strongest daily signal), else composite slot0;
    toggle back to composite once MT2 shows a learning curve. Allocation path needs convert_weights.py
    to emit mt1_{ind}_dir_best.pt (else degrades to composite cleanly).
  → MT2 slot0 forward pass → (12,4) logits → argmax per industry → tier map
  → allocation/liquidation (unchanged from MasterNN path)
```

`run_master_allocation()` in `production_v2.py` tries MT2 first (`mt2_best.pt` exists), falls back to MasterNN, then equal allocation. MT2 input: 48 raw MT1 activations (no normalization; `mt2_norm_stats.json` no longer used).

**v0.2.6.0 additions:** (1) MT2 mutation sigma lowered 0.002→0.001 (`master_sigma/6`; tighter pool). (2) **Raw slot0 logging** — each industry's slot0 activations `[conf, delta_t, range_pct, conf4]` are written per day to `mt_training_log.bin` (record V7, 1244 bytes, `MT_LOG_VERSION=6`) so any run can be re-graded offline under any target. (3) **Pool-consensus allocation diagnostic** — a wisdom-of-crowds ensemble: rank industries by the pool's mean tier vote, re-tier by the pool's *observed* mean tier-0 count (NOT `opt_tier` — that would leak the realized outcome), grade with the normal pts formula. Two variants — flat (all slots) and look-behind-weighted (elites weighted by reliability over the last `MT2_LB_DAYS=5` days, leak-free because read-only). Logged to the binary log + CSV (`mt2_consensus_flat_pts`, `mt2_consensus_wtd_pts`); plotted as thick purple lines (solid=weighted, dashed=flat) on `mt2_performance.svg`. The MT2 injection already preserves the top-half elites (slots 0–7), so the limit-cycle is regime-driven, not injection-driven; the consensus is the regime-robustness mechanism.

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

`production_v2.py` runs once per trading day: fetch data from yfinance → run MT1×12 + MT2 (or MasterNN fallback) to rebalance capital → run StockNN per active industry → submit limit/stop orders to Alpaca → perform one upkeep evolution step on yesterday's data (via `upkeep.py`). Alpaca credentials are read from `keyring` or environment variables (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`). All per-account state files (`state.json`, `owners.json`, `master_state.json`, `mt1_rolling_state.json`) live under `--model-dir`, enabling multiple accounts to run independently with different `--model-dir` paths and credentials.

`--paper` routes all API calls to Alpaca's paper trading endpoint — orders are submitted and portfolio state is read from the paper account, giving real paper trading history without risking real money. Omit `--paper` for live trading.

### Changelog hook

A `PostToolUse` hook in `.claude/settings.json` auto-updates `CHANGELOG.md` and amends the commit whenever Claude makes a `git commit`. This is intentional — do not skip it.

## Key constants (defined at top of each training/production file)

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_SLOTS` | 200 | Total model slots per pool |
| `ELITE_COUNT` | 17 | Direct elite slots (industry + MT2) |
| `ELITE_POOL` | 20 | Elites + weighted-average slots (industry + MT2) |
| `MT1_COMP_SLOTS` | 200 | Slots per MT1 pool (5 pools: composite, dir, acc, rng, cfd) |
| `HT_PARENTS` | 20 | Replay-pool parents (17 elites + 3 wavg); no injection slots |
| `HT_MUTS` | 180 | Replay-pool mutations, assigned by the `kChildren` weighted table |
| `MT1_SCALE_DOLLARS` | $10,000 | tanh(out[1]) × scale = dollar P&L prediction |
| `MT1_DIR_DAYS` | 10 | Replay scoring window (acc/rng/cfd + head): linear-weighted, oldest=1.0 → today=2.0 |
| `MT1_FLOOR_COLD` | $250 | Cold-start seed only; `acc_floor` cold = MT1_FLOOR_COLD/2 = $125, then per-industry adaptive |
| `MT1_ROLLING_DAYS` | 10 | Days in per-industry |actual_d| / residual rolling buffers (floor + ceiling) |
| `MT1_VOL_DAYS` | 20 | Forward horizon for the volatility target (channel 2) |
| `MT1_VOL_SCALE` | $500 | softplus(out[2]) × scale = predicted vol; separate from the $10,000 delta scale |
| `MT1_UNGRADED_FEED` | 0.5 | Constant the ungraded conf4 channel forwards to MT2 |
| `MT1_SKILL_DAYS` | 250 | Trailing window for the read-only out-of-sample per-channel skill scores (V10 log) |
| `MT1_DIR_HIST_BITS` | 16 | Direction pool: rolling per-model prediction record (2 bytes), bit 0 = newest |
| `MT1_DIR_MIN_AGE` | 8 | Predictions before a direction model may be culled OR breed |
| `MT1_DIR_CULL_PCT` | 0.083 | Fraction of MATURE direction models culled per day (→ ~60% mature, ~20-prediction life) |
| `MT1_DIR_ELITE_PCT` | 0.10 | Top fraction of mature direction models used as mutation parents |
| `MT1_DIR_LINEAGE_CAP` | 0.25 | Lineage above this share of the pool is barred from breeding |
| `--dir-reps` | 20 | Times the direction pool replays each block (CLI flag, not a constant) |
| `MT1_RANGE_CEIL_MULT` | 4 | range_ceiling = 4 × mean |actual_d − comp0_δd| (backward-looking) |
| `MT1_LOGIT_CAP` | 4 | Pre-sigmoid squash for conf/conf4 → (0.018, 0.982); prevents saturation lock-in |
| `MT1_SOFTPLUS_CLAMP` | 20 | Clamp on the softplus input for range_pct (guards → inf) |
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

Current version: **0.6.2.0**

To bump the version, edit `VERSION` in `version.py` and `TRAINER_VERSION` in `training_v4.cpp`, then rebuild the C++ binary.

## Ignored directories

`models/`, `logs/`, `stock_data/`, and `data_dump/` are git-ignored (large binaries, runtime logs, data, and diagnostics).
