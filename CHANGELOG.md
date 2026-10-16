# Changelog

Notable changes, grouped by release. Versions follow `0.BREAKING.FEATURE.BUILD` (see
**Versioning** in CLAUDE.md); a CI job bumps `BUILD` on every push to `dev`. Full detail for any
entry is in its commit message.

---

## Unreleased

### Changed
- **Training trades under production's limits.** The fill simulators used to re-buy what they had
  just sold, spend the same morning's sale proceeds, size buys at the fill price and sell a
  position back the day it was bought, none of which production can do (on paper, 16 of 26
  holdings matched the simulation on 2026-10-14). Production's order rules now live once, in
  `training_lib.plan_orders`: production queues the plan and every simulator fills exactly it,
  including the C++ trainer through a twin pinned by `tests/test_order_plan.py`.
- **New order rules replace every model at the next weekly pass.** `training_lib.ORDER_RULES`
  (now 2) is recorded in the race run as `order_rules.txt`; when the two differ, the Friday pass
  crowns every industry outright and promotes all 12 to paper, because models raised under the old
  rules learned on trades the broker does not make.
- **`sell_qty` is a market order at the next open** in production, as the trainer fills it. It
  was a limit at the decision close; on 2026-10-15 AR stayed held while the trainer sold it.
- **A buy's stop-loss is 10% below the lower of its limit and the decision close**, not 10% below
  the limit. A limit above the close fills near the next open, so the old stop could sit at the
  market; on paper, Alpaca cancelled SSRM's stop the moment its buy filled. The rule is one helper
  (`buy_stop_price`) shared by production, the C++ trainer and the Python simulator, so the models
  train on the stops the broker holds. `tests/test_buy_stop.py` pins the C++ twin.

---

## 0.8.1.46 – 0.8.1.57 — Paper trading goes live (2026-10-05 to 2026-10-13)

The v5 race finished, its committees were wired into production, and paper trading started on
2026-10-13 with the daily cycle download → train (C++) → convert → decide → order, and a weekly
training pass on Fridays.

### Added
- `production_v2.py --sell-only IND,...` puts industries in sell-only mode: StockNN still runs
  and its sells are placed, but nothing is bought. Positions it does not sell stay held, and the
  stop on any partly sold position is re-placed for the remainder.
- **Daily training with the C++ trainer.** `training_v4_cpp --upkeep` trains one day on
  production's real books and carries everything a pass keeps in memory in files, starting from
  the champion store. `production_v2.py --cpp-upkeep` runs it each session before deciding, then
  converts the StockNN elites, so the cycle is download → train → convert → decide → order.
  `--race-only` trains just the committee members.
- **Committee allocation.** `production_v2.py --committee` allocates with the race committees:
  the loser committee puts 4 industries in sell-only and the winner committee tilts the other 8
  (half the cash spread evenly, half to its 4 picks). Predictions come from the daily-trained race
  models (`committee_alloc.py`); the vote reproduces `committee_race.py` on every v5 day.
- **Paper's daily record.** Each committee decision is settled the next session against the
  account's real fills and appended to `committee_daily.csv` in `race_daily.csv`'s format, and the
  trainer's own logs for every training day accumulate under `logs/<account>/<mode>/training/`,
  so paper and training can be compared with the same tools.
- **Per-symbol suspension.** A universe symbol whose data stops arriving (its newest bar is
  behind the universe's newest session) gets no orders until its data returns; its shares stay
  held and the rest of the universe trades as usual. Logged to `suspended.log`.
- **The weekly pass.** Every Friday after trading, `weekly_pass.py` adds one full pass to the
  standing race run (`training_v4_cpp --continue-run`), promotes every champion it takes to paper,
  and re-runs the committee races as a report. Symbols that stopped arriving, or that pushed their
  industry's prices out of band, are swapped first.
- **Ready replacements.** `replacements.py` keeps three vetted replacement symbols per industry in
  `universe_watchlist.json`, screened from the industry's own Yahoo industries and refreshed every
  Monday so concerns can be raised before Friday.
- `training_v4_cpp --continue-run` adds passes to an existing run, numbered after its last, with
  each invocation's logs in their own `continue_p<N>/`; `--crown` crowns named industries outright.
- `committee_race.py` takes several `race_daily.csv` files and matches passes by date.

- **Simulated vs real fills.** Each paper session's training day writes slot 0's simulated
  execution (`orders_sim.csv`), and `fill_check.py` compares it with what the broker did, at the
  level of the model's outputs, the orders and the fills, into `fill_check.csv`.

### Fixed
- **StockNN's inputs in production.** Its 15-day history ended on the decision day instead of the
  day before, zeroing every day-over-day input; on the v5 champions only 27 of 144 symbols matched
  the trainer's outputs for the same day and book (144 of 144 after the fix). Found by the first
  fill check; the first paper session had traded on the wrong inputs.
- **Orders Alpaca accepts.** The first real paper session was rejected order by order: prices were
  sent with sub-penny digits, and a buy's stop-loss was sent as a separate sell stop, which Alpaca
  refuses beside an open buy. Prices are rounded to whole cents in the order's favour, each buy
  carries its stop in one OTO order, held positions have their stops re-placed every run in every
  industry, and a symbol sold today is not also bought.
- Production saved the bar it fetched under today's date even when the feed returned an older
  one, which made a halted symbol look current. Bars are saved under their own date.
- Production read its cash and positions from `state.json`, which defaults to $20,000 and nothing
  held, instead of from the Alpaca account. Both now come from Alpaca on every run.

## 0.8.1.39 – 0.8.1.45 — Qualifier races and committee voting (2026-10-03 to 2026-10-04)

### Added
- **Qualifier races.** `race_qualify.py` decides which race entries go on to the next race: an
  entry must beat flat allocation in more than 50% of its counted passes (raised from 40% after
  13 of 17 entries had qualified with a pass still to run). Entries whose model
  changed count only the passes since the change.
- **Contender tier.** A non-qualifier whose mean gap over flat is above the qualifiers' average
  still runs in the next race.
- `--race-skip` switches race entries off at runtime, so the roster in the code stays complete.
- **`race_daily.csv`.** For the last 252 days of every pass: each entry's 12 predictions, the 4
  industries it funded, its result, and every industry's realised return.
  `race_daily_check.py` verifies each row against the realised returns.
- **`committee_race.py`.** Replays `race_daily.csv` to test whether committees of 3–6 entries,
  voting under seven methods (including an online Bayesian vote-pattern tree), beat their best
  member consistently across passes. `--watch` tracks across passes which entries keep becoming
  entrants, which members lift the committees they sit on, which methods do best, and which
  committees recur near the top. `--champions pass_reference.csv` races the committees on the
  champion composite, each industry taken from its champion pass, which is the mix production
  deploys and the race that picks the committee to trade.
  `--losers` runs the same races on the other end of the ranking: committees flag the 4
  industries expected to do worst, scored on how far below flat they fall and how often they
  lost money.

### Changed
- StockNN champions are crowned on slot 0's **recency-weighted mean daily book return** over the
  whole pass (`pass_reference.csv` schema v2), the same criterion the race uses. The previous
  15-day metric dethroned at exactly the rate expected from chance.
- `champion/` now carries each pass's race models beside the StockNN elites they were trained
  with, so a champion is a matched set.
- README, CLAUDE.md and design notes brought up to date; the trainer's `--help` text corrected.

## 0.8.1.27 – 0.8.1.38 — The MT1 race (2026-09-22 to 2026-10-03)

### Added
- An in-loop **race** of allocator candidates: MT1, MT1C, MT1S (per-symbol) and TANH32
  architectures × evolutionary and gradient search × restart-or-carry across passes, all on the
  same days and order sets.
- **Allocation backtest** per entry: top-4 industries vs an even 1/12 book, in bp/day, plus
  recency-weighted and 100-day trend lines.
- Demeaned-MSE and ListNet ranking losses, and a Bayesian (discounted Gaussian) allocator.
- Per-stock MT1S-stk entries: one network per stock, 144 in all.
- `prune_runs.sh` in the repository, with a whitelist and a `KEEP` marker for deliverable runs.
- A census of how often StockNN's price fractions sit at exactly 0 or 1.

### Fixed
- Gradient entries collapsing under Adam: leaky ReLU, and a per-industry learning rate steered by
  the fraction of distinct predictions rather than by dead units.
- TANH32 collapse traced to 200 per-row Adam steps per day; it now takes one batched step.
- Evolutionary entries of one architecture ran bit-identically; each now draws its own seed.

### Removed
- Three entries that beat flat in at most 1 of 5 passes (MT1C grdC, MT1S-sum grad, TANH32 grnk).
- MT2 as a trained stage of the race.

## 0.8.1.10 – 0.8.1.26 — Structure studies and gradient MT1 (2026-09-20 to 2026-09-21)

### Added
- Studies of whether StockNN's P&L series has exploitable structure: `pnl_persistence.py`,
  `pnl_conditional.py`, `depth_sweep.py`, `dyn_tree.py`, `living_bn.py`, each with controls of
  known answer.
- Online conditional models trained inside the training pass (`bayes_online.h`), and
  `--no-nn-race`.
- A pre-registered out-of-sample test of the volatility allocation signal (`oos_vol.py`).
- **Gradient-trained MT1** (`mt1_backprop.h`, hand-coded backprop + Adam, finite-difference
  checked), trained on each model's own order intent in raw dollars.
- Order-level attribution (`analysis/order_level.py`, `analysis/order_ev.py`): passive limit
  placement is 10–14× better per order placed, replicated across passes.

### Fixed
- Test stubs leaking into `sys.modules` for every later test.

## 0.8.1.0 – 0.8.1.9 — Correct target, measurable horizon (2026-09-18 to 2026-09-20)

### Changed
- **MT1's target corrected** to the industry's next-session book P&L. The previous target
  cancelled the market move exactly and measured only the trade delta.
- `mt1_dataset.bin` logs every day's features and outcome components, so the forecast horizon is
  an offline measurement; `mt1_analysis.py` evaluates it with block-bootstrap errors, shuffled and
  planted controls, and no look-ahead.

### Added
- MT1CNet (today's data plus order intent, scale-free) and MT2INet (an independent 888-feature
  allocator) as competitor pools.

### Fixed
- `-ffast-math` had folded every `isfinite()` guard in the trainer to a constant; the guards now
  run, pinned by `test_fp_guards.cpp`.
- Compiler warnings enabled for the trainer and all `-O3` warnings cleared.

## 0.8.0.0 — MT1 rebuild (2026-09-17) — BREAKING

### Changed
- MT1 rebuilt from a shared head with four tails and five pools per industry into **one
  `MT1Net` (3,501 params), one output, one 200-slot pool per industry**, scored daily against the
  next session. The score is anchored to a trailing-mean baseline, and its causality contract is
  pinned in both C++ and Python.
- MT1/MT2 model files and logs from earlier versions are unloadable.

### Fixed
- MT1 retirement accounting read a slot's age after it had been reset.
- `upkeep.py` sorted the MT1 pool with its key components reversed.

## 0.6.8.0 – 0.7.0.6 — Production safety and controls (2026-09-16 to 2026-09-17)

### Added
- `--no-orders` for catch-up sessions without trading.
- **No-learning controls:** `--control-untrained` and the in-process `--control-random`.
- `check_determinism.py` and `fill_reconciliation.py` (simulated vs actual Alpaca fills).
- A slot-0 holdings log for P&L decomposition.

### Changed
- **Whole-share fills** in the simulator, matching production (training semantics change).
- **Universe re-normalized** to a $30–$90 price band (max/min 3× per industry), selected for
  daily range and liquidity; 92 of 144 symbols replaced. Forces a full retrain.
- `stock_data/` is the union across dev, paper and prod universes, and fails safe.

### Fixed
- Production over-committed each industry's capital ~8× and applied no position limits.
- Non-finite OHLC bars reached the portfolio valuation; one NaN overflowed a logged value to
  −2,147,483,648.
- Seven copies of the buy-sizing arithmetic consolidated into one `size_buy()`.

## Earlier history (2026-04-10 to 2026-09-16)

- 0.6.7.0 — Mutation success rate logged (Rechenberg's 1/5 statistic).
- 0.6.6.0 — `--no-save` no longer disables training; `--load-dir` is a seed again.
- 0.6.5.0 — Training pauses while a production cycle runs.
- 0.6.4.0 — RNG seeded from the clock and re-seeded every pass.
- 0.6.3.0 — Per-industry proportional pass-boundary seeding.
- 0.6.2.0 — `--flat-allocation` for paper trading.
- 0.6.1.0 — StockNN `close_pos` and `close_vs_wap` features (breaking).
- 2026-06-01 — Docstrings, linting configuration and README for public release.
- 2026-04-10 — Initial commit.
