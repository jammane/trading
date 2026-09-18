> # SUPERSEDED — the architecture this plan describes was deleted in v0.8.0.0
>
> MT1 is no longer a shared dual head plus four specialized tails, and there is no block-alternating
> T1/H/T2 schedule, no five 200-slot pools per industry, and no four graded channels. It is one
> `MT1Net` (74→1, 3,501 params), one output, one pool of 200 persistent individuals per industry,
> scored daily. See **MT1 target and scoring** in CLAUDE.md.
>
> Kept because it records *why* the heads/tails design was built and what it was trying to solve —
> the reasoning is still worth reading even though the answer was wrong. What it could not do was
> be measured: a 10-day scoring window over a 10-day-forward target overlaps 9-of-10, leaving ~1.6
> independent observations, and every model was selected on a window containing the day it was
> graded on. The out-of-sample instrument built to work around that reported **49.54% OOS against
> 61.09% in-sample**, with negative skill in all 12 industries.
>
> **Do not implement anything from this document.**

# MT1 heads/tails: shared trunk + specialized tails, block-alternating training

> Recovered from plan-mode transcript (session d620c1ab). Kept in-repo so it survives `/clear`.
> **Progress markers added inline** — search `STATUS:`.

## Context

The MT1 direction pool was deadlocked (unreachable score floor → frozen below no-skill →
MT2 flat). That deadlock is now **fixed** (two-half selection + correct-count floor, committed
`61aa2a1`/`0dc1b0f`): a diagnostic shows direction reaching real skill (8–9) and MT2 reaching
~+11.7 vs +13.9 ideal. On top of that fix, we're restructuring MT1 for **feature sharing**:
today each of the 4 component pools (dir/acc/rng/cfd) trains a *complete independent* 4-output
net — relearning feature extraction four times. The redesign shares one feature trunk and
specializes only the output transform:

- **One shared `MT1Head`** (37→28) per industry — the feature extractor.
- **Four specialized `MT1Tail`s** (28→1) — direction / accuracy / range / confidence.
- Trained by **alternating coordinate descent** in 25-day blocks (freeze one side, evolve the
  other, so each has a stationary fitness), which is what shared-trunk + evolution needs.

Intended outcome: better-conditioned MT1 that shares learned features across the four outputs,
measured against the (now-unblocked) current architecture as baseline.

Architecture note: `MT1Head` (998 params) + `MT1Tail` (1,187) already added + validated in
`models.py` (composed `MT1NN` = head + 4 tails = 5,746; composition exact).

## Design

### Pools (per industry) — replaces 4 component pools + composite blends
- **1 head pool** — 200 slots × 998 params. Fitness = **composite** score of
  `head_slot → concat → frozen-best 4 tails → 4 outputs` (reuse `compute_mt1_scores(...).composite`).
- **4 tail pools** (dir/acc/rng/cfd) — 200 slots × 1,187 each. Fitness = **that component's** score
  of `frozen-best head → concat(cached) → tail_slot → 1 output`. Direction tail keeps the
  class-balanced score + **two-half selection** + correct-count collapse floor (just built);
  acc/rng/cfd reuse `compute_mt1_scores`.
- Production model `comp0` = best head + best 4 tails, composed → the 4-output MT1 that feeds MT2.
  The current `step_mt1_composite` blend pool + `blend_hist` are retired (the head pool *is* the
  shared-trunk optimizer on the composite objective).

### Block-alternating training (25-day blocks)
Per block (same 25 days replayed per phase; snapshot data-state at block start, restore per phase):
1. **T1 — tails + StockNN**: freeze head; evolve the 4 tail pools + `step_industry`.
2. **H — head**: freeze tails; evolve the head pool (fitness = composite).
3. **T2 — tails again**: freeze the *new* head; evolve tails once more (tails get the last word).
4. **M — MT2**: freeze MT1; evolve MT2.
Then advance to the next block. The 10-day rolling window / dir-day buffer / history are **data**
(feature+target), identical across phases — snapshot at block start, restore before each phase's
replay, advance once at block end.

### Key reuse (do NOT reinvent)
- Two-half direction selection, correct-count floor, collapse-injection, balanced day weights —
  `training_v4.cpp` `step_mt1_component` (dir path) → moves to the direction *tail* pool.
- `compute_mt1_scores` — acc/rng/cfd tails + head composite fitness.
- Mutation/breeding (`kChildren`/weighted children table), wavg blends, history buffers, `mlock`.
- MT2 unchanged — still fed the composed MT1 output via the existing `sign(conf)×|delta|` reassembly.
- `mt1_win_weight`, adaptive `acc_floor`/`range_ceiling`, `_dir_day_weights` (upkeep).

## Files / changes (staged into increments)

**Increment 1 — architecture + serialization (foundation). STATUS: DONE (commit b70b154).**
- `models.py`: `MT1Head`/`MT1Tail`/composed `MT1NN` — done + validated.
- `training_v4.cpp`: `head_forward`/`tail_forward`/`mt1_composed_forward`; head/tail offset
  constants + `HEADNN_PARAMS=998`/`TAILNN_PARAMS=1187`; `init_head_weights`/`init_tail_weights`.
- `prepare_models.py`: `HEAD_LAYER_DEFS` + `TAIL_LAYER_DEFS`.
- `tests/test_models.py`: param counts 998/1187/5746 + roundtrip. Parity 6e-08/0.0.

**Increment 2 — pool restructure + fitness (C++ core).**
- part 1 — STATUS: DONE (commit fe1d8a4). `MT1Scratch` head pool + 4 tail pool buffers (+ history/
  new_elites/mut/best-production), accessors, alongside old branched pools. Compiles clean, unused.
- part 2 — STATUS: CODE COMPLETE (local `g++ -std=c++20 -fsyntax-only` passes; droplet build
  pending). Added `step_mt1_pool` (generic select/mutate/wavg/history/inject, parameterized by
  param count + a per-model score callback), `step_mt1_tail(comp, ...)` (frozen best head → cached
  concat + frozen other-3-tail logits → score one tail slot; dir path reuses balanced weights /
  two-half selection / flip cull / correct-count collapse floor) and `step_mt1_head(...)` (head
  slots → 4 frozen best tails → windowed composite). All three are additive/unused until Inc 3.
  **No injection slots** (user decision): the old composite→pool injection existed to propagate one
  component's learning to the others; the shared head trunk now does that. So the head/tail pools
  use `HT_PARENTS=20` (17 elites + 3 wavg, no injected slots) and the reclaimed capacity (former 15
  immigrant children + 5 parent slots) is redistributed into more elite mutations — `HT_MUTS=180`,
  kChildren `{16,13,13,12,12, 8×12, 6,6,6}` (avg 9/parent). `mut_seeds` bumped 175→180.
  DEFERRED to a part 3 / Increment 3: making `step_mt1` a dispatcher, `save_mt1_all` /
  `load_or_init_mt1` head+tail files, `MTLogRecord` sourcing — those touch the live day-loop, so
  they land with the block loop to keep each commit compilable + isolated.

**Increment 3 — block loop (main). Sub-staged 3A/3B/3C.**
- 3A — STATUS: DONE (local syntax-check passes; droplet build pending). Foundation, additive/unused:
  `MT1_BLOCK_DAYS=25`; `MT1DataState` + `snapshot_mt1_data`/`restore_mt1_data` (captures the
  replay-critical DATA window only — dir_day buffer, rolling floors, streak/cooldown — NOT model
  pools/histories, which carry evolution forward); `save_mt1_ht`/`load_or_init_mt1_ht` (BREAKING
  head/tail file format, additive `head_*`/`tail_*` filenames so it coexists with old files until 3C;
  production bests fall back to elite slot 0 so they are always valid).
- 3B — STATUS: CODE COMPLETE (local syntax-check passes; droplet build pending). `run_mt1_block`
  driver: snapshot data-state; T1 (freeze head0+tail0, evolve 4 tail pools across the block, set
  tail0=slot0); restore; H (freeze tail0, evolve head, set head0=slot0); restore; T2 (freeze new
  head0+tail0, evolve tails, set tail0); leave data advanced at block end. Per-day floors + rolling
  residual sourced from the composed head0+tail0 production model (`mt1_composed_forward`). Returns
  an `MT1Result` from the last block day (head pool → "composite" stats, tail pools → components,
  composed output → slot0/dir0 activations). Additive/unused until 3C.
- 3C — STATUS: DONE (validated). Local syntax-check + 93 tests pass; droplet build clean; diagnostic
  run confirmed the block flush works end-to-end (EXIT=0, valid MT log records, no NaN, MT1 direction
  climbing 5.4→7.7 across blocks, MT2 ≈ideal). Validated with a throwaway MT1_BLOCK_DAYS=5 build for a
  fast flush; committed source keeps MT1_BLOCK_DAYS=25.
  `run_training`'s day loop now: per day does StockNN + OHLCV/market history + caches each fwd-valid
  day's 444-features/targets/StockNN-results; every MT1_BLOCK_DAYS cached days (or at range end / first
  non-fwd day) calls `process_block` → run_mt1_block ×12 (T1/H/T2), MT2 M-phase over the block, then
  flushes deferred CSV rows + one MT log record + a save. Non-fwd days get an inline CSV row. Switched
  `run_training` to `load_or_init_mt1_ht`/`save_mt1_ht`. Old `step_mt1`/component/composite + old pools
  kept as dead code (still used by `run_drift_study`). Retiring them is a later cleanup.
  Decisions locked: (a) **cached-feature block model** — advance StockNN + market/OHLCV history + CSV
  per day once, cache each fwd-valid day's 444-features + targets, trigger a block every
  MT1_BLOCK_DAYS cached days; (b) **post-block MT1 feeds MT2** for every block day (mild within-block
  leak accepted); (c) MT2 M-phase + CSV rows are written at block end (buffer per-day StockNN results
  + mkt values for the block); (d) one `MTLogRecord` per block, layout unchanged; (e) MT1 production
  = composed head0+tail0 → in48 built from it (MT2_FEED_DIRECTION toggle becomes moot — one
  production model). **Keep the old `step_mt1`/component/composite + old pools as dead code** so
  `run_drift_study` still compiles; retiring them + the old MT1Scratch buffers is a later cleanup once
  the drift study is migrated. Wire `save_mt1_ht`/`load_or_init_mt1_ht` in `run_training` only.

**Increment 4 — Python production mirror.** Larger than the original one-line note: the whole
Python MT1 chain (convert/prepare seam, upkeep, production) still used the old component-pool
architecture. Sub-staged 4A/4B/4C.
- 4A — STATUS: DONE (validation pending droplet pytest). The convert/prepare seam for head/tail:
  `convert_weights.py` now converts head + 4 tail pool `.bin`→`.pt` (`_convert_mt1_pools`) and
  composes `mt1_{ind}_best.pt` from the head + 4 tail production `_0.bin` (`_convert_mt1_best`,
  replacing the dead `comp_0.bin` path that 3C broke). `prepare_models.py` main emits head/tail pool
  `.bin` from `.pt` (`HT_PARENTS=20`). Added a compose roundtrip test. File naming (must match C++
  `save_mt1_ht` + upkeep): `mt1_{ind}_head_{elite_N|model_N|0}.bin/.pt`,
  `mt1_{ind}_tail_{dir|acc|rng|cfd}_{...}`.
- 4B — STATUS: harness written (`tests/test_upkeep_mt1.py`, skip-marked — unskip when the rewrite
  lands; it is the validation gate). Contract locked: `upkeep_mt1_industry(ind, model_dir, in37_t,
  actual_d, rolling_state=…)` → finite 10-tuple; writes `mt1_{ind}_head_model_{0..19}.pt` (MT1Head) +
  `mt1_{ind}_tail_{dir|acc|rng|cfd}_model_{0..19}.pt` (MT1Tail) + composed `mt1_{ind}_best.pt`;
  tails evolve every run, head only every `MT1_BLOCK_DAYS` runs (block counter in
  `mt1_rolling_state.json`).
- 4B — STATUS: DONE (validated). Rewrote `upkeep_mt1_industry` to the head/tail block cycle: daily
  tail phase (freeze best head + other best tails, `_score_tail` mirrors C++ `step_mt1_tail`;
  direction keeps class-balanced weights + two-half selection + flip cull + collapse backfill) + a
  head phase every `MT1_BLOCK_DAYS` runs (freeze best tails, composite fitness). New `_ht_*` helpers
  (select/mutate with HT layout — 20 parents, 180 muts, no injection — + per-pool history). Bootstraps
  pools from `best.pt` or fresh; composes `best.pt` from new head0+tail0. Added `MT1_BLOCK_DAYS`,
  `HT_PARENTS`, `_HT_CHILDREN`, `UPKEEP_HEAD_SIGMA`. Harness unskipped → 95 tests pass.
- 4C — STATUS: DONE. `run_mt_inference` already loads the composed `mt1_{ind}_best.pt`; removed the
  now-moot `MT2_FEED_DIRECTION`/`dir_best.pt` special-case (single production model → composite and
  direction feeds identical). Production `upkeep_mt1_industry` interface (10-tuple) preserved, so
  `production_v2.py` needs no change.

**Cleanup — retire old component/composite code. STATUS: DONE (validated).**
- Migrated `run_drift_study` to head/tail: `drift_mt1_day` (daily tails + periodic head, mirror of
  upkeep), band = head-pool top-8 composed with frozen best tails, scoring via `mt1_composed_forward`,
  `drift_reseed` → `load_or_init_mt1_ht`; the `MT2_FEED_DIRECTION` guard is gone (moot).
- Removed the dead C++: `mt1_forward`, `step_mt1`, `step_mt1_component`, `step_mt1_composite`,
  `gen_mt1_blend`, `save_mt1_all`, `load_or_init_mt1`, `MT1BlendResult`, the old `MT1Scratch` pool
  buffers (comp_elites/new_elites/mut_buf/pool_hist/blend_hist/comp_inject/dir_inject/rng_inject/
  comp0_buf) + accessors, and the old constants (`MT1_COMP_PARENTS/CHILDREN/INJECT/RANGE_INJECT/
  BLEND_SLOTS`, `MT1_DIR_STREAK_TRIP/COOLDOWN_LEN/QUALIFY/BACKFILL`). Head/tail elite buffers now
  sized `HT_PARENTS` (20) not 25.
- Removed the dead Python: upkeep `_select_and_mutate_mt1_component`, `_mt1_burst_component`,
  `_load_comp_pool_hist_models`, `_save_comp_pool_hist`, the old children table + unused imports;
  `MT1_LAYER_DEFS` from prepare/convert.
- ~1,130 net lines removed. Validated: C++ build clean (both targets), 95 pytest tests, and a
  BLOCK=5 smoke run of the trainer (valid block record, no NaN, no crash).

## Risks / watch-items
- **State snapshot/restore across phases** is the subtle core — rolling window/history must be
  identical per phase replay or fitness drifts. Snapshot MT1Scratch data buffers (not model pools)
  at block start.
- **Compute**: head fitness = 4 tail forwards per head slot; tail fitness caches frozen-head concat
  (1 forward/day). A few× MT1 forward passes — cheap nets, acceptable.
- **Ordering** (T2 second tail pass) — refinement; can drop to plain T1→H→M if it doesn't help.
- **BREAKING again**: model format changes (head/tail files) — bump BUILD (already v0.3.0.0).
- Large rewrite of MT1 core; each increment must compile + validate before the next.

## Verification (per increment, droplet — no local torch)
1. Build (`cmake --build build`), `ctest`/`pytest tests/` (param counts, roundtrip).
2. **Parity** (Inc 1): numpy C++-offset replica vs `MT1Head`/`MT1Tail` (≈1e-7). DONE.
3. **Short diagnostic** (Inc 2–3): `--start-day 16 --stop-day 90 --no-save`, separate output dir;
   confirm no NaN, tail pools score, head composite improves, direction above no-skill, injections
   low. `read_mt_log.py` + `plot_training.py`.
4. **Baseline A/B**: full block-trained run vs deadlock-fixed current architecture
   (branch `mt1-absolute-redesign`) — redesign must beat the unblocked baseline.
