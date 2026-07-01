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
- 3B — STATUS: TODO. `run_mt1_block` driver: snapshot data-state; phase T1 (freeze head0+tail0,
  evolve 4 tail pools across the block, then set tail0=slot0); restore; phase H (freeze tail0, evolve
  head, then set head0=slot0); restore; phase T2 (freeze new head0+tail0, evolve tails, set tail0);
  leave data-state advanced at block end. Floors sourced from the composed head0+tail0 production model.
- 3C — STATUS: TODO. Rewire `main`'s day loop into the block loop; interleave StockNN + MT2's M phase;
  wire `save_mt1_ht`/`load_or_init_mt1_ht`; source `MTLogRecord` from tail pools + composed comp0;
  retire the old branched component/composite pools + `step_mt1`/`step_mt1_component`/`step_mt1_composite`.

**Increment 4 — upkeep block cycle. STATUS: TODO.**
- `upkeep.py`: daily upkeep evolves tails + MT2 only; block counter (persisted in
  `mt1_rolling_state.json`) triggers a head cycle every `MT1_BLOCK_DAYS` runs. Mirror the
  frozen-cross-reference (tails↔head) exactly as the trainer.

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
