# Phase 3 — Lag/Daily Drift Study (design)

**Status:** design (pre-implementation). Target home: a new `--drift-study` mode in
`training_v4.cpp`. **Not** production upkeep — this is an offline calibration tool whose
*only* output is a number: the optimal **lag : dailies** ratio that the separate, memory-efficient
production upkeep will then be configured with.

---

## 1. Purpose

The **look-ahead (heavy) training** is the high-quality anchor, but it carries an *unavoidable*
2-week blind spot: its target is a 10-session forward return (`MT1_FWD_DAYS=10`), so it can never be
trained on the most recent 2 weeks of market behavior — the freshest *possible* look-ahead model is
already 2 weeks stale. **Daily upkeep is a deliberately sub-optimal patch** for exactly that gap: it
cheaply nudges the model through the changing market across the look-ahead lag, accepting lower
quality because nothing better exists for the last 2 weeks.

The question this study answers: **how often must the look-ahead training run to keep the model on
track** — i.e. how stale can the look-ahead anchor get before the sub-optimal daily patching can no
longer hold the line. The answer fixes the look-ahead retrain cadence (the lag:dailies ratio).

This tool is run **on demand, ~once per codebase increment**, not on a schedule. It may spend CPU
freely (the run must *finish on the droplet*, not be cheap), but RAM is the binding constraint.

### Scope

- **Graphed outcomes: mt1-composite slot0 and MT2 only.** These are the allocation-decision
  outputs. (StockNN and the four MT1 component pools still *evolve* in the daily track — see §3 —
  they're just not graphed.)
- Validity is conditional on production feeding **composite** slot0 to MT2, i.e.
  `MT2_FEED_DIRECTION == false`. By the time this study runs, component pools should show real
  learning curves and the toggle should have flipped back to composite. **The harness must read
  `MT2_FEED_DIRECTION` and refuse/loudly warn if it is still `true`** — otherwise it silently
  measures the wrong pool.

---

## 2. Timeline, lag, and warm-up

Two independent 10-session windows bracket every prediction. Both equal `MT1_FWD_DAYS == 10` /
`MT1_DIR_DAYS == 10` / `MT1_ROLLING_DAYS == 10` — all 10 sessions, i.e. **2 business weeks each**:

| Window | Constant(s) | Why |
|--------|-------------|-----|
| **Forward lag** (2 wk) | `MT1_FWD_DAYS=10` | A prediction's target = cumulative relative return over the next 10 sessions. Nothing made in the last 10 sessions is gradable yet. |
| **Trailing warm-up** (2 wk) | `MT1_DIR_DAYS=10`, `MT1_ROLLING_DAYS=10` | Per-day scores ride 10-session rolling windows (scoring window + adaptive floor/ceiling). Until they fill, scores are cold-start artifacts, not signal. |

Total runway before a track's ±1 curve means anything = **~20 sessions (4 weeks)**. The first
2 weeks of a track's dailies are *establishing the starting position* (filling rolling state), not
a measurement.

### Week structure of one base (the "belt")

A base ages through six week-slots. Counting from its freeze point:

```
slot   role                              forward target?   scored?
week 0 fresh freeze, initial dailies     no (lag)          no  → advance
week 1 +1 week dailies                   no (lag)          no  → advance
week 2 1-week drift base                 yes               yes → advance
week 3 2-week drift base                 yes               yes → advance
week 4 3-week drift base                 yes               yes → advance
week 5 4-week drift base                 yes               yes → discard
```

Weeks 0–1 are the forward-lag pair (generate-only). Weeks 2–5 are gradable. At week 5 the base is
graded a final time and dropped.

> **The four scored slots are four ratio buckets, evaluated rolling — not a progression curve.**
> Weeks 2/3/4/5 correspond to a 1/2/3/4-week lag (heavy-retrain staleness). Every base contributes
> its scored value into whichever bucket its current age maps to, so across the run each bucket
> becomes a **rolling time series**: at each calendar week we read one value per bucket and ask
> *which lag:dailies ratio is doing best this week*, compared across buckets. We are **not** watching
> a single curve converge over total training progression — the axis is calendar weeks, the four
> lines are the four ratios.
>
> **4 weeks is the ceiling by design, not by belt length.** The look-ahead training is the quality
> anchor; daily upkeep is only the sub-optimal patch bridging its inherent 2-week lag. A look-ahead
> anchor over a month stale should never be the best bucket. If the week-5 (4-week-lag) series is
> consistently strongest with no falloff, the answer is **not** "extend the belt to test even staler
> anchors" — it means the **look-ahead training isn't earning its keep**: either the cheap daily
> patch is silently doing all the real work, or the anchor is no better than its own patched-stale
> self. That's a flag to fix the training methodology, not to tolerate month-old anchors.

---

## 3. What a base holds, and what the daily track does

Each base = **two coupled things**, born at the same freeze point and aging together:

1. **Frozen band — the yardstick.** The top-8 elites (slots 0–7) of the composite MT1 pool *and*
   of MT2, captured at freeze time and **never mutated**. Rolled forward each day, the 8 produce a
   per-day **max / mean / min** envelope (same idiom as the existing MT2 max/mean/min graph). The
   band carries its rolling buffers from freeze time, so a base may **only** be frozen from a point
   already past the 20-session warm-up — the harness must refuse to freeze a cold base.

2. **Daily track — the thing under test.** Seeded from the same freeze checkpoint, then run forward
   with the **full production upkeep**: StockNN + all five MT1 pools + MT2. (Full upkeep is required
   for fidelity — StockNN evolution changes the 444-feature vector feeding MT1, so freezing it would
   measure an idealized drift the real pipeline never experiences.) Only its composite slot0 and MT2
   outputs are scored/graphed.

The ±1 metric (§4) measures the **daily track's** score against its base's **frozen band** — i.e.
"what does daily upkeep buy over just freezing this checkpoint," as a function of base age.

---

## 4. The ±1 asymptotic metric

Per day `t`, for a base, score all 8 frozen band models and the daily track against the **realized
10-session-forward actuals** (same leak-free target as MT1 training) → `mean, max, min` (band) and
`S` (track). Map to a saturating relative score:

```
spread_signed = (max - mean)  if S >= mean   else  (mean - min)
rel = tanh( (S - mean) / max(spread_signed, floor) )
```

- `S = mean` → **0** (track tied with the ensemble mean).
- `S` at the band edge (best/worst of 8) → `tanh(1) ≈ 0.76` ("tied with the ensemble's best").
- `rel → ±1` only when the track is several band-widths beyond the whole ensemble's spread —
  i.e. **decisive**. Asymptotic, smooth, magnitude-preserving, no clipping.
- `floor` on the denominator is **mandatory** — right after a fresh retrain the 8 models agree
  tightly and `max - mean → 0` would blow the ratio up. Derive `floor` from the rolling band width
  (the existing `acc_floor` adaptive-floor idiom), not a flat constant.

This metric subsumes the earlier "deviates-but-wins" concern: a track that wanders from the band
but scores better simply reads **positive**, which is the desired signal.

### Reading the result

The result is **four rolling series — week 2 / week 3 / week 4 / week 5** (1/2/3/4-week look-ahead
lag), each plotted over calendar weeks, **not** split per base. At each calendar week, compare the
four buckets: the lag at which `rel` stops being decisively positive is the staleness point = the
longest the look-ahead retrain can be deferred while daily upkeep still holds the line. All four
hugging 0 → daily upkeep buys little, retrain rarely. Week-4 strongly positive while week-2 hugs
0 → retrain roughly every 1–2 weeks. Draw bootstrap confidence bands over the weekly axis; adjacent
buckets (week 3 vs week 4) may sit inside noise.

---

## 5. Implementation: sequential-per-base (memory-bounded)

C++ is **required** (not just optimal) — the run must finish on the droplet, and only the trainer's
`mlock`'d pools + ~6× speedup make ~51 bases × 6 weeks of full-upkeep replay tractable.

The belt is a *conceptual* model. Because this is a **retrospective batch over 255 days of
history** (not a live weekly job), bases are independent and need not be co-resident. Hold **one
base at a time** — the StockNN pool dominates (`STOCKNN_PARAMS = 921,625`; a 200-slot industry pool
is ~737 MB of raw weights, ~74 MB resident via the 20-slot elite buffer). Six concurrent full tracks
would not fit; one at a time is comfortable. Trade CPU (freed by the once-per-increment cadence) for
RAM (tight).

**State isolation (non-destructive).** Run only once models are **fully trained / mature**. The
study seeds every base from a **read-only copy** of the fully-trained production models and writes
**only** to a scratch output dir (real disk — *not* `/tmp` tmpfs). Every base and daily track it
generates is **throwaway**; nothing is retained. Production model dirs are never modified — the
models that *start* the test remain the ones used for live trading and production upkeep afterward.

```
seed = read-only copy of fully-trained production models      # never written back
for freeze_week in 0 .. last_valid:                 # slide the freeze point through the rolling window
    assert freeze_week past 20-session warm-up
    assert MT2_FEED_DIRECTION == false
    band  = freeze slots 0-7 of (composite MT1, MT2)           # immutable yardstick
    track = seed full upkeep (StockNN + 5 MT1 pools + MT2)     # throwaway
    for age_week in 0 .. 5:
        run 5 daily upkeep steps on track
        if age_week >= 2:                            # past forward lag → gradable bucket
            bucket = age_week                        # 2/3/4/5 → the four rolling series
            for day in week:
                score band(8) and track vs realized 10-fwd actuals
                rel = tanh(...)                      # §4
                log record (calendar_week, bucket, day, rel, band stats, track score)
    discard band + track                             # ONE base resident; nothing kept
# graphs assemble the four series by (bucket) over calendar_week, across all bases
```

Output is identical to the lockstep conveyor; only RAM differs. The bottom-up belt-shift remains the
right mental model for *aging*; sequential is how it's *realized*.

---

## 6. Drift log + graphs

- **Log:** extend the existing binary-log machinery rather than a parallel system — `read_mt_log.py`
  and the SVG plotters stay the single read path. Either a new record type in `mt_training_log.bin`
  or a sidecar `drift_log.bin`. Per record: `freeze_week, calendar_week, bucket(2–5), day_index,
  target='composite'|'mt2', rel, band_mean, band_max, band_min, track_score`.
- **Graphs (composite + MT2 each):** four series — **week 2 / week 3 / week 4 / week 5** (the four
  lag buckets) — plotted over the rolling calendar-week axis, **not** split per base. Each base
  contributes its scored value into whichever bucket its current age maps to (a base born at week W
  feeds week-2 at calendar W+2, week-3 at W+3, …), so the buckets accumulate across all bases into
  four rolling lines. ±1 drawn as reference gridlines, bootstrap confidence bands, warm-up region
  (first ~20 sessions) greyed out. Reuse `plot_training.py` styling (max/mean/min colors already
  exist for MT2).

---

## 7. Locked decisions

| Decision | Value |
|----------|-------|
| Graphed scope | mt1-composite slot0 + MT2 |
| Production feed assumption | `MT2_FEED_DIRECTION == false` (assert) |
| Drift base | slots 0–7, frozen (immutable), max/mean/min band |
| Daily track | full production upkeep (StockNN + 5 MT1 pools + MT2) |
| Daily track per base | yes (own track per freeze checkpoint) |
| Metric | `tanh((S-mean)/max(spread,floor))`, ±1 asymptotic |
| Denominator floor | adaptive (rolling band width), mandatory |
| Runway | 20 sessions (10 fwd lag + 10 rolling fill) |
| Belt | 6 slots; weeks 0–1 generate-only, 2–5 scored, 5 discarded |
| Output | four rolling series (week 2/3/4/5 lag buckets) over calendar weeks, not per-base |
| Implementation | C++ `--drift-study`, sequential-per-base (1 resident) |
| State isolation | seed = read-only copy of fully-trained production models; all generated models throwaway; production state untouched |
| Cadence | on demand, ~once per codebase increment, only when models are mature |

## 8. Open / future

- Belt length is fixed at 6 (4-week ceiling) by design — a month-stale anchor winning is a
  methodology red flag (see §2), not a reason to test staler anchors.
- Multiple sub-weekly daily cadences (daily vs every-2/3-days) — deferred; current study fixes the
  *lag* (heavy-retrain interval) with daily-cadence upkeep, not the daily step frequency.
- Statistical power: ~51 weekly points over 255 days; enough to rank, bootstrap bands required for
  adjacent ratios.
