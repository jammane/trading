// mt1_pool.h — MT1 scoring and pool lifecycle for the single-output rebuild.
//
// Replaces the head/tail, five-pool, 10-day-forward design. One network per industry emitting ONE
// number — predicted next-day StockNN P&L in dollars — trained in ONE pool of persistent
// individuals.
//
// WHY THE REBUILD (measured, not assumed):
//   * Nothing predicted the old 10-day forward relative-return target, INCLUDING the control that
//     MT1 already receives as features[10..16] (rank IC t -0.17/-0.28/-0.52, control t -0.10,
//     n=123 non-overlapping). Direction scored 52.25% OOS against 83.66% in-sample with negative
//     skill in all 12 industries, across two architectures.
//   * A 10-day window over a 10-day-forward target overlaps 9-of-10 — about 1.6 independent
//     observations. Selecting best-of-200 on that is close to pure selection-on-noise.
//   * conf4 was ungraded and fed MT2 a constant; the old range channel was graded against the
//     delta head's own residual (noise by construction) and converged to a constant.
//   * With ONE output the shared trunk has one consumer, so the head/tail split is indirection.
//
// SCORING — anchored to the naive predictor, which is the point:
//
//     score = d / (|actual - predicted| + d),   d = max(|actual - baseline|, floor)
//
//     1.0  perfect
//     0.5  EXACTLY as good as predicting the trailing mean
//     ->0  much worse than doing nothing
//
// Anchoring d to the BASELINE's error rather than to |actual| is what makes 0.5 mean "learned
// nothing". Under the old `|actual|` denominator a collapsed pool also scored ~0.5, and 0.5 looks
// like a respectable number — which is how conf4 and the range channel survived for years with
// dead pools. Here a collapse is legible in the score itself, and the windowed mean IS the
// acceptance gate: above 0.5 means it beats the trailing mean out-of-sample.
//
// L1 rather than squared error: |next-day P&L| runs p10 $86 to p90 $1650, a 19x spread with fat
// tails, so squared error would let one day dominate a window. L1 targets the conditional median.
//
// The floor is required: p10 of |actual| is $86 and the baseline's error can be near zero on a day
// the trailing mean happens to land, which would make the score hypersensitive.
//
// Expect scores near 0.5. Measured on 14,819 industry-days, next-day P&L has mean $24.9 against
// sd $1121 — mean/sd = 0.022, very nearly pure noise. A correct estimator SHRINKS toward the mean
// when conditioning buys little, so near-constant output is right, not broken. What distinguishes
// a working pool is a windowed mean reliably above 0.5, not impressive-looking predictions.
#ifndef MT1_POOL_H
#define MT1_POOL_H

#include <cmath>
#include <cstdint>

// ── Output decode ────────────────────────────────────────────────────────────────
// tanh keeps the raw logit bounded; the scale sets the dollar range the net can express.
// $10,000 is inherited from the old delta head, which DID work at this scale (it was the one
// channel that measured as functional). Unlike the old vol channel there is no softplus tail to
// get stuck in: tanh is symmetric and its gradient is largest near zero, where this target lives.
static constexpr float MT1_PRED_SCALE = 10000.f;

static inline float mt1_pred(float raw) { return tanhf(raw) * MT1_PRED_SCALE; }

// ── Baseline and floor windows ───────────────────────────────────────────────────
//
// CAUSALITY CONTRACT — both windows MUST end at day t, the same information the model had.
//
// MT1 predicts on day t from day-t features; `actual` is the t->t+1 change, known at t+1. So a
// day-t prediction is scored at t+1: a ONE-day delay, the minimum possible, against the old
// design's 10 (and 20 for vol) which is what left a 10-day window holding ~1.6 independent
// observations.
//
// The prediction path cannot look ahead. `baseline` and `floor` can, because both are windowed
// statistics over the very series being predicted:
//
//   * a baseline whose window includes day t+1 has seen the answer, so it would beat the model
//     for free and drive the score below 0.5 regardless of skill;
//   * a floor whose window includes day t+1 lets a large move inflate its own denominator,
//     flattering the score exactly on the days that matter most.
//
// Both must be computed from days <= t, BEFORE the day's scoring — the same ordering the old
// acc_floor used. mt1_windows_are_causal() below is the assertion to call at the point of use.
static constexpr int MT1_BASELINE_DAYS = 20;   // trailing mean = the predictor MT1 must beat
static constexpr int MT1_FLOOR_DAYS    = 10;   // rolling window behind the score floor
static constexpr float MT1_FLOOR_FRAC  = 0.5f; // floor = mean(|actual|) * frac, per industry

// Guard for the call site: the last day feeding either window must be strictly before the day
// whose outcome is being scored. Cheap enough to assert every day.
static inline bool mt1_windows_are_causal(int scored_day, int baseline_last_day,
                                          int floor_last_day) {
    return baseline_last_day < scored_day && floor_last_day < scored_day;
}

// ── Per-day score ────────────────────────────────────────────────────────────────
// `baseline` is the trailing-mean prediction for the same day; `floor` guards a near-zero
// denominator. Symmetric in over/under by construction, bounded in (0,1], no singularity.
static inline float mt1_score(float actual, float predicted, float baseline, float floor_v) {
    const float err  = fabsf(actual - predicted);
    const float base = fabsf(actual - baseline);
    const float d    = fmaxf(base, fmaxf(floor_v, 1e-6f));
    return d / (err + d);
}

// Skill against the baseline over a window, in the same units the acceptance gate uses:
// >0 beats the trailing mean, 0 ties it, <0 worse. Reported alongside the score so a pool that
// has collapsed onto the mean cannot hide behind a healthy-looking average.
static inline float mt1_skill(float sse_model, float sse_baseline) {
    return sse_baseline > 1e-9f ? 1.f - sse_model / sse_baseline : 0.f;
}

// ── Pool lifecycle ───────────────────────────────────────────────────────────────
// Forward accumulation, carried over from the v0.5.0.0 direction rebuild. That design was right
// and aimed at the wrong target: with a 10-day-forward target its 16-prediction register held
// about 2 independent facts. A next-day target makes all 16 independent, which is what it was
// built for. Every prediction is out-of-sample by construction — the model was selected before
// the outcome existed — so the old block-start snapshot machinery is no longer needed.
static constexpr int   MT1_POOL_SLOTS    = 200;
static constexpr int   MT1_SCORE_HIST    = 16;     // rolling per-model score window
static constexpr int   MT1_POOL_MIN_AGE  = 8;      // predictions before a model may be culled OR breed
static constexpr float MT1_POOL_CULL_PCT = 0.083f; // fraction of MATURE models culled per day
static constexpr float MT1_POOL_ELITE_PCT = 0.10f; // top fraction of mature models used as parents
static constexpr float MT1_POOL_LINEAGE_CAP    = 0.125f; // lineage share that stops breeding
static constexpr float MT1_POOL_LINEAGE_RESUME = 0.10f;  // and resumes below this (hysteresis)

// Recency weights in blocks of 4 over the 16-slot window, newest block first.
static constexpr float MT1_RECENCY_W[4] = {1.0f, 0.8f, 0.6f, 0.4f};

// A persistent individual. The old version stored a 16-BIT register because direction was
// binary; a continuous score needs the values themselves.
struct MT1SlotMeta {
    float    score[MT1_SCORE_HIST];  // index 0 = most recent
    uint16_t n_pred;                 // predictions made, saturating at MT1_SCORE_HIST for weighting
    uint32_t lineage;                // inherited id; blends and fresh inits get a new one
};

static inline void mt1_slot_init(MT1SlotMeta& m, uint32_t lineage) {
    for (int i = 0; i < MT1_SCORE_HIST; i++) m.score[i] = 0.f;
    m.n_pred  = 0;
    m.lineage = lineage;
}

// Push today's score, shifting the window. n_pred saturates so a long-lived model is still
// judged on its most recent MT1_SCORE_HIST predictions.
static inline void mt1_slot_record(MT1SlotMeta& m, float score) {
    for (int i = MT1_SCORE_HIST - 1; i > 0; i--) m.score[i] = m.score[i - 1];
    m.score[0] = score;
    if (m.n_pred < 0xFFFF) m.n_pred++;
}

static inline bool mt1_slot_mature(const MT1SlotMeta& m) {
    return m.n_pred >= (uint16_t)MT1_POOL_MIN_AGE;
}

// Recency-weighted mean over however much of the window is actually occupied — an 8-prediction
// model is judged on its 8, not penalised for the empty slots.
static inline float mt1_slot_score(const MT1SlotMeta& m) {
    const int n = m.n_pred < (uint16_t)MT1_SCORE_HIST ? (int)m.n_pred : MT1_SCORE_HIST;
    if (n <= 0) return 0.f;
    float num = 0.f, den = 0.f;
    for (int i = 0; i < n; i++) {
        const float w = MT1_RECENCY_W[i / 4];
        num += w * m.score[i];
        den += w;
    }
    return den > 0.f ? num / den : 0.f;
}

// Plain mean over the occupied window — the secondary key, and the honest number to report:
// unlike the recency-weighted score it cannot be inflated by one good recent day.
static inline float mt1_slot_mean(const MT1SlotMeta& m) {
    const int n = m.n_pred < (uint16_t)MT1_SCORE_HIST ? (int)m.n_pred : MT1_SCORE_HIST;
    if (n <= 0) return 0.f;
    float s = 0.f;
    for (int i = 0; i < n; i++) s += m.score[i];
    return s / (float)n;
}

// Sort key: recency-weighted first, plain mean as tie-break. Immature models rank last regardless
// of score — a model with two lucky predictions must not displace one with a record.
static inline bool mt1_slot_better(const MT1SlotMeta& a, const MT1SlotMeta& b) {
    const bool ma = mt1_slot_mature(a), mb = mt1_slot_mature(b);
    if (ma != mb) return ma;
    const float sa = mt1_slot_score(a), sb = mt1_slot_score(b);
    if (sa != sb) return sa > sb;
    return mt1_slot_mean(a) > mt1_slot_mean(b);
}

#endif  // MT1_POOL_H
