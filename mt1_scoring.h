// mt1_scoring.h — MT1 output decode + per-day score breakdown.
//
// Single source of truth, included by BOTH training_v4.cpp and tests/test_mt1.cpp. It exists
// because the test file previously carried its own hand-copied replica of compute_mt1_scores,
// which silently drifted from production on three counts (a different sc_acc formula, the retired
// 0.50/0.33/0.17 composite, and only 3 of 4 components) — so the scoring tests were green while
// testing a function that no longer existed. Keep this header pure arithmetic: no BLAS, no I/O,
// no globals, so the test target keeps linking against nothing but libc/libm.
//
// Python mirror: upkeep._mt1_conf / _mt1_range_pct / _mt1_delta_t / _mt1_score_breakdown.

#ifndef MT1_SCORING_H
#define MT1_SCORING_H

#include <cmath>
#include <cstdint>

// ── Constants ────────────────────────────────────────────────────────────────────
static constexpr float MT1_SCALE_DOLLARS   = 10000.f;  // tanh(out[1]) × scale = dollar P&L prediction
// Channel 2 is forward realized VOLATILITY as of v0.5.0.0, decoded as
// softplus(out[2]) × MT1_SCALE_DOLLARS so size and risk share units and MT2 receives a comparable
// pair. The retired band scale (r = range_pct × max(|delta_d|, acc_floor)) and its ceiling went
// with it: a band graded against the delta head's own residual is graded against noise.
static constexpr int   MT1_VOL_DAYS        = 20;       // forward horizon for the volatility target
// Volatility gets its OWN scale, not MT1_SCALE_DOLLARS. The measured target averages $242 (p90
// $336), so at the $10,000 delta scale a model would have to hold softplus in 0.005-0.06 — deep in
// its flat tail, where Gaussian mutations barely move the output. That is the same squashing that
// pinned the magnitude head. At $500 the useful range is softplus 0.1-1.2 (raw logit -2.2..+0.9),
// well conditioned, and a randomly initialised tail already lands within ~1.5x of the target
// instead of ~31x above it.
static constexpr float MT1_VOL_SCALE       = 500.f;
// What the ungraded channel-3 tail forwards to MT2. A frozen-but-arbitrary function would be
// structured noise MT2 could fit; a constant cannot carry information either way. (0.0 would be
// strictly more inert — it zeroes the weights reading it — but 0.5 keeps the channel mid-range if
// it is ever re-graded.)
static constexpr float MT1_UNGRADED_FEED   = 0.5f;
// Pre-activation cap for the sigmoid outputs (direction conf + conf4). Without it the raw logits
// blow up, sigmoid saturates to exactly 0/1, and weight mutations stop changing the output at all —
// the pool freezes genetically (96.7% of conf values were exactly 0 or 1 by pass 5 of the v0.4.0.0
// run, with best−min spread of 0.35% across 200 slots). Cap 4 → conf ∈ (0.018, 0.982), always mobile.
static constexpr float MT1_LOGIT_CAP       = 4.f;
static constexpr float MT1_SOFTPLUS_CLAMP  = 20.f;     // guards softplus(raw) → inf

// ── MT1 output decode ────────────────────────────────────────────────────────────
// Every consumer of an MT1 raw logit MUST go through these — scoring, the MT2 in48 build, the
// drift study, and logging — so the bounded activation is impossible to bypass at one site.
static inline float mt1_sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// conf/conf4 squash the logit through tanh before the sigmoid so the output can never reach the
// 0/1 rails, where the derivative vanishes and Gaussian weight mutations stop moving the output.
static inline float mt1_conf(float raw) {
    return mt1_sigmoidf(MT1_LOGIT_CAP * tanhf(raw / MT1_LOGIT_CAP));
}
static inline float mt1_conf4(float raw) { return mt1_conf(raw); }
static inline float mt1_range_pct(float raw) {
    float c = fminf(fmaxf(raw, -MT1_SOFTPLUS_CLAMP), MT1_SOFTPLUS_CLAMP);
    return log1pf(expf(c));                       // softplus, guarded against overflow → inf
}
static inline float mt1_delta_t(float raw) { return tanhf(raw); }   // already bounded; unchanged

// ── Per-day score breakdown ──────────────────────────────────────────────────────

struct MT1ScoreBreakdown { float composite, direction, range, accuracy, confidence; };

// acc_floor: per-industry adaptive floor for the accuracy denom (half the 10-day mean |actual_d|)
// vol_actual: realized volatility of the industry's portfolio over the next MT1_VOL_DAYS sessions,
//             scaled by MT1_SCALE_DOLLARS to match vol_pred
// vol_floor:  half the 10-day rolling mean of vol_actual, so 2×vol_floor is the naive predictor
static inline MT1ScoreBreakdown compute_mt1_scores(
    float actual_d, float vol_actual, const float raw4[4], float acc_floor, float vol_floor)
{
    float conf      = mt1_conf(raw4[0]);
    float delta_d   = mt1_delta_t(raw4[1]) * MT1_SCALE_DOLLARS;
    float vol_pred  = mt1_range_pct(raw4[2]) * MT1_VOL_SCALE;       // channel 2 is VOLATILITY now
    float conf4     = mt1_conf4(raw4[3]);                           // present but UNGRADED

    float sc_dir = (actual_d >= 0.f) ? conf : (1.f - conf);

    // Magnitude: signless, so the delta head owns size and the direction head owns sign.
    float err    = fabsf(fabsf(actual_d) - fabsf(delta_d));
    float denom  = fmaxf(fabsf(actual_d), acc_floor);
    float sc_acc = denom / (err + denom);

    // Volatility (v0.5.0.0). Channel 2 used to be a band width graded against the delta head's own
    // residual — i.e. against the part of the target the model had just failed to predict, which is
    // noise by construction, so the pool correctly converged to a constant (range_pct 0.62, spread
    // 2.2%). It now predicts the industry's forward realized volatility, which IS predictable: a
    // 40-day trailing vol reaches OOS r = 0.444 on this target, and the same pool machinery scored
    // on a full window reaches 0.527. Same shape as sc_acc so the two are directly comparable.
    float err_vol = fabsf(vol_pred - vol_actual);
    // Guarded: a degenerate all-zero vol history makes vol_actual and vol_floor both 0, and an
    // unguarded den_vol turns sc_vol into 0/0 = NaN, which then poisons the composite and every
    // selection downstream of it for the rest of the run.
    float den_vol = fmaxf(fmaxf(vol_actual, vol_floor), 1e-6f);
    float sc_vol  = den_vol / (err_vol + den_vol);

    // conf4 is no longer scored. It graded itself against the band geometry that just went away,
    // and with err > r always its optimum was the degenerate conf4 = 0 (89% of values < 0.01,
    // pool spread 0.05%). The tail stays in the model so file formats and MT1NN are unchanged.
    float sc_cfd = 0.f;
    (void)conf4;

    // Secondary [0,1] normalization per component vs its naive baseline B (ideal = 1):
    //   sec = clamp((raw - B) / (1 - B), 0, 1)  → naive maps to 0, ideal maps to 1.
    // Equal-weight composite gives each axis equal influence; raw scores sit at very different
    // levels (confidence ≈1, range ≈0.85, acc/dir ≈0.5) and would otherwise weight themselves.
    auto clamp01 = [](float x){ return fminf(fmaxf(x, 0.f), 1.f); };

    float b_dir   = 0.5f;                                       // conf = 0.5, no directional view
    float sec_dir = clamp01((sc_dir - b_dir) / fmaxf(1.f - b_dir, 1e-6f));

    float b_acc   = denom / (fabsf(actual_d) + denom);           // predict-0 null model
    float sec_acc = clamp01((sc_acc - b_acc) / fmaxf(1.f - b_acc, 1e-6f));

    // Naive vol predictor = the rolling mean, which is 2 × vol_floor by construction.
    float err_nv  = fabsf(2.f * vol_floor - vol_actual);
    float b_vol   = den_vol / (err_nv + den_vol);
    float sec_vol = clamp01((sc_vol - b_vol) / fmaxf(1.f - b_vol, 1e-6f));

    // Three graded channels, not four — see sc_cfd above.
    float composite = (sec_dir + sec_acc + sec_vol) * (1.f / 3.f);
    return {composite, sc_dir, sc_vol, sc_acc, sc_cfd};
}

// ── Direction pool: forward accumulation (v0.5.0.0) ──────────────────────────────
// The other three tails score a model by REPLAYING it over the last MT1_DIR_DAYS stored days. At
// MT1_FWD_DAYS = 10 those targets overlap 9-of-10, so a 10-day window carries ~1.6 independent
// observations — the v0.4.3.0 out-of-sample instrument measured direction at 49.54% OOS against
// 61.09% in-sample, i.e. all of the apparent skill was fitting that window. Direction now keeps 200
// PERSISTENT individuals, each carrying its own rolling 16-prediction record, culled gently on a
// percentile floor. Scoring lives here rather than in training_v4.cpp so the tests grade the same
// arithmetic production does — a hand-copied replica is exactly what this header exists to prevent.

static constexpr int   MT1_DIR_HIST_BITS   = 16;     // rolling record length (2 bytes), bit 0 = newest
static constexpr int   MT1_DIR_MIN_AGE     = 8;      // predictions before a model may be culled OR breed
static constexpr float MT1_DIR_CULL_PCT    = 0.083f; // fraction of MATURE models culled per day
// Steady state mature/N = 1/(1 + MIN_AGE × CULL_PCT) → ~60% mature (120 of 200), ~10 births/day,
// ~20-prediction lifespan.
static constexpr float MT1_DIR_ELITE_PCT   = 0.10f;  // top fraction of mature models used as parents
// Tightened 0.25 → 0.125 after the --dir-reps sweep: at 20 reps the pool ran a 21% max lineage
// share with only 14.5 distinct lineages, against 3.7% / 77.7 at 1 rep. Twenty cull-and-breed
// rounds per calendar day converge lineages ~5× faster, so the cap has to bind correspondingly
// harder to keep the diversity the gentle cull was meant to buy.
static constexpr float MT1_DIR_LINEAGE_CAP = 0.125f;  // lineage above this share stops breeding
// Re-enable only once the share falls well clear of the cap. Without hysteresis a lineage sitting
// at the cap flips barred/unbarred on almost every rep — the first --dir-reps 20 run logged 1850
// barrings and 1838 re-enables over 35 days, so the bar never bound long enough to suppress
// anything. The gap stays proportional to the cap (20% of it), as it was at 0.25/0.20.
static constexpr float MT1_DIR_LINEAGE_RESUME = 0.10f;
// Recency weights for the secondary score, most-recent-first, in blocks of 4.
// A full 16-slot record sums to 4×(1.0+0.8+0.6+0.4) = 11.2.
static constexpr float MT1_DIR_RECENCY_W[4] = {1.0f, 0.8f, 0.6f, 0.4f};

// Per-slot identity. Everywhere else in the trainer a "slot" is positional — the pool ranks
// candidates and memcpy's them over the elite buffer, so slot k just means "whoever placed k-th
// today" and nothing survives a day. Direction needs the opposite: a model that persists,
// accumulates its own record, and is replaced only when culled. Whenever direction weights move
// between slots, this must move with them.
struct DirSlotMeta {
    uint16_t hist;      // rolling record: bit i = the prediction i steps back was correct (bit 0 newest)
    uint16_t n_pred;    // predictions made = age (saturating). correct + incorrect.
    uint16_t lineage;   // inherited from the mutation parent; blends and fresh inits get a new id
    uint16_t pad;
};

// Score one rolling record. n = min(n_pred, 16) occupied slots; a partial record normalizes by the
// weight actually occupied rather than the full 11.2, so an 8-prediction model is judged on its 8.
static inline void mt1_dir_record_scores(uint16_t hist, uint16_t n_pred,
                                         float* primary, float* secondary, float* tertiary)
{
    int n = (n_pred < (uint16_t)MT1_DIR_HIST_BITS) ? (int)n_pred : MT1_DIR_HIST_BITS;
    if (n <= 0) { *primary = 0.f; *secondary = 0.f; *tertiary = 0.f; return; }
    int   correct = 0;
    float wsum = 0.f, wcorrect = 0.f;
    for (int i = 0; i < n; i++) {
        float w  = MT1_DIR_RECENCY_W[i / 4];
        bool  ok = ((hist >> i) & 1u) != 0u;
        if (ok) { correct++; wcorrect += w; }
        wsum += w;
    }
    *primary   = (float)correct / (float)n;
    *secondary = (wsum > 0.f) ? (wcorrect / wsum) : 0.f;
    // Tertiary is a deterministic function of the other two, so as a THIRD sort key it can never
    // break a tie they left. Kept because the blend distribution is worth logging.
    *tertiary  = 0.4f * (*primary) + 0.6f * (*secondary);
}

// Descending (primary, secondary, tertiary).
static inline bool mt1_dir_better(const DirSlotMeta& a, const DirSlotMeta& b) {
    float ap, as, at, bp, bs, bt;
    mt1_dir_record_scores(a.hist, a.n_pred, &ap, &as, &at);
    mt1_dir_record_scores(b.hist, b.n_pred, &bp, &bs, &bt);
    if (ap != bp) return ap > bp;
    if (as != bs) return as > bs;
    return at > bt;
}

#endif  // MT1_SCORING_H
