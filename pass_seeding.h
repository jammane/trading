// pass_seeding.h — pure logic for the pass-boundary seeding decision (v0.6.3.0).
//
// Lives in a header, like mt1_pool.h, so tests/test_pass_seeding.cpp grades the same arithmetic
// production runs. A replica drifted from production once already and the tests stayed green while
// testing a function that no longer existed; nothing here is duplicated anywhere.
//
// No I/O, no globals, no allocation. The file-shuffling half lives in training_v4.cpp.
//
// See PASS_SEEDING.md for the design and the measurements behind it.
#pragma once

#include <cmath>

// ── The champion judging metric (pass_reference v2) ──────────────────────────────────────────
// A finishing pass is judged against the standing champion on slot 0's RECENCY-WEIGHTED MEAN DAILY
// BOOK RETURN over the whole pass -- the same criterion the MT1 race's allocation lines use
// (ALLOC_RECENCY in training_v4.cpp is defined from PASS_JUDGE_RECENCY, so the two cannot drift):
//
//   r_d   = (slot0_score - book_prev) / book_prev          one session's book return
//   w_d   = PASS_JUDGE_RECENCY ^ (days before the pass's last day)     half-life ~138 days
//   score = sum(w_d r_d) / sum(w_d)
//
// v1 judged the percent change in slot 0's book over the LAST 15 DAYS. That measured no better
// than chance: 13 dethronings against a null expectation of 13.0. Daily book volatility is ~1.5-2%,
// so a 15-day return has sd ~7% against skill differences of ~1.5% -- SNR ~0.2. A hard-floor reset
// inside the window also jumped the book $22.4k -> $25k, a spurious +11.6%.
//
// v2 fixes both. The decay keeps an effective sample of ~400 days while leaning toward the end of
// the pass, which is what the seeded models are: v1's reason for a short window -- the pool at the
// end of a pass is not the mid-pass pool, which has long since been culled -- is answered by the
// weighting rather than by discarding 95% of the pass. A reset day has book_prev == slot0_score by
// construction, so it contributes a return of exactly 0 instead of a jump. Summing RETURNS rather
// than dollars keeps a pass that grew its book from being judged on a bigger base.
//
// pass_share_new only uses the two scores' signs and ratio, so it works unchanged on this scale.
static constexpr double PASS_JUDGE_RECENCY = 0.995;

// A pass shorter than this is not judged (smoke runs, short diagnostics).
static constexpr int PASS_JUDGE_MIN_DAYS = 15;

// pass_reference.csv schema version. Bump when columns change -- OR WHEN WHAT A COLUMN MEANS
// CHANGES: v2 kept every column but champ_pct/chal_pct/new_champ_pct now hold the v2 score (a
// mean daily return, ~1e-3) instead of a 15-day percent change (~1e-2). Mixing the two would crown
// on a units mismatch, so a file of another version is set aside, never read.
static constexpr int PASS_REF_VERSION = 2;

struct PassJudge {
    double w = 0.0, w2 = 0.0, wr = 0.0;
    long   n = 0;
    // book_prev / book_now: slot 0's reference book at today's close and its post-trade book at
    // the next close. age: days before the pass's last day (0 on the last day).
    void add(double book_prev, double book_now, int age) {
        if (!(book_prev > 1.0) || !std::isfinite(book_now)) return;
        const double wt = std::pow(PASS_JUDGE_RECENCY, (double)(age > 0 ? age : 0));
        w += wt; w2 += wt * wt; wr += wt * (book_now - book_prev) / book_prev;
        n++;
    }
    bool   have()     const { return w > 0.0; }
    double score()    const { return w > 0.0 ? wr / w : 0.0; }
    double eff_days() const { return w2 > 0.0 ? w * w / w2 : 0.0; }   // Kish effective sample
};

// Fraction of the seed slots that should come from the CHALLENGER — the pass that just finished —
// given each side's percent portfolio change over the judging window.
//
// Direction is sign(chal - champ) in every branch; only the magnitude differs:
//   both positive : chal / (champ + chal)
//   one negative  : everything to the positive side
//   both negative : |champ| / (|champ| + |chal|)  — the SMALLER loss takes the LARGER share
//
// Measured on the v0.6.2.1 run: only the both-positive branch carries direction signal (60.6%);
// one-negative is 51.6% and both-negative 48.7%. The proportional form is what makes that
// survivable — under noise it lands near 50/50 instead of flipping categorically.
static inline double pass_share_new(double champ_pct, double chal_pct)
{
    if (champ_pct > 0.0 && chal_pct > 0.0) {
        double d = champ_pct + chal_pct;
        return (d > 1e-12) ? chal_pct / d : 0.5;
    }
    if (chal_pct > 0.0)  return 1.0;      // challenger positive, champion not
    if (champ_pct > 0.0) return 0.0;      // champion positive, challenger not
    double a = std::fabs(champ_pct), b = std::fabs(chal_pct);
    double d = a + b;
    return (d > 1e-12) ? a / d : 0.5;     // both <= 0, or both exactly 0
}

// Proportional interleave of two rank-ordered sets into n output slots, preserving each set's
// internal order. Two Bresenham-style accumulators race; whichever is further ahead when one
// crosses 100 emits next.
//
//   src[k] = 0 -> take champion's model idx[k];  1 -> take challenger's model idx[k]
//
// Order preservation matters twice over: the best of both sets land in the low slots, and because
// StockNN allocates mutation children uniformly (9 per parent, mut_i / MUTATIONS_PER_PARENT), the
// 180 children inherit the seed proportion automatically.
static inline void pass_interleave(double share_new, int n, int* src, int* idx)
{
    const double bpct = share_new * 100.0;   // challenger
    const double apct = 100.0 - bpct;        // champion
    int apos = 0, bpos = 0;
    double aspd = 0.0, bspd = 0.0;

    for (int k = 0; k < n; k++) {
        // Both pct values cannot be zero (they sum to 100), so one accumulator always advances.
        // The counter is belt-and-braces against a NaN share slipping through.
        int guard = 0;
        while (aspd < 100.0 && bspd < 100.0) {
            aspd += apct;
            bspd += bpct;
            if (++guard > 1000) { bspd = 100.0; break; }
        }
        // apos + bpos == k < n at every step, so neither index can run past the source arrays.
        if (aspd > bspd) { src[k] = 0; idx[k] = apos++; aspd -= 100.0; }
        else             { src[k] = 1; idx[k] = bpos++; bspd -= 100.0; }
    }
}

// Label for the pass_reference.csv `regime` column — which branch of pass_share_new fired.
static inline const char* pass_regime(double champ_pct, double chal_pct)
{
    if (champ_pct > 0.0 && chal_pct > 0.0) return "both+";
    if (champ_pct > 0.0 || chal_pct > 0.0) return "one-";
    return "both-";
}
