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

// Trailing window used to judge a finishing pass against the standing champion. Deliberately
// near-term: the pool at a pass end is not the pool that existed mid-pass — those models are long
// since culled — so a full-pass percentage measures the trajectory rather than the models that
// actually seed the next pass.
static constexpr int PASS_JUDGE_DAYS  = 15;

// pass_reference.csv schema version. Bump when columns change.
static constexpr int PASS_REF_VERSION = 1;

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
