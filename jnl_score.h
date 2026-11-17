// jnl_score.h -- the daily score of an ORDER entry (ARCH_JNL, training_v4.cpp).
//
// A net outputs one value per industry; their sort is its predicted order of tomorrow's industry
// returns, higher = better. The score is lexicographic:
//
//   primary    hits - egregious errors (-4..4): how many of its first 4 are in the realised first 4,
//              minus how many are in the realised LAST 4 (a predicted winner that was a bottom-4
//              loser, or a predicted loser that was a top-4 winner). `top` = true scores the top end
//              (winners, LOSS_LDEC); false the bottom end (losers, LOSS_LINC). Only the front of the
//              net's list -- its own first 4 -- is scored.
//   secondary  pairwise comparisons in its order that held (0..66 for 12): position 1 against the
//              11 below it, 2 against the 10 below, ... Every pair once, so the same in either
//              direction.
//   tertiary   Pearson correlation of the raw outputs with the realised returns (-1..1; 0 when
//              either side is constant).
//
// Tied outputs carry no order, so they are scored as the expectation over a random order within
// the tie: an industry tied across the 4th place is in the first 4 with probability (places left)
// / (tied), and a tied pair counts 1/2. A net that outputs one constant therefore scores the
// random-guess value (hits 16/12 - egregious 16/12 = 0, 33, 0), never a lucky universe-order answer.
//
// jnl_composite packs the three into the one float the pool's score register holds, so that
// ranking by the register mean is lexicographic up to rounding over a full 16-day register:
// the largest possible difference in mean secondary (66 x JNL_W2) is below the smallest step in
// mean primary (1/16; primary steps by whole points, or tie fractions), and the tertiary's whole range is below the smallest step in mean
// secondary (half a pair / 16).
#pragma once

#include <cmath>

static constexpr int    JNL_K  = 4;                                   // the first 4
static constexpr double JNL_W2 = 1.0 / (66.0 * 17.0);                 // secondary weight
static constexpr double JNL_W3 = JNL_W2 * 0.5 / 16.0 / 1.05;          // tertiary weight (on 0..1)

struct JNLScore {
    double primary = 0.0, secondary = 0.0, tertiary = 0.0;
    double hits = 0.0, egregious = 0.0;   // primary = hits - egregious (each 0..4)
};

// pred, real: n values (n <= 12). Every value must be finite.
static inline JNLScore jnl_score(const float* pred, const float* real, int n, bool top) {
    JNLScore sc;
    const double sgn = top ? 1.0 : -1.0;
    // Primary. Realised first and last 4: strict order, ties broken by index (a realised tie is
    // a measure-zero event in returns; a deterministic break keeps the score exact).
    bool in_first[16] = {}, in_last[16] = {};
    for (int i = 0; i < n; i++) {
        int ahead = 0, behind = 0;
        for (int j = 0; j < n; j++) {
            const double a = sgn * real[j], b = sgn * real[i];
            if (a > b || (a == b && j < i)) ahead++;
            if (a < b || (a == b && j > i)) behind++;
        }
        in_first[i] = ahead < JNL_K;
        in_last[i]  = behind < JNL_K;
    }
    // Membership of industry i in the net's first 4, as an expectation over ties.
    for (int i = 0; i < n; i++) {
        if (!in_first[i] && !in_last[i]) continue;
        int greater = 0, equal = 0;   // equal counts i itself
        for (int j = 0; j < n; j++) {
            const double a = sgn * pred[j], b = sgn * pred[i];
            if (a > b) greater++;
            else if (a == b) equal++;
        }
        const int left = JNL_K - greater;
        const double mem = left <= 0 ? 0.0 : left >= equal ? 1.0 : (double)left / equal;
        if (in_first[i]) sc.hits += mem;
        if (in_last[i])  sc.egregious += mem;
    }
    sc.primary = sc.hits - sc.egregious;
    // Secondary: concordant pairs; a tie on either side is half.
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            const double dp = (double)pred[i] - pred[j], dr = (double)real[i] - real[j];
            if (dp == 0.0 || dr == 0.0) sc.secondary += 0.5;
            else if ((dp > 0.0) == (dr > 0.0)) sc.secondary += 1.0;
        }
    // Tertiary: Pearson correlation.
    double mp = 0.0, mr = 0.0;
    for (int i = 0; i < n; i++) { mp += pred[i]; mr += real[i]; }
    mp /= n; mr /= n;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int i = 0; i < n; i++) {
        const double x = pred[i] - mp, y = real[i] - mr;
        sxy += x * y; sxx += x * x; syy += y * y;
    }
    sc.tertiary = (sxx > 0.0 && syy > 0.0) ? sxy / std::sqrt(sxx * syy) : 0.0;
    if (!std::isfinite(sc.tertiary)) sc.tertiary = 0.0;
    return sc;
}

static inline float jnl_composite(const JNLScore& s) {
    return (float)(s.primary + JNL_W2 * s.secondary + JNL_W3 * 0.5 * (s.tertiary + 1.0));
}
