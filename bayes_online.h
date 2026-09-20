#pragma once
// Online conditional ("Bayesian network") models, trained INSIDE the training pass.
//
// The offline versions (living_bn.py, dyn_tree.py) read a logged dataset after a run finishes.
// These update as each day is scored, from the scores already in hand, so a pass produces trained
// models rather than a pile of data to analyse later.
//
// STATE: the recent days of an industry's deployed book P&L, ordered by profitability with the
// profitability inversion point -- identical to the encoding raced offline.
//     k = 1   2 states   sign(today)
//     k = 2   6 states   up_more / up_less / up_down / dn_more / dn_less / dn_up
// Only k = 1 and 2 are carried: depth_sweep.py measured k = 3 at break-even and k = 4-5 at 8-50
// millinats WORSE than a coin flip, because the state space outruns the data.
//
// POOLED across industries, deliberately. The between-industry variance of this effect measured
// SMALLER than sampling noise alone (DerSimonian-Laird tau^2 = 0 with a 99% Cochran's Q gate), so
// twelve separate tables would be fitting noise.
//
// OUTCOMES: every scored elite, not only the deployed slot. Worth knowing what that buys: all 200
// slots reset to slot 0's portfolio each morning, so they share the market move IDENTICALLY and
// differ only in trade delta. Measured on the v0.8.1.7 run, market move is 516k of the 545k total
// P&L variance, so averaging 20 elites is worth about 1.11x in effective sample, not 20x. The
// elites make the outcome estimate slightly less noisy; they do not multiply the data.
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

static constexpr int BO_STATES = 6;         // k=2; k=1 occupies the first 2
static constexpr int BO_UP_MORE = 0, BO_UP_LESS = 1, BO_UP_DOWN = 2;
static constexpr int BO_DN_MORE = 3, BO_DN_LESS = 4, BO_DN_UP = 5;

// State from two consecutive days. Returns -1 when either day is flat (a hard-floor reset writes
// exactly $0, which is not a real direction) or non-finite.
static inline int bo_state2(float prev, float cur) {
    if (prev == 0.f || cur == 0.f || !std::isfinite(prev) || !std::isfinite(cur)) return -1;
    if (prev > 0.f) {
        if (cur > 0.f) return (cur >= prev) ? BO_UP_MORE : BO_UP_LESS;
        return BO_UP_DOWN;
    }
    if (cur < 0.f) return (cur <= prev) ? BO_DN_MORE : BO_DN_LESS;
    return BO_DN_UP;
}

static inline int bo_state1(float cur) {
    if (cur == 0.f || !std::isfinite(cur)) return -1;
    return cur > 0.f ? 0 : 1;
}

// One day's contribution: counts per state, so a rectangular window can subtract a whole day.
struct BODay {
    std::array<double, BO_STATES> n{}, same{}, sum{};
};

struct BayesOnline {
    int   k        = 2;       // 1 or 2
    float half_life = 504.f;  // used when window == 0
    int   window    = 0;      // days retained; 0 selects exponential decay
    const char* name = "";

    std::array<double, BO_STATES> n{}, same{}, sum{};
    std::deque<BODay> ring;

    int n_states() const { return k == 1 ? 2 : BO_STATES; }

    // P(tomorrow keeps today's sign | state), with a weak Beta(1,1) prior so an unseen state
    // reads 0.5 rather than dividing by zero.
    double p_continue(int s) const {
        if (s < 0 || s >= n_states()) return 0.5;
        return (same[s] + 1.0) / (n[s] + 2.0);
    }
    double expected_dollars(int s) const {
        if (s < 0 || s >= n_states() || n[s] < 1.0) return 0.0;
        return sum[s] / n[s];
    }
    // N: expected further length of the current run, derived from the posterior rather than
    // assumed. Reported, not used for prediction -- measured across every state it sits within
    // 0.25 days of the 2.0 a coin flip gives.
    double streak_n(int s) const {
        double p = p_continue(s);
        return (p >= 0.999) ? 999.0 : 1.0 / (1.0 - p);
    }

    void commit(const BODay& d) {
        if (window <= 0) {
            const double g = std::pow(0.5, 1.0 / (double)std::max(half_life, 1.f));
            for (int s = 0; s < BO_STATES; s++) { n[s] *= g; same[s] *= g; sum[s] *= g; }
        }
        for (int s = 0; s < BO_STATES; s++) { n[s] += d.n[s]; same[s] += d.same[s]; sum[s] += d.sum[s]; }
        if (window > 0) {
            ring.push_back(d);
            while ((int)ring.size() > window) {
                const BODay& o = ring.front();
                for (int s = 0; s < BO_STATES; s++) { n[s] -= o.n[s]; same[s] -= o.same[s]; sum[s] -= o.sum[s]; }
                ring.pop_front();
            }
        }
    }
};

// The family raced: 5 memory kernels x 2 depths. Kernel choice is a measurement, not a
// convention -- on the v0.8.1.7 run a 500-day half-life won on predictive log-likelihood and a
// 252-day window came fifth, because the conditionals are stationary and a window discards data.
static inline std::vector<BayesOnline> bo_make_family() {
    struct Cfg { const char* lab; float hl; int win; };
    const Cfg cfgs[5] = {{"1y-decay", 252.f, 0}, {"2y-decay", 504.f, 0},
                         {"1y-win", 1e9f, 252},  {"2y-win", 1e9f, 504},
                         {"forever", 1e9f, 0}};
    static const char* names[10] = {
        "k1-1y-decay", "k1-2y-decay", "k1-1y-win", "k1-2y-win", "k1-forever",
        "k2-1y-decay", "k2-2y-decay", "k2-1y-win", "k2-2y-win", "k2-forever"};
    std::vector<BayesOnline> v;
    for (int k = 1; k <= 2; k++)
        for (int c = 0; c < 5; c++) {
            BayesOnline m;
            m.k = k; m.half_life = cfgs[c].hl; m.window = cfgs[c].win;
            m.name = names[(k - 1) * 5 + c];
            v.push_back(m);
        }
    return v;
}
