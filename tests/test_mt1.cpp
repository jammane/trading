// tests/test_mt1.cpp — Unit tests for MT1 pure-math logic in training_v4.cpp
//
// No BLAS, no file I/O — links against nothing but libc/libm.
//
// compute_mt1_scores and the MT1 decode helpers are NOT replicated here: they come from the shared
// mt1_scoring.h that training_v4.cpp also includes. The replica they replaced had silently drifted
// from production (different sc_acc formula, the retired 0.50/0.33/0.17 composite, 3 of 4
// components), so these tests were green while testing a function that no longer existed. Anything
// still copied below (PCG32, slot layout) is pure arithmetic with no production-drift risk.
//
// Build:  cmake --build build --target test_mt1_cpp
// Run:    ./build/test_mt1_cpp   (or: ctest --test-dir build)

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "mt1_scoring.h"

// ── Constants (must match training_v4.cpp exactly) ─────────────────────────

static constexpr int   ELITE_COUNT         = 17;
static constexpr int   WAVG_COUNT          = 3;
// Heads/tails layout (v0.4.0.0+). The legacy per-component pool had 25 parents (17 elites +
// 3 wavg + 5 injection) x 8; injection slots were removed when the shared head took over
// cross-component propagation, so parents = 20 and the freed capacity became mutations.
static constexpr int   HT_PARENTS          = ELITE_COUNT + WAVG_COUNT;   // 20
static constexpr int   MT1_COMP_SLOTS      = 200;
static constexpr int   HT_MUTS             = MT1_COMP_SLOTS - HT_PARENTS; // 180
// MT1_RANGE_CEIL_MULT / MT1_SCALE_DOLLARS / MT1_LOGIT_CAP come from mt1_scoring.h
static constexpr float MT1_FLOOR_COLD      = 250.f;
static constexpr int   MT1_ROLLING_DAYS    = 10;
static constexpr int   HIST_DAYS           = 5;
static constexpr int   HIST_PER_DAY        = 10;
static constexpr int   HIST_ELITE          = 7;
static constexpr int   HIST_WAVG           = 3;
static constexpr int   MT1_DIR_DAYS        = 10;

// ── Test harness ───────────────────────────────────────────────────────────

static int pass_count = 0;
static int fail_count = 0;
static const char* current_suite = "";

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "  FAIL [%s] %s:%d  %s\n", current_suite, __FILE__, __LINE__, #cond); \
            fail_count++; \
        } else { \
            pass_count++; \
        } \
    } while(0)

#define NEAR(a, b, eps)   (fabsf((float)(a) - (float)(b)) < (float)(eps))
#define SUITE(name)        do { current_suite = (name); printf("  %s\n", name); } while(0)

// ── PCG32 (exact copy from training_v4.cpp) ────────────────────────────────

struct PCG32 {
    uint64_t state{0}, inc{1};

    void seed(uint64_t s, uint64_t seq = 1) {
        state = 0; inc = (seq << 1) | 1;
        next(); state += s; next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xs  = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xs >> rot) | (xs << ((-rot) & 31u));
    }
    float next_float() { return (next() >> 8) * (1.0f / (1 << 24)); }
};


// ── Blend weight math (logic extracted from gen_mt1_blend in training_v4.cpp) ─

// Returns true and fills weights[8] / ranks[8] from the given PCG32 state.
// Mirrors the exact sampling and deduplication logic in gen_mt1_blend.
static void blend_draw(PCG32& rng, int* out_ranks, float* out_weights) {
    for (int p = 0; p < 4; p++) {
        int r1 = (int)(rng.next() % ELITE_COUNT);
        int r2 = (int)(rng.next() % (ELITE_COUNT - 1));
        if (r2 >= r1) r2++;
        out_ranks[p * 2]     = r1;
        out_ranks[p * 2 + 1] = r2;
    }
    float wsum = 0.f;
    for (int k = 0; k < 8; k++) {
        out_weights[k] = 20.f - (float)out_ranks[k];
        wsum += out_weights[k];
    }
    for (int k = 0; k < 8; k++) out_weights[k] /= wsum;
}

// Volatility-channel fixtures. Channel 2 predicts forward realized vol as of v0.5.0.0, so every
// compute_mt1_scores call needs a vol target and a vol floor. These are deliberately neutral:
// kVolA sits at exactly 2×kVolF, i.e. the naive "predict the rolling mean" answer, so the vol term
// contributes its baseline and cannot skew the direction/accuracy assertions.
static constexpr float kVolF = 100.f;   // vol_floor  (half the rolling mean)
static constexpr float kVolA = 200.f;   // vol_actual (= the rolling mean itself)

// ── Tests ──────────────────────────────────────────────────────────────────

static void test_constants()
{
    SUITE("constants: compile-time arithmetic");

    // Heads/tails pool: 20 parents (17 direct elites + 3 wavg) + 180 mutations = 200 slots.
    // Injection slots were removed when the shared head took over cross-component propagation.
    CHECK(HT_PARENTS     == ELITE_COUNT + WAVG_COUNT);
    CHECK(HT_PARENTS     == 20);
    CHECK(MT1_COMP_SLOTS == 200);
    CHECK(HT_MUTS        == 180);
    CHECK(HT_PARENTS + HT_MUTS == MT1_COMP_SLOTS);
    CHECK(HIST_DAYS == 5);
    CHECK(HIST_PER_DAY == 10);
    CHECK(HIST_ELITE + HIST_WAVG == HIST_PER_DAY);   // 7+3 == 10

    // Slot layout for the REPLAY pools (acc/rng/cfd/head): elites 0-16, wavg 17-19, muts 20-199
    CHECK(ELITE_COUNT               == 17);
    CHECK(ELITE_COUNT + WAVG_COUNT  == 20);  // first mutation slot

    // MT1_FLOOR_COLD / 2 is the cold-start acc_floor
    CHECK(NEAR(MT1_FLOOR_COLD / 2.f, 125.f, 0.001f));

    // Replay scoring window (acc/rng/cfd). Direction left this regime in v0.5.0.0.
    CHECK(MT1_DIR_DAYS == 10);
    CHECK(MT1_DIR_DAYS > 1);        // must be multi-day
}

static void test_pcg32()
{
    SUITE("PCG32: determinism and independence");

    // Same seed → identical sequence
    PCG32 a, b;
    a.seed(12345ULL);
    b.seed(12345ULL);
    for (int i = 0; i < 20; i++)
        CHECK(a.next() == b.next());

    // Different seeds → different first outputs
    PCG32 c, d;
    c.seed(12345ULL);
    d.seed(99999ULL);
    bool any_different = false;
    for (int i = 0; i < 10; i++)
        if (c.next() != d.next()) { any_different = true; break; }
    CHECK(any_different);

    // next_float() in [0, 1)
    PCG32 e;
    e.seed(7777ULL);
    for (int i = 0; i < 50; i++) {
        float f = e.next_float();
        CHECK(f >= 0.f && f < 1.f);
    }

    // Deterministic seed for blend: same (actual_day, ind_i, b) → same draw
    {
        int actual_day = 42, ind_i = 3, blend_b = 17;
        PCG32 r1, r2;
        r1.seed((uint64_t)actual_day * 654321ULL +
                (uint64_t)ind_i      * 31337ULL  +
                (uint64_t)blend_b    *  1009ULL);
        r2.seed((uint64_t)actual_day * 654321ULL +
                (uint64_t)ind_i      * 31337ULL  +
                (uint64_t)blend_b    *  1009ULL);
        CHECK(r1.next() == r2.next());
        CHECK(r1.next() == r2.next());
    }
}

// Saturation ceiling of the bounded decode: conf can approach but never reach 1.0.
// This is the whole point of MT1_LOGIT_CAP — a saturated sigmoid has zero derivative, so weight
// mutations stop changing the output and the pool freezes genetically.
static const float kConfMax = mt1_conf(1e6f);   // = sigmoid(MT1_LOGIT_CAP) ≈ 0.982

static void test_scores_direction()
{
    SUITE("compute_mt1_scores: direction component");

    float acc_floor = 125.f;

    // High conf, actual positive → sc_dir = conf ≈ kConfMax (bounded, never 1.0)
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s.direction, kConfMax, 0.001f));
    }
    // Low conf, actual positive → sc_dir = conf ≈ kConfMin (bounded, never 0)
    {
        float raw4[4] = {-10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s.direction, 1.f - kConfMax, 0.001f));
    }
    // Low conf, actual negative → sc_dir = 1 - conf ≈ kConfMax
    {
        float raw4[4] = {-10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(-100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s.direction, kConfMax, 0.001f));
    }
    // High conf, actual negative → sc_dir = 1 - conf ≈ 1 - kConfMax
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(-100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s.direction, 1.f - kConfMax, 0.001f));
    }
    // Neutral (conf=0.5) → sc_dir = 0.5 regardless of sign — random baseline
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s.direction, 0.5f, 0.001f));
        auto s2 = compute_mt1_scores(-100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s2.direction, 0.5f, 0.001f));
        auto s3 = compute_mt1_scores(0.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(s3.direction, 0.5f, 0.001f));
    }
    // Known exact value under the bounded decode: conf = sigmoid(CAP*tanh(raw/CAP)).
    // actual positive → sc_dir = conf; actual negative → sc_dir = 1 - conf (complements).
    {
        float raw4[4] = {logf(3.f), 0.f, 0.f, 0.f};
        float want = mt1_conf(logf(3.f));
        auto sp = compute_mt1_scores(100.f, kVolA, raw4, acc_floor, kVolF);
        auto sn = compute_mt1_scores(-100.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(NEAR(sp.direction, want, 0.001f));
        CHECK(NEAR(sn.direction, 1.f - want, 0.001f));
        CHECK(want > 0.5f && want < 0.75f);   // squash pulls it in from the raw sigmoid's 0.75
    }
    // sc_dir always in [0, 1]
    {
        float raw4[4] = {5.f, 0.f, 0.f, 0.f};
        auto sp = compute_mt1_scores(200.f, kVolA, raw4, acc_floor, kVolF);
        auto sn = compute_mt1_scores(-200.f, kVolA, raw4, acc_floor, kVolF);
        CHECK(sp.direction >= 0.f && sp.direction <= 1.f);
        CHECK(sn.direction >= 0.f && sn.direction <= 1.f);
        // symmetric: dir(positive) + dir(negative) == 1.0
        CHECK(NEAR(sp.direction + sn.direction, 1.f, 0.001f));
    }
}

static void test_scores_vol()
{
    SUITE("compute_mt1_scores: volatility component (channel 2)");

    // Channel 2 predicts forward realized volatility as of v0.5.0.0:
    //   vol_pred = softplus(raw4[2]) x MT1_SCALE_DOLLARS
    //   sc_vol   = den / (|vol_pred - vol_actual| + den),  den = max(vol_actual, vol_floor)
    // Same shape as sc_acc, so size and risk are graded on comparable footing.
    const float acc_floor = 125.f, vf = 100.f;

    // A perfect prediction scores exactly 1.
    {
        // Pick raw so vol_pred lands on the target: softplus(raw) = va / SCALE.
        float va  = 250.f;
        float x   = logf(expf(va / MT1_VOL_SCALE) - 1.f);
        float raw4[4] = {0.f, 0.f, x, 0.f};
        auto s = compute_mt1_scores(100.f, va, raw4, acc_floor, vf);
        CHECK(NEAR(s.range, 1.f, 0.005f));
    }

    // Predicting ~zero vol against a real target: err -> va, den = max(va, vf) = va, so sc -> 1/2.
    {
        float va = 400.f;
        float raw4[4] = {0.f, 0.f, -20.f, 0.f};        // softplus clamps to ~0
        auto s = compute_mt1_scores(100.f, va, raw4, acc_floor, vf);
        CHECK(NEAR(s.range, 0.5f, 0.01f));
    }

    // Monotone: a closer prediction must score higher.
    {
        float va = 300.f;
        float near_x = logf(expf(280.f / MT1_VOL_SCALE) - 1.f);
        float far_x  = logf(expf( 80.f / MT1_VOL_SCALE) - 1.f);
        float rn[4] = {0.f, 0.f, near_x, 0.f};
        float rf[4] = {0.f, 0.f, far_x,  0.f};
        auto sn = compute_mt1_scores(100.f, va, rn, acc_floor, vf);
        auto sf = compute_mt1_scores(100.f, va, rf, acc_floor, vf);
        CHECK(sn.range > sf.range);
    }

    // The floor protects a near-zero target: without den = max(va, vf) a tiny realized vol would
    // make any error look catastrophic and the channel would grade pure noise.
    {
        float raw4[4] = {0.f, 0.f, -20.f, 0.f};
        auto s_tiny = compute_mt1_scores(100.f, 1.f, raw4, acc_floor, vf);
        CHECK(s_tiny.range > 0.9f);                     // err ~1 against den = vf = 100
        CHECK(s_tiny.range <= 1.f);
    }

    // Score is bounded in (0,1] for any raw input.
    for (float x = -25.f; x <= 25.f; x += 5.f) {
        float raw4[4] = {0.f, 0.f, x, 0.f};
        auto s = compute_mt1_scores(100.f, 300.f, raw4, acc_floor, vf);
        CHECK(s.range > 0.f && s.range <= 1.f);
    }

    // Degenerate all-zero vol history: vol_actual and vol_floor both 0 would make den_vol 0 and
    // sc_vol = 0/0 = NaN, which then propagates through the composite into every selection for the
    // rest of the run. The guard must keep every channel finite.
    {
        float raw4[4] = {0.f, 0.f, -20.f, 0.f};
        auto s = compute_mt1_scores(100.f, 0.f, raw4, acc_floor, 0.f);
        CHECK(s.range == s.range);            // not NaN
        CHECK(s.composite == s.composite);
        CHECK(s.range >= 0.f && s.range <= 1.f);
        CHECK(s.composite >= 0.f && s.composite <= 1.f);
    }
}

static void test_scores_accuracy()
{
    SUITE("compute_mt1_scores: accuracy component");

    // sc_acc = denom / (err + denom) — smooth, always in (0, 1]. (The old replica in this file
    // had max(0, 1 - err/denom), which production has not used for some time.)

    // acc_floor active: |actual_d|=50 < acc_floor=200 → denom=200
    // delta_d=0, err=50 → sc_acc = 200/(50+200) = 0.8
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(50.f, kVolA, raw4, 200.f, kVolF);
        CHECK(NEAR(s.accuracy, 0.8f, 0.001f));
    }

    // |actual_d| > acc_floor: denom = |actual_d|
    // actual_d=500, acc_floor=100, delta_d=0, err=500 → sc_acc = 500/(500+500) = 0.5.
    // This is the predict-nothing null model: it scores exactly 0.5, never 0.
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(500.f, kVolA, raw4, 100.f, kVolF);
        CHECK(NEAR(s.accuracy, 0.5f, 0.001f));
    }

    // Perfect prediction (actual_d == delta_d): err=0, sc_acc=1.0
    // raw4[1]=atanh(0.01)≈0.01 → delta_d≈100, actual_d=100
    {
        float raw4[4] = {0.f, 0.01f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, kVolA, raw4, 10.f, kVolF);
        // err = |100 - tanh(0.01)*10000| = |100 - 99.9967...| ≈ 0.003
        CHECK(s.accuracy > 0.999f);
    }

    // sc_acc never negative (clamped at 0)
    // actual_d=10, delta_d=0, err=10, acc_floor=5, denom=10: sc_acc=0 (not negative)
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(10.f, kVolA, raw4, 5.f, kVolF);
        CHECK(s.accuracy >= 0.f);
    }
}

static void test_scores_composite()
{
    SUITE("compute_mt1_scores: composite formula");

    // composite = equal-weight mean of the THREE graded components' SECONDARY [0,1] normalizations
    // (direction, accuracy, volatility), each mapping its own naive baseline B -> 0 and ideal -> 1:
    //   sec = clamp((raw - B) / (1 - B), 0, 1)
    // conf4 was dropped from the composite in v0.5.0.0 along with the band it was graded against.
    float acc_floor = 125.f;

    float raw4_cases[][4] = {
        {10.f,  0.f, 0.f, 0.f},
        {-10.f, 0.f, 0.f, 0.f},
        {0.f,   0.f, 0.f, 0.f},
        {10.f,  0.f, 50.f, 0.f},
        {10.f,  0.f, 0.f, 5.f},
    };
    float actual_ds[] = {100.f, 100.f, 0.f, 5.f, 100.f};
    int n = 5;

    for (int i = 0; i < n; i++) {
        auto s = compute_mt1_scores(actual_ds[i], kVolA, raw4_cases[i], acc_floor, kVolF);
        CHECK(s.composite  >= 0.f && s.composite  <= 1.f);
        CHECK(s.direction  >= 0.f && s.direction  <= 1.f);
        CHECK(s.range      >= 0.f && s.range      <= 1.f);
        CHECK(s.accuracy   >= 0.f && s.accuracy   <= 1.f);
    }

    // Known numeric case. actual_d = 0 so the predict-0 null model is already perfect: b_acc = 1
    // and sec_acc = 0, i.e. accuracy earns nothing for matching a null it could not have lost.
    // kVolA == 2*kVolF is exactly the naive vol predictor, so sec_vol = 0 as well. Only direction
    // contributes, and the composite is now a THIRD of it rather than a quarter.
    {
        float raw4[4] = {10.f, 0.f, 1.f, 0.f};
        auto s = compute_mt1_scores(0.f, kVolA, raw4, 125.f, kVolF);
        CHECK(NEAR(s.direction, kConfMax, 0.001f));
        CHECK(s.accuracy == 1.f);
        CHECK(NEAR(s.composite, (kConfMax - 0.5f) / 0.5f / 3.f, 0.002f));
    }
}

static void test_scores_confidence()
{
    SUITE("compute_mt1_scores: conf4 is present but UNGRADED");

    // Channel 3 graded conf4 against the band geometry that channel 2 used to produce. With the
    // band retired there is nothing to grade it against, and its old optimum was degenerate anyway
    // (err > r always => ideal ~ 0 => conf4 = 0 wins; 89% of values fell below 0.01, pool spread
    // 0.05%). The tail stays in the model so MT1NN and the .bin/.pt layouts are unchanged, but it
    // contributes nothing to any score.
    const float acc_floor = 125.f, vf = 100.f;

    float lo[4] = {0.f, 0.f, 0.f, -10.f};
    float mid[4] = {0.f, 0.f, 0.f, 0.f};
    float hi[4] = {0.f, 0.f, 0.f, 10.f};
    auto s_lo  = compute_mt1_scores(100.f, 300.f, lo,  acc_floor, vf);
    auto s_mid = compute_mt1_scores(100.f, 300.f, mid, acc_floor, vf);
    auto s_hi  = compute_mt1_scores(100.f, 300.f, hi,  acc_floor, vf);

    // Reported as a constant 0 regardless of the tail's output...
    CHECK(s_lo.confidence  == 0.f);
    CHECK(s_mid.confidence == 0.f);
    CHECK(s_hi.confidence  == 0.f);
    // ...and it cannot move the composite either, which is what "ungraded" has to mean.
    CHECK(NEAR(s_lo.composite, s_hi.composite, 1e-6f));
    CHECK(NEAR(s_mid.composite, s_hi.composite, 1e-6f));
}

static void test_blend_weights()
{
    SUITE("blend weights: deduplication, normalization, rank ordering");

    // Run several draws and verify invariants
    for (int trial = 0; trial < 20; trial++) {
        PCG32 rng;
        rng.seed((uint64_t)trial * 9999ULL + 1ULL);

        int   ranks[8];
        float weights[8];
        blend_draw(rng, ranks, weights);

        // Each pair (p*2, p*2+1) must have distinct ranks
        for (int p = 0; p < 4; p++) {
            CHECK(ranks[p*2] != ranks[p*2+1]);
        }

        // All ranks in [0, ELITE_COUNT)
        for (int k = 0; k < 8; k++) {
            CHECK(ranks[k] >= 0 && ranks[k] < ELITE_COUNT);
        }

        // All weights > 0 (rank < 20 always since ELITE_COUNT=17 < 20)
        float wsum = 0.f;
        for (int k = 0; k < 8; k++) {
            CHECK(weights[k] > 0.f);
            wsum += weights[k];
        }

        // Weights sum to 1.0
        CHECK(NEAR(wsum, 1.0f, 1e-5f));
    }

    // Weight ordering: rank 0 gets weight 20/sum, rank 16 gets 4/sum
    // 20/sum > 4/sum always → rank-0 source outweighs rank-16 source
    CHECK(20.f > (20.f - (float)(ELITE_COUNT - 1)));  // 20 > 4

    // Determinism: same seed → same ranks and weights
    {
        PCG32 r1, r2;
        r1.seed(42ULL); r2.seed(42ULL);
        int ranks1[8], ranks2[8];
        float w1[8], w2[8];
        blend_draw(r1, ranks1, w1);
        blend_draw(r2, ranks2, w2);
        for (int k = 0; k < 8; k++) {
            CHECK(ranks1[k] == ranks2[k]);
            CHECK(NEAR(w1[k], w2[k], 1e-7f));
        }
    }

    // Different blend indices → different seeds → (almost certainly) different draws
    {
        int actual_day = 100, ind_i = 2;
        PCG32 r1, r2;
        r1.seed((uint64_t)actual_day * 654321ULL + (uint64_t)ind_i * 31337ULL + 0ULL  * 1009ULL);
        r2.seed((uint64_t)actual_day * 654321ULL + (uint64_t)ind_i * 31337ULL + 1ULL  * 1009ULL);
        bool any_diff = (r1.next() != r2.next());
        CHECK(any_diff);
    }
}

static void test_mutation_parent_assignment()
{
    SUITE("replay pool: weighted children table");

    // step_mt1_pool concentrates breeding on proven elites rather than assigning parents
    // round-robin. kChildren must sum to exactly HT_MUTS or the tail of the mutation range
    // silently falls back to idx % HT_PARENTS.
    static const int kChildren[20] = {
        16, 13, 13, 12, 12,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        6, 6, 6
    };
    int total = 0;
    for (int p = 0; p < HT_PARENTS; p++) total += kChildren[p];
    CHECK(total == HT_MUTS);
    CHECK(total == 180);

    // Expand the table the way step_mt1_pool does and check the boundaries.
    int parent_of_mut[180];
    int idx = 0;
    for (int p = 0; p < HT_PARENTS && idx < HT_MUTS; p++)
        for (int c = 0; c < kChildren[p] && idx < HT_MUTS; c++)
            parent_of_mut[idx++] = p;
    CHECK(idx == HT_MUTS);
    CHECK(parent_of_mut[0]  == 0);    // first 16 children belong to slot 0
    CHECK(parent_of_mut[15] == 0);
    CHECK(parent_of_mut[16] == 1);    // slot 1 starts at child 16
    CHECK(parent_of_mut[HT_MUTS - 1] == 19);   // last child belongs to the last wavg blend

    // Slot 0 gets 16 of 180 children — 8.9% of every generation from one model. That
    // concentration is why the direction pool moved to flat round-robin backfill.
    CHECK(kChildren[0] == 16);
    CHECK(kChildren[0] > kChildren[19] * 2);
}

static void test_elite_slot_layout()
{
    SUITE("replay pool: elite slot layout (17 direct + 3 wavg, no injection)");

    // Direct elites: slots 0..ELITE_COUNT-1 = 0..16
    CHECK(ELITE_COUNT == 17);
    // Wavg blend slots: ELITE_COUNT + 0..WAVG_COUNT-1 = 17, 18, 19
    for (int b = 0; b < WAVG_COUNT; b++) {
        int slot = ELITE_COUNT + b;
        CHECK(slot >= 17 && slot <= 19);
    }
    // Parents end at 19; mutations start at 20 and run to 199. There are no injection slots —
    // the shared head propagates cross-component learning instead.
    CHECK(HT_PARENTS == 20);
    CHECK(MT1_COMP_SLOTS - HT_PARENTS == 180);
}

static void test_rolling_buffers()
{
    SUITE("rolling buffers: acc_floor and vol_floor computation");

    // acc_floor = mean(rolling_actual) / 2
    {
        float buf[MT1_ROLLING_DAYS] = {400.f, 600.f, 200.f};
        int count = 3;
        float sum = 0.f;
        for (int k = 0; k < count; k++) sum += buf[k];
        float floor = sum / (float)count / 2.f;
        // (400+600+200)/3 = 400, /2 = 200
        CHECK(NEAR(floor, 200.f, 0.01f));
    }

    // Cold-start: rolling_count=0 → use MT1_FLOOR_COLD/2 = 125
    CHECK(NEAR(MT1_FLOOR_COLD / 2.f, 125.f, 0.01f));

    // vol_floor = mean(rolling_vol) / 2 — same shape as acc_floor, because the vol channel is
    // graded with the same denominator form. 2 x vol_floor is the naive "predict the mean" answer,
    // which is what sec_vol normalizes against.
    {
        float buf[MT1_ROLLING_DAYS] = {100.f, 200.f, 300.f, 400.f};
        int count = 4;
        float sum = 0.f;
        for (int k = 0; k < count; k++) sum += buf[k];
        float vfloor = sum / (float)count / 2.f;
        // (100+200+300+400)/4 = 250, /2 = 125
        CHECK(NEAR(vfloor, 125.f, 0.01f));
        CHECK(NEAR(2.f * vfloor, 250.f, 0.01f));   // naive predictor == the rolling mean
    }

    // Rolling circular buffer wrap-around: head advances modulo MT1_ROLLING_DAYS
    {
        float buf[MT1_ROLLING_DAYS] = {};
        int head = 0, count = 0;
        // Fill 12 values; only last 10 should remain
        for (int i = 0; i < 12; i++) {
            buf[head] = (float)(i + 1);
            head = (head + 1) % MT1_ROLLING_DAYS;
            if (count < MT1_ROLLING_DAYS) count++;
        }
        CHECK(count == MT1_ROLLING_DAYS);
        // Sum of last 10: 3+4+5+6+7+8+9+10+11+12 = (3..12) = 75
        float sum = 0.f;
        for (int k = 0; k < MT1_ROLLING_DAYS; k++) sum += buf[k];
        CHECK(NEAR(sum, 75.f, 0.01f));
    }
}

static void test_hist_buffer_addressing()
{
    SUITE("history circular buffer: oldest-slot addressing");

    const int total = HIST_DAYS * HIST_PER_DAY;  // 50
    CHECK(total == 50);

    // Helper: compute oldest absolute position
    auto oldest_of = [&](int head, int count) -> int {
        int n_hist = count * HIST_PER_DAY;
        return (head * HIST_PER_DAY - n_hist + total) % total;
    };

    // head=0, count=5 (full, just completed wrap): oldest at position 0
    // (0*10 - 50 + 50) % 50 = 0
    CHECK(oldest_of(0, 5) == 0);

    // head=3, count=5 (full, mid-buffer): oldest at position 30
    // (3*10 - 50 + 50) % 50 = 30
    CHECK(oldest_of(3, 5) == 30);

    // head=1, count=3 (partial, 30 models stored): oldest at position 30
    // (1*10 - 30 + 50) % 50 = 30
    CHECK(oldest_of(1, 3) == 30);

    // head=0, count=1 (first day just written): oldest at position 40
    // (0*10 - 10 + 50) % 50 = 40
    CHECK(oldest_of(0, 1) == 40);

    // head=4, count=4: oldest at position 0
    // (4*10 - 40 + 50) % 50 = 10 % 50... wait: 40 - 40 + 50 = 50 % 50 = 0
    CHECK(oldest_of(4, 4) == 0);

    // Iteration: verify the circular walk covers exactly n_hist models
    {
        int head = 2, count = 3;
        int n_hist = count * HIST_PER_DAY;
        int oldest  = oldest_of(head, count);
        int visited = 0;
        for (int k = 0; k < n_hist; k++) {
            int abs_pos = (oldest + k) % total;
            (void)abs_pos;
            visited++;
        }
        CHECK(visited == n_hist);
    }
}

static void test_pool_score_dispatch()
{
    SUITE("component pool: score index dispatch (pool_id → component)");

    // pool_id 0=dir, 1=acc, 2=rng, 3=cfd
    // In step_mt1_component: switch(pool_id) { 0→direction, 1→accuracy, 2→range, 3→confidence }
    // Verify with a score where all components differ
    float raw4[4] = {10.f, 0.f, 50.f, 0.f};
    // actual_d=5: sc_dir=conf≈1 (actual positive), sc_rng≈0.1, sc_acc=0, sc_cfd varies
    auto s = compute_mt1_scores(5.f, kVolA, raw4, 1.f, kVolF);

    // Scores are distinguishable
    CHECK(s.direction != s.range);    // 1.0 != 0.1
    CHECK(s.direction != s.accuracy); // 1.0 != 0.0
    CHECK(s.range     != s.accuracy); // 0.1 != 0.0

    // Pool 0 (direction) selects s.direction
    // Pool 1 (accuracy) selects s.accuracy
    // Pool 2 (range) selects s.range
    // Pool 3 (confidence) selects s.confidence
    float pool_scores[4] = {s.direction, s.accuracy, s.range, s.confidence};
    // Verify each is in [0,1]
    for (int p = 0; p < 4; p++)
        CHECK(pool_scores[p] >= 0.f && pool_scores[p] <= 1.f);
}

static void test_blend_slots_count()
{
    SUITE("replay pool: candidate count per step");

    CHECK(MT1_COMP_SLOTS == 200);
    // History: 5 days x 10 models = 50 max candidates from history
    CHECK(HIST_DAYS * HIST_PER_DAY == 50);
    // Max candidates per replay step: 200 slots + 50 history = 250
    CHECK(MT1_COMP_SLOTS + HIST_DAYS * HIST_PER_DAY == 250);
    // Top 10 pushed to history each step (7 direct elites + 3 wavg blends)
    CHECK(HIST_PER_DAY == 10);
    CHECK(HIST_ELITE == 7);
    CHECK(HIST_WAVG == 3);
}

// ── Drift study ±1 metric (replica of drift_rel_metric in training_v4.cpp) ──

static inline float drift_rel_metric(float S, float mean, float mx, float mn, float floor) {
    float spread = (S >= mean) ? (mx - mean) : (mean - mn);
    spread = fmaxf(spread, floor);
    return tanhf((S - mean) / spread);
}

static void test_drift_rel_metric()
{
    SUITE("drift_rel_metric");
    // Track == band mean → exactly 0.
    CHECK(NEAR(drift_rel_metric(5.0f, 5.0f, 8.0f, 2.0f, 1e-3f), 0.0f, 1e-6f));
    // Track at band best (one spread above mean) → tanh(1) ≈ 0.7616.
    CHECK(NEAR(drift_rel_metric(8.0f, 5.0f, 8.0f, 2.0f, 1e-3f), tanhf(1.0f), 1e-5f));
    // Track at band worst (one spread below mean) → -tanh(1).
    CHECK(NEAR(drift_rel_metric(2.0f, 5.0f, 8.0f, 2.0f, 1e-3f), -tanhf(1.0f), 1e-5f));
    // Decisive (3 spreads above) → tanh(3) ≈ 0.995, in (0.99, 1.0).
    float decisive = drift_rel_metric(14.0f, 5.0f, 8.0f, 2.0f, 1e-3f);  // (14-5)/3 = 3
    CHECK(decisive > 0.99f && decisive <= 1.0f);
    CHECK(NEAR(decisive, tanhf(3.0f), 1e-5f));
    // Bounded in [-1, 1] always (float tanh saturates to ±1 at large args — never beyond).
    CHECK(drift_rel_metric(-1000.f, 5.f, 8.f, 2.f, 1e-3f) >= -1.0f);
    CHECK(drift_rel_metric( 1000.f, 5.f, 8.f, 2.f, 1e-3f) <=  1.0f);
    // Tight band (mx≈mean) must not blow up — floor caps the swing.
    float tight = drift_rel_metric(5.001f, 5.0f, 5.0f, 5.0f, 1.0f);
    CHECK(tight > 0.0f && tight < 0.01f);
    // Asymmetric spreads: below-mean uses the down spread (5-0=5), above uses the up (6-5=1).
    CHECK(NEAR(drift_rel_metric(0.0f, 5.0f, 6.0f, 0.0f, 1e-3f), -tanhf(1.0f), 1e-5f)); // (0-5)/5 = -1
    CHECK(NEAR(drift_rel_metric(6.0f, 5.0f, 6.0f, 0.0f, 1e-3f),  tanhf(1.0f), 1e-5f)); // (6-5)/1 = +1
}

// ── Direction class-balanced weighting (step_mt1_component, absolute-target retune) ──
// Replicates the day_weight balancing: normalize so up/down days each carry half the window
// weight. Key property: a constant-prediction model scores the no-skill baseline dir_W/2
// regardless of the target's sign skew (so base-rate betting earns no edge).
static float balanced_dir_total(const int* up, const float* w, const float* day_score, int n) {
    float w_up = 0.f, w_down = 0.f, W = 0.f;
    for (int i = 0; i < n; i++) { W += w[i]; if (up[i]) w_up += w[i]; else w_down += w[i]; }
    bool bal = (w_up > 0.f) && (w_down > 0.f);
    float total = 0.f;
    for (int i = 0; i < n; i++) {
        float dw = bal ? w[i] * (up[i] ? W/(2.f*w_up) : W/(2.f*w_down)) : w[i];
        total += day_score[i] * dw;
    }
    return total;
}

static void test_direction_balanced_weighting()
{
    SUITE("retired: class-balanced direction weighting");

    // Kept as a record of WHY the direction pool was redesigned, not as a test of live code.
    // Under class-balanced day weights every constant predictor scored exactly dir_W/2 regardless
    // of its confidence — a flat plateau of enormous volume in weight space, which is where the
    // pool sat. The arithmetic below reproduces that plateau; the code that implemented it was
    // deleted in v0.5.0.0 when direction moved to forward accumulation (step_mt1_dir_pool).
    const int dc = MT1_DIR_DAYS;
    float w[MT1_DIR_DAYS], dir_W = 0.f, w_up = 0.f, w_down = 0.f;
    bool up[MT1_DIR_DAYS];
    for (int di = 0; di < dc; di++) {
        int age = dc - 1 - di;
        w[di] = 2.f - (float)age / (float)(MT1_DIR_DAYS - 1);
        up[di] = (di % 3 == 0);                    // a lopsided 4/6 split
        dir_W += w[di];
        if (up[di]) w_up += w[di]; else w_down += w[di];
    }
    // Any constant confidence scores exactly dir_W/2 — the no-skill baseline, for every c.
    for (float c = 0.f; c <= 1.0001f; c += 0.25f) {
        float total = 0.f;
        for (int di = 0; di < dc; di++) {
            float dw = w[di] * (up[di] ? dir_W / (2.f * w_up) : dir_W / (2.f * w_down));
            total += dw * (up[di] ? c : (1.f - c));
        }
        CHECK(NEAR(total, dir_W / 2.f, 0.01f));
    }
    // Which is the defect: confidence 0.0 and 1.0 are indistinguishable to selection.
    CHECK(NEAR(dir_W / 2.f, 7.5f, 0.01f));
}

// ── Main ──────────────────────────────────────────────────────────────────


// ── Direction pool: forward accumulation (v0.5.0.0) ────────────────────────────────
// These grade mt1_dir_record_scores / mt1_dir_better straight out of mt1_scoring.h, so the test
// and the trainer cannot drift apart. The lifecycle arithmetic below mirrors step_mt1_dir_pool.

static void test_dir_record_scoring()
{
    SUITE("direction pool: 16-bit record scoring");

    float p, s, t;

    // Empty record scores zero on every key rather than dividing by zero.
    mt1_dir_record_scores(0x0000, 0, &p, &s, &t);
    CHECK(NEAR(p, 0.f, 1e-6f) && NEAR(s, 0.f, 1e-6f) && NEAR(t, 0.f, 1e-6f));

    // All 16 correct: primary and secondary both 1.0, so the blend is 1.0 too.
    mt1_dir_record_scores(0xFFFF, 16, &p, &s, &t);
    CHECK(NEAR(p, 1.f, 1e-6f));
    CHECK(NEAR(s, 1.f, 1e-6f));
    CHECK(NEAR(t, 1.f, 1e-6f));

    // All 16 wrong.
    mt1_dir_record_scores(0x0000, 16, &p, &s, &t);
    CHECK(NEAR(p, 0.f, 1e-6f) && NEAR(s, 0.f, 1e-6f) && NEAR(t, 0.f, 1e-6f));

    // Half right, but WHICH half decides the secondary. Newest 8 correct (bits 0-7):
    // weighted = 4*1.0 + 4*0.8 = 7.2 of 11.2 = 0.642857
    mt1_dir_record_scores(0x00FF, 16, &p, &s, &t);
    CHECK(NEAR(p, 0.5f, 1e-6f));
    CHECK(NEAR(s, 7.2f / 11.2f, 1e-5f));
    CHECK(s > p);                                   // recency rewards fresh correctness

    // Oldest 8 correct (bits 8-15): weighted = 4*0.6 + 4*0.4 = 4.0 of 11.2
    mt1_dir_record_scores(0xFF00, 16, &p, &s, &t);
    CHECK(NEAR(p, 0.5f, 1e-6f));
    CHECK(NEAR(s, 4.0f / 11.2f, 1e-5f));
    CHECK(s < p);                                   // and punishes stale correctness

    // Same primary, opposite secondary -> the ordering is decided by recency.
    DirSlotMeta fresh{0x00FF, 16, 0, 0}, stale{0xFF00, 16, 0, 0};
    CHECK(mt1_dir_better(fresh, stale));
    CHECK(!mt1_dir_better(stale, fresh));

    // Full-record weight sum is exactly 11.2.
    float wsum = 0.f;
    for (int i = 0; i < MT1_DIR_HIST_BITS; i++) wsum += MT1_DIR_RECENCY_W[i / 4];
    CHECK(NEAR(wsum, 11.2f, 1e-5f));
}

static void test_dir_partial_record()
{
    SUITE("direction pool: partial records normalize by actual count");

    float p, s, t;

    // 11 predictions, all correct. Both scores must read 1.0 — normalizing by a full 16 slots or
    // a full 11.2 of weight would score a perfect young model below a perfect old one.
    mt1_dir_record_scores(0x07FF, 11, &p, &s, &t);
    CHECK(NEAR(p, 1.f, 1e-6f));
    CHECK(NEAR(s, 1.f, 1e-6f));

    // 11 predictions, newest 4 correct: primary 4/11; weighted 4*1.0 of (4*1.0 + 4*0.8 + 3*0.6)
    mt1_dir_record_scores(0x000F, 11, &p, &s, &t);
    CHECK(NEAR(p, 4.f / 11.f, 1e-5f));
    CHECK(NEAR(s, 4.0f / (4.0f + 3.2f + 1.8f), 1e-5f));

    // Bits beyond n_pred are ignored, so stale garbage above the fill mark cannot inflate a score.
    mt1_dir_record_scores(0xFF0F, 4, &p, &s, &t);
    CHECK(NEAR(p, 1.f, 1e-6f));      // only the newest 4 bits count, and all 4 are set

    // A model at exactly the minimum age is scoreable.
    mt1_dir_record_scores(0x00FF, MT1_DIR_MIN_AGE, &p, &s, &t);
    CHECK(NEAR(p, 1.f, 1e-6f));      // 8 predictions, all 8 correct
}

static void test_dir_tertiary_is_inert_as_third_key()
{
    SUITE("direction pool: tertiary blend can never break a (primary, secondary) tie");

    // tertiary = 0.4*primary + 0.6*secondary is a function of the other two, so whenever both are
    // equal it is equal as well. It is computed and logged, but as a third sort key it never fires.
    float p1, s1, t1, p2, s2, t2;
    mt1_dir_record_scores(0x0F0F, 16, &p1, &s1, &t1);
    mt1_dir_record_scores(0x0F0F, 16, &p2, &s2, &t2);
    CHECK(NEAR(p1, p2, 1e-9f) && NEAR(s1, s2, 1e-9f) && NEAR(t1, t2, 1e-9f));

    // And it is the stated 60/40 blend.
    CHECK(NEAR(t1, 0.4f * p1 + 0.6f * s1, 1e-6f));
}

static void test_dir_lifecycle_arithmetic()
{
    SUITE("direction pool: cull rate, mature share, lineage cap");

    // Steady state: mature/N = 1 / (1 + MIN_AGE * CULL_PCT). The chosen constants target ~60%.
    float mature_frac = 1.f / (1.f + (float)MT1_DIR_MIN_AGE * MT1_DIR_CULL_PCT);
    CHECK(mature_frac > 0.58f && mature_frac < 0.62f);

    float mature = mature_frac * (float)MT1_COMP_SLOTS;
    CHECK(mature > 115.f && mature < 125.f);          // ~120 of 200

    float births = mature * MT1_DIR_CULL_PCT;
    CHECK(births > 8.f && births < 12.f);             // ~10 per day

    // Expected lifespan = min age + 1/cull rate, in predictions.
    float lifespan = (float)MT1_DIR_MIN_AGE + 1.f / MT1_DIR_CULL_PCT;
    CHECK(lifespan > 18.f && lifespan < 22.f);        // ~20

    // Parents = top 10% of mature, plus the 3 ephemeral wavg blends.
    int want = (int)(mature * MT1_DIR_ELITE_PCT + 0.5f);
    CHECK(want == 12);
    CHECK(want + 3 == 15);

    // Bound on MT1Scratch::dir_barred. Shares are strictly greater-than, so at most
    // ceil(1/CAP)-1 lineages can exceed the cap at once. Under hysteresis a barred lineage stays
    // barred all the way down to RESUME, so the real bound is driven by RESUME, not CAP — that is
    // the one that must fit the array (16 slots).
    int max_over_cap    = (int)ceilf(1.f / MT1_DIR_LINEAGE_CAP) - 1;
    int max_barred      = (int)ceilf(1.f / MT1_DIR_LINEAGE_RESUME) - 1;
    CHECK(max_over_cap == 7);
    CHECK(max_barred == 9);
    CHECK(max_barred >= max_over_cap);   // hysteresis can only widen the barred set
    CHECK(max_barred <= 16);             // sizeof(dir_barred)

    // Hysteresis: release must sit strictly below the bar, or a lineage parked near the cap
    // flips state every rep and the bar suppresses nothing. The first --dir-reps 20 run logged
    // 1850 barrings against 1838 re-enables over 35 days with the thresholds equal.
    CHECK(MT1_DIR_LINEAGE_RESUME < MT1_DIR_LINEAGE_CAP);
    CHECK(MT1_DIR_LINEAGE_CAP - MT1_DIR_LINEAGE_RESUME > 0.02f);
}

static void test_dir_record_shift()
{
    SUITE("direction pool: record shifts newest-first and saturates age");

    // step_mt1_dir_pool does: hist = (hist << 1) | correct, so bit 0 is always the newest call.
    uint16_t h = 0;
    h = (uint16_t)((h << 1) | 1u);      // correct
    h = (uint16_t)((h << 1) | 0u);      // wrong
    h = (uint16_t)((h << 1) | 1u);      // correct
    CHECK(((h >> 0) & 1u) == 1u);       // newest
    CHECK(((h >> 1) & 1u) == 0u);
    CHECK(((h >> 2) & 1u) == 1u);       // oldest of the three

    // 17 correct predictions leave a full register: the 17th pushed the 1st out.
    h = 0;
    for (int i = 0; i < 17; i++) h = (uint16_t)((h << 1) | 1u);
    CHECK(h == 0xFFFF);

    float p, s, t;
    mt1_dir_record_scores(h, 17, &p, &s, &t);
    CHECK(NEAR(p, 1.f, 1e-6f));         // n clamps to 16, so a longer life does not dilute

    // Age saturates rather than wrapping to 0 and making an old model look newborn.
    uint16_t n = 0xFFFF;
    if (n < 0xFFFFu) n++;
    CHECK(n == 0xFFFF);
}

int main()
{
    printf("MT1 C++ unit tests\n");
    printf("==================\n");

    test_constants();
    test_pcg32();
    test_scores_direction();
    test_scores_vol();
    test_scores_accuracy();
    test_scores_composite();
    test_scores_confidence();
    test_blend_weights();
    test_mutation_parent_assignment();
    test_elite_slot_layout();
    test_rolling_buffers();
    test_hist_buffer_addressing();
    test_pool_score_dispatch();
    test_blend_slots_count();
    test_drift_rel_metric();
    test_direction_balanced_weighting();
    test_dir_record_scoring();
    test_dir_partial_record();
    test_dir_tertiary_is_inert_as_third_key();
    test_dir_lifecycle_arithmetic();
    test_dir_record_shift();

    printf("\n==================\n");
    printf("%d passed, %d failed\n", pass_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
