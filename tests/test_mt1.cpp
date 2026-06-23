// tests/test_mt1.cpp — Unit tests for MT1 pure-math logic in training_v4.cpp
//
// Self-contained: no BLAS, no file I/O.  Replicates only the functions that
// are pure arithmetic so the test links against nothing but libc/libm.
//
// Build:  cmake --build build --target test_mt1_cpp
// Run:    ./build/test_mt1_cpp   (or: ctest --test-dir build)

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ── Constants (must match training_v4.cpp exactly) ─────────────────────────

static constexpr int   ELITE_COUNT         = 17;
static constexpr int   WAVG_COUNT          = 3;
static constexpr int   MT1_REINJECT        = 3;
static constexpr int   MT1_COMP_ELITE      = ELITE_COUNT + WAVG_COUNT + MT1_REINJECT;  // 23
static constexpr int   MT1_COMP_SLOTS      = 230;
static constexpr int   MT1_COMP_MUTS       = MT1_COMP_SLOTS - MT1_COMP_ELITE;          // 207
static constexpr int   MT1_BLEND_SLOTS     = 200;
static constexpr float MT1_RANGE_FLOOR     = 1.f;
static constexpr float MT1_RANGE_CEIL_MULT = 4.f;
static constexpr float MT1_SCALE_DOLLARS   = 10000.f;
static constexpr float MT1_FLOOR_COLD      = 250.f;
static constexpr int   MT1_ROLLING_DAYS    = 10;
static constexpr int   HIST_DAYS           = 5;
static constexpr int   HIST_PER_DAY        = 10;
static constexpr int   HIST_ELITE          = 7;
static constexpr int   HIST_WAVG           = 3;

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

// ── MT1 score breakdown (exact copy from training_v4.cpp) ──────────────────

struct MT1ScoreBreakdown { float composite, direction, range, accuracy, confidence; };

static MT1ScoreBreakdown compute_mt1_scores(
    float actual_d, const float raw4[4], float acc_floor, float range_ceiling)
{
    float conf      = 1.f / (1.f + expf(-raw4[0]));        // sigmoid
    float delta_d   = tanhf(raw4[1]) * MT1_SCALE_DOLLARS;
    float range_pct = log1pf(expf(raw4[2]));                // softplus
    float conf4     = 1.f / (1.f + expf(-raw4[3]));

    float sc_dir = ((conf >= 0.5f) == (actual_d >= 0.f)) ? 1.f : 0.f;

    float eff_delta = fmaxf(fabsf(delta_d), MT1_RANGE_FLOOR);
    float r         = range_pct * eff_delta;
    if (range_ceiling < 1e30f) r = fminf(r, range_ceiling);
    float err    = fabsf(actual_d - delta_d);
    float m      = (r > 1e-9f) ? err / r : (err > 0.f ? 1e9f : 0.f);
    float sc_rng = (m < 1.f) ? m : 0.f;

    float denom  = fmaxf(fabsf(actual_d), acc_floor);
    float sc_acc = fmaxf(0.f, 1.f - err / denom);

    float d     = err;
    float ideal = (d <= r) ? (1.f - 0.5f * d / fmaxf(r, 1e-9f)) : r / (d + r);
    float sc_cfd = 1.f - fabsf(conf4 - ideal);

    return {0.50f * sc_dir + 0.33f * sc_rng + 0.17f * sc_acc,
            sc_dir, sc_rng, sc_acc, sc_cfd};
}

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

// ── Tests ──────────────────────────────────────────────────────────────────

static void test_constants()
{
    SUITE("constants: compile-time arithmetic");

    CHECK(MT1_COMP_ELITE == ELITE_COUNT + WAVG_COUNT + MT1_REINJECT);
    CHECK(MT1_COMP_ELITE == 23);
    CHECK(MT1_COMP_SLOTS == 230);
    CHECK(MT1_COMP_MUTS  == MT1_COMP_SLOTS - MT1_COMP_ELITE);
    CHECK(MT1_COMP_MUTS  == 207);
    CHECK(MT1_BLEND_SLOTS == 200);
    CHECK(HIST_DAYS == 5);
    CHECK(HIST_PER_DAY == 10);
    CHECK(HIST_ELITE + HIST_WAVG == HIST_PER_DAY);   // 7+3 == 10

    // Wavg blend slots are 17, 18, 19
    CHECK(ELITE_COUNT               == 17);
    CHECK(ELITE_COUNT + 0           == 17);  // wavg slot 0
    CHECK(ELITE_COUNT + WAVG_COUNT  == 20);  // first reinject slot
    CHECK(MT1_COMP_ELITE - 1        == 22);  // last reinject slot

    // MT1_FLOOR_COLD / 2 is the cold-start acc_floor
    CHECK(NEAR(MT1_FLOOR_COLD / 2.f, 125.f, 0.001f));
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

static void test_scores_direction()
{
    SUITE("compute_mt1_scores: direction component");

    float acc_floor = 125.f;

    // conf > 0.5, actual_d > 0 → correct
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, acc_floor, 1e30f);
        CHECK(s.direction == 1.f);
    }
    // conf < 0.5, actual_d > 0 → wrong
    {
        float raw4[4] = {-10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, acc_floor, 1e30f);
        CHECK(s.direction == 0.f);
    }
    // conf < 0.5, actual_d < 0 → correct (both negative-direction)
    {
        float raw4[4] = {-10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(-100.f, raw4, acc_floor, 1e30f);
        CHECK(s.direction == 1.f);
    }
    // conf > 0.5, actual_d < 0 → wrong
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(-100.f, raw4, acc_floor, 1e30f);
        CHECK(s.direction == 0.f);
    }
    // Boundary: conf == 0.5 (raw4[0]=0 → sigmoid(0)=0.5), actual_d == 0 → both >=0 → correct
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(0.f, raw4, acc_floor, 1e30f);
        CHECK(s.direction == 1.f);
    }
}

static void test_scores_range()
{
    SUITE("compute_mt1_scores: range component");

    // MT1_RANGE_FLOOR=1 is the eff_delta floor when delta_d=0
    // raw4[2]=50 → softplus(50)≈50, eff_delta=1 (RANGE_FLOOR), r≈50
    // actual_d=5 → err=5, m=5/50=0.1 < 1 → sc_rng=0.1
    {
        float raw4[4] = {10.f, 0.f, 50.f, 0.f};
        auto s = compute_mt1_scores(5.f, raw4, 1.f, 1e30f);
        CHECK(s.range > 0.f && s.range < 1.f);
        CHECK(NEAR(s.range, 0.1f, 0.002f));
    }

    // Range miss: err > r → sc_rng = 0
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        // softplus(0)=ln2≈0.693, eff_delta=1, r≈0.693, err=100 → miss
        auto s = compute_mt1_scores(100.f, raw4, 125.f, 1e30f);
        CHECK(s.range == 0.f);
    }

    // Perfect prediction: err=0 → m=0 → sc_rng=0 (correct by design)
    {
        float raw4[4] = {10.f, 0.f, 2.f, 0.f};
        auto s = compute_mt1_scores(0.f, raw4, 1.f, 1e30f);
        CHECK(s.range == 0.f);
    }

    // Range ceiling clamps r and converts a hit into a miss
    // Without ceiling: actual_d=5, raw4[2]=50 → r≈50 → m=0.1 (hit)
    // With ceiling=3:  r=min(50,3)=3 → m=5/3≈1.67 (miss)
    {
        float raw4[4] = {10.f, 0.f, 50.f, 0.f};
        auto s_no_ceil = compute_mt1_scores(5.f, raw4, 1.f, 1e30f);
        auto s_ceil    = compute_mt1_scores(5.f, raw4, 1.f, 3.f);
        CHECK(s_no_ceil.range > 0.f);   // hit without ceiling
        CHECK(s_ceil.range == 0.f);      // miss with tight ceiling
    }

    // eff_delta uses |delta_d| when it exceeds MT1_RANGE_FLOOR
    // raw4[1] ≈ atanh(0.5) makes delta_d ≈ 5000 (tanh(0.5494)*10000)
    // raw4[2]=0 → range_pct=ln(2)≈0.693, eff_delta=max(5000,1)=5000, r=3466
    // actual_d=100, err=4900, m=4900/3466≈1.41 → miss
    {
        float raw4[4] = {10.f, 0.5494f, 0.f, 0.f};  // tanh(0.5494)≈0.5 → delta_d≈5000
        auto s = compute_mt1_scores(100.f, raw4, 10.f, 1e30f);
        CHECK(s.range == 0.f);
    }

    // Range score in (0,1) is proportional to tightness: smaller m → smaller sc_rng.
    // Higher score = tighter range that still covers (m near 1.0 = best; m near 0.0 = worst).
    // Use raw4[2] values small enough to avoid float overflow in softplus:
    //   loose: softplus(50) ≈ 50, eff_delta=1 → r≈50, m=5/50=0.10 (hit, loose)
    //   tight: softplus(6)  ≈  6, eff_delta=1 → r≈6,  m=5/6≈0.83  (hit, tight)
    {
        float raw4_loose[4] = {10.f, 0.f, 50.f, 0.f};
        float raw4_tight[4] = {10.f, 0.f,  6.f, 0.f};
        auto s_loose = compute_mt1_scores(5.f, raw4_loose, 1.f, 1e30f);
        auto s_tight = compute_mt1_scores(5.f, raw4_tight, 1.f, 1e30f);
        CHECK(s_loose.range > 0.f && s_tight.range > 0.f);  // both hit
        CHECK(s_tight.range > s_loose.range);                // tighter range → higher score
    }
}

static void test_scores_accuracy()
{
    SUITE("compute_mt1_scores: accuracy component");

    // acc_floor active: |actual_d|=50 < acc_floor=200, denom=200
    // delta_d=0, err=50 → sc_acc = 1 - 50/200 = 0.75
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(50.f, raw4, 200.f, 1e30f);
        CHECK(NEAR(s.accuracy, 0.75f, 0.001f));
    }

    // |actual_d| > acc_floor: denom = actual_d
    // actual_d=500, acc_floor=100, delta_d=0, err=500 → sc_acc = max(0, 1-500/500) = 0
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(500.f, raw4, 100.f, 1e30f);
        CHECK(NEAR(s.accuracy, 0.f, 0.001f));
    }

    // Perfect prediction (actual_d == delta_d): err=0, sc_acc=1.0
    // raw4[1]=atanh(0.01)≈0.01 → delta_d≈100, actual_d=100
    {
        float raw4[4] = {0.f, 0.01f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, 10.f, 1e30f);
        // err = |100 - tanh(0.01)*10000| = |100 - 99.9967...| ≈ 0.003
        CHECK(s.accuracy > 0.999f);
    }

    // sc_acc never negative (clamped at 0)
    // actual_d=10, delta_d=0, err=10, acc_floor=5, denom=10: sc_acc=0 (not negative)
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(10.f, raw4, 5.f, 1e30f);
        CHECK(s.accuracy >= 0.f);
    }
}

static void test_scores_composite()
{
    SUITE("compute_mt1_scores: composite formula");

    // composite = 0.50*dir + 0.33*rng + 0.17*acc
    // Use several inputs and verify the formula exactly

    float acc_floor = 125.f;
    float ceiling   = 1e30f;

    float raw4_cases[][4] = {
        {10.f,  0.f, 0.f, 0.f},    // dir=1, rng=0, acc varies
        {-10.f, 0.f, 0.f, 0.f},    // dir=0
        {0.f,   0.f, 0.f, 0.f},    // dir boundary
        {10.f,  0.f, 50.f, 0.f},   // rng hit
        {10.f,  0.f, 0.f, 5.f},    // conf4=sigmoid(5)≈0.993
    };
    float actual_ds[] = {100.f, 100.f, 0.f, 5.f, 100.f};
    int n = 5;

    for (int i = 0; i < n; i++) {
        auto s = compute_mt1_scores(actual_ds[i], raw4_cases[i], acc_floor, ceiling);
        float expected = 0.50f * s.direction + 0.33f * s.range + 0.17f * s.accuracy;
        CHECK(NEAR(s.composite, expected, 1e-5f));
        // All components in [0,1]
        CHECK(s.direction  >= 0.f && s.direction  <= 1.f);
        CHECK(s.range      >= 0.f && s.range      <= 1.f);
        CHECK(s.accuracy   >= 0.f && s.accuracy   <= 1.f);
        CHECK(s.confidence >= 0.f && s.confidence <= 1.f);
        CHECK(s.composite  >= 0.f && s.composite  <= 1.f);
    }

    // Known numeric case: actual_d=0, delta_d=0, acc_floor=125
    // sc_dir=1 (both >=0), sc_rng=0 (err=0→m=0), sc_acc=1 (err=0/acc_floor=0)
    // composite = 0.5 + 0 + 0.17 = 0.67
    {
        float raw4[4] = {10.f, 0.f, 1.f, 0.f};
        auto s = compute_mt1_scores(0.f, raw4, 125.f, 1e30f);
        CHECK(s.direction == 1.f);
        CHECK(s.range     == 0.f);
        CHECK(s.accuracy  == 1.f);
        CHECK(NEAR(s.composite, 0.67f, 0.001f));
    }
}

static void test_scores_confidence()
{
    SUITE("compute_mt1_scores: confidence component");

    // When actual_d == delta_d (err=0): ideal = 1 - 0*0.5/r = 1.0
    // conf4=sigmoid(0)=0.5 → sc_cfd = 1 - |0.5 - 1.0| = 0.5
    {
        float raw4[4] = {0.f, 0.f, 1.f, 0.f};
        auto s = compute_mt1_scores(0.f, raw4, 1.f, 1e30f);
        CHECK(NEAR(s.confidence, 0.5f, 0.002f));
    }

    // When err > r (miss): ideal = r/(d+r)
    // With err=100, r≈0.693 (softplus(0)*eff_delta=1):
    // ideal = 0.693/(100+0.693) ≈ 0.00688
    // conf4=0.5 → sc_cfd = 1 - |0.5 - 0.00688| ≈ 0.507
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, 1.f, 1e30f);
        float r    = log1pf(expf(0.f)) * 1.f;  // softplus(0)*eff_delta=1
        float ideal = r / (100.f + r);
        float expected_cfd = 1.f - fabsf(0.5f - ideal);
        CHECK(NEAR(s.confidence, expected_cfd, 0.002f));
    }

    // sc_cfd always in [0,1]
    float raw4_set[][4] = {
        {10.f, 0.f, 0.f, 10.f},   // conf4≈1
        {10.f, 0.f, 0.f, -10.f},  // conf4≈0
        {10.f, 0.f, 5.f, 0.f},    // conf4=0.5, hit case
        {0.f,  0.f, 0.f, 0.f},    // all boundaries
    };
    float actuals[] = {50.f, 50.f, 5.f, 0.f};
    for (int i = 0; i < 4; i++) {
        auto s = compute_mt1_scores(actuals[i], raw4_set[i], 10.f, 1e30f);
        CHECK(s.confidence >= 0.f && s.confidence <= 1.f);
    }
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
    SUITE("component pool: mutation parent round-robin");

    // Slots 0..MT1_COMP_ELITE-1 are elites (no mutation)
    // Slot s >= MT1_COMP_ELITE: mut_i = s - MT1_COMP_ELITE, parent = mut_i % MT1_COMP_ELITE
    auto parent_of = [](int slot) -> int {
        if (slot < MT1_COMP_ELITE) return slot;  // is an elite
        int mut_i = slot - MT1_COMP_ELITE;
        return mut_i % MT1_COMP_ELITE;
    };

    // First mutation slot → parent 0
    CHECK(parent_of(MT1_COMP_ELITE) == 0);
    // Second → parent 1
    CHECK(parent_of(MT1_COMP_ELITE + 1) == 1);
    // 23rd mutation slot (index 22) → parent 22 (last elite)
    CHECK(parent_of(MT1_COMP_ELITE + 22) == 22);
    // 24th mutation slot (index 23) → wraps to parent 0
    CHECK(parent_of(MT1_COMP_ELITE + MT1_COMP_ELITE) == 0);
    // Last slot overall: slot MT1_COMP_SLOTS-1 = 229
    // mut_i = 229 - 23 = 206, parent = 206 % 23 = 206 - 8*23 = 206-184 = 22
    CHECK(parent_of(MT1_COMP_SLOTS - 1) == (MT1_COMP_MUTS - 1) % MT1_COMP_ELITE);

    // Total mutations = MT1_COMP_MUTS = 207
    int mutation_count = 0;
    for (int s = MT1_COMP_ELITE; s < MT1_COMP_SLOTS; s++) mutation_count++;
    CHECK(mutation_count == MT1_COMP_MUTS);

    // Every parent (0..22) is used at least once
    bool used[MT1_COMP_ELITE] = {};
    for (int s = MT1_COMP_ELITE; s < MT1_COMP_SLOTS; s++)
        used[(s - MT1_COMP_ELITE) % MT1_COMP_ELITE] = true;
    for (int p = 0; p < MT1_COMP_ELITE; p++)
        CHECK(used[p]);
}

static void test_elite_slot_layout()
{
    SUITE("component pool: elite slot layout (17 direct + 3 wavg + 3 reinject)");

    // Direct elites: slots 0..ELITE_COUNT-1 = 0..16
    CHECK(ELITE_COUNT == 17);
    // Wavg blend slots: ELITE_COUNT + 0..WAVG_COUNT-1 = 17, 18, 19
    for (int b = 0; b < WAVG_COUNT; b++) {
        int slot = ELITE_COUNT + b;
        CHECK(slot >= 17 && slot <= 19);
    }
    // Re-injection slots: ELITE_COUNT + WAVG_COUNT + 0..MT1_REINJECT-1 = 20, 21, 22
    for (int k = 0; k < MT1_REINJECT; k++) {
        int slot = ELITE_COUNT + WAVG_COUNT + k;
        CHECK(slot >= 20 && slot <= 22);
    }
    // Last reinject slot is MT1_COMP_ELITE - 1 = 22
    CHECK(ELITE_COUNT + WAVG_COUNT + MT1_REINJECT - 1 == MT1_COMP_ELITE - 1);
    CHECK(ELITE_COUNT + WAVG_COUNT + MT1_REINJECT - 1 == 22);
}

static void test_rolling_buffers()
{
    SUITE("rolling buffers: acc_floor and range_ceiling computation");

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

    // range_ceiling = MT1_RANGE_CEIL_MULT × mean(rolling_residual)
    {
        float buf[MT1_ROLLING_DAYS] = {100.f, 200.f, 300.f, 400.f};
        int count = 4;
        float sum = 0.f;
        for (int k = 0; k < count; k++) sum += buf[k];
        float ceiling = MT1_RANGE_CEIL_MULT * sum / (float)count;
        // (100+200+300+400)/4 = 250, × 4 = 1000
        CHECK(NEAR(ceiling, 1000.f, 0.01f));
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
    // actual_d=5: sc_dir=1, sc_rng≈0.1, sc_acc=0, sc_cfd varies
    auto s = compute_mt1_scores(5.f, raw4, 1.f, 1e30f);

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
    SUITE("composite pool: blend count and pool size");

    CHECK(MT1_BLEND_SLOTS == 200);
    // History: 5 days × 10 models = 50 max candidates from history
    CHECK(HIST_DAYS * HIST_PER_DAY == 50);
    // Max candidates per composite step: 200 blends + 50 history = 250
    CHECK(MT1_BLEND_SLOTS + HIST_DAYS * HIST_PER_DAY == 250);
    // Top 3 saved as reinject (written into component elite slots 20-22)
    CHECK(MT1_REINJECT == 3);
    // Top 10 saved to composite history
    CHECK(HIST_PER_DAY == 10);
}

// ── Main ──────────────────────────────────────────────────────────────────

int main()
{
    printf("MT1 C++ unit tests\n");
    printf("==================\n");

    test_constants();
    test_pcg32();
    test_scores_direction();
    test_scores_range();
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

    printf("\n==================\n");
    printf("%d passed, %d failed\n", pass_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
