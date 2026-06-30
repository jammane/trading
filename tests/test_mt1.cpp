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
static constexpr int   MT1_COMP_INJECT     = 5;
static constexpr int   MT1_RANGE_INJECT    = 5;
static constexpr int   MT1_COMP_PARENTS    = ELITE_COUNT + WAVG_COUNT + MT1_COMP_INJECT; // 25
static constexpr int   MT1_COMP_CHILDREN   = 7;
static constexpr int   MT1_COMP_SLOTS      = MT1_COMP_PARENTS * (MT1_COMP_CHILDREN + 1); // 200
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
static constexpr int   MT1_DIR_DAYS        = 5;

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

    float sc_dir = (actual_d >= 0.f) ? conf : (1.f - conf);

    float eff_delta = fmaxf(fabsf(delta_d), MT1_RANGE_FLOOR);
    float r         = range_pct * eff_delta;
    if (range_ceiling < 1e30f) r = fminf(r, range_ceiling);
    float err    = fabsf(fabsf(actual_d) - fabsf(delta_d));  // signless (matches compute_mt1_scores)
    float m      = (r > 1e-9f) ? err / r : (err > 0.f ? 1e9f : 0.f);
    float sc_rng = (m < 1.f) ? m : 0.f;

    float denom  = fmaxf(fabsf(actual_d), acc_floor);
    float sc_acc = fmaxf(0.f, 1.f - err / denom);

    float d      = err;
    float dor    = (r > 1e-9f) ? d / r : (d > 0.f ? 1e9f : 0.f);
    float ideal  = 1.f / (1.f + dor * dor);
    float diff   = conf4 - ideal;
    float sc_cfd = 1.f - diff * diff;
    if (err > r) sc_cfd = 0.5f + 0.25f * sc_cfd;  // compress outside-range to [0.5, 0.75]

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

    // Uniform pool: 25 parents × 8 (7 children + 1 parent) = 200 slots
    CHECK(MT1_COMP_PARENTS  == ELITE_COUNT + WAVG_COUNT + MT1_COMP_INJECT);
    CHECK(MT1_COMP_PARENTS  == 25);
    CHECK(MT1_COMP_CHILDREN == 7);
    CHECK(MT1_COMP_SLOTS    == MT1_COMP_PARENTS * (MT1_COMP_CHILDREN + 1));
    CHECK(MT1_COMP_SLOTS    == 200);
    CHECK(MT1_BLEND_SLOTS   == 200);
    CHECK(HIST_DAYS == 5);
    CHECK(HIST_PER_DAY == 10);
    CHECK(HIST_ELITE + HIST_WAVG == HIST_PER_DAY);   // 7+3 == 10

    // Slot layout: elites 0–16, wavg 17–19, injection 20–24, mutations 25–199
    CHECK(ELITE_COUNT               == 17);
    CHECK(ELITE_COUNT + WAVG_COUNT  == 20);  // first injection slot
    CHECK(ELITE_COUNT + WAVG_COUNT + MT1_COMP_INJECT - 1 == 24);  // last injection slot

    // MT1_FLOOR_COLD / 2 is the cold-start acc_floor
    CHECK(NEAR(MT1_FLOOR_COLD / 2.f, 125.f, 0.001f));

    // Injection cascade counts
    CHECK(MT1_COMP_INJECT  == 5);
    CHECK(MT1_RANGE_INJECT == 5);

    // Direction pool: multi-day scoring window
    CHECK(MT1_DIR_DAYS == 5);
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

static void test_scores_direction()
{
    SUITE("compute_mt1_scores: direction component");

    float acc_floor = 125.f;

    // High conf, actual positive → sc_dir = conf ≈ 1.0
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s.direction, 1.f, 0.001f));
    }
    // Low conf, actual positive → sc_dir = conf ≈ 0.0
    {
        float raw4[4] = {-10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s.direction, 0.f, 0.001f));
    }
    // Low conf, actual negative → sc_dir = 1 - conf ≈ 1.0
    {
        float raw4[4] = {-10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(-100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s.direction, 1.f, 0.001f));
    }
    // High conf, actual negative → sc_dir = 1 - conf ≈ 0.0
    {
        float raw4[4] = {10.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(-100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s.direction, 0.f, 0.001f));
    }
    // Neutral (conf=0.5) → sc_dir = 0.5 regardless of sign — random baseline
    {
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};
        auto s = compute_mt1_scores(100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s.direction, 0.5f, 0.001f));
        auto s2 = compute_mt1_scores(-100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s2.direction, 0.5f, 0.001f));
        auto s3 = compute_mt1_scores(0.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(s3.direction, 0.5f, 0.001f));
    }
    // Known exact value: sigmoid(log(3)) = 0.75
    // actual positive → sc_dir = 0.75; actual negative → sc_dir = 0.25
    {
        float raw4[4] = {logf(3.f), 0.f, 0.f, 0.f};
        auto sp = compute_mt1_scores( 100.f, raw4, acc_floor, 1e30f);
        auto sn = compute_mt1_scores(-100.f, raw4, acc_floor, 1e30f);
        CHECK(NEAR(sp.direction, 0.75f, 0.001f));
        CHECK(NEAR(sn.direction, 0.25f, 0.001f));
    }
    // sc_dir always in [0, 1]
    {
        float raw4[4] = {5.f, 0.f, 0.f, 0.f};
        auto sp = compute_mt1_scores( 200.f, raw4, acc_floor, 1e30f);
        auto sn = compute_mt1_scores(-200.f, raw4, acc_floor, 1e30f);
        CHECK(sp.direction >= 0.f && sp.direction <= 1.f);
        CHECK(sn.direction >= 0.f && sn.direction <= 1.f);
        // symmetric: dir(positive) + dir(negative) == 1.0
        CHECK(NEAR(sp.direction + sn.direction, 1.f, 0.001f));
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

    // Known numeric case: actual_d=0, raw4[0]=10 → conf≈1, actual_d>=0 → sc_dir≈1
    // sc_rng=0 (err=0→m=0), sc_acc=1 (err=0), composite≈0.5+0+0.17=0.67
    {
        float raw4[4] = {10.f, 0.f, 1.f, 0.f};
        auto s = compute_mt1_scores(0.f, raw4, 125.f, 1e30f);
        CHECK(NEAR(s.direction, 1.f, 0.001f));
        CHECK(s.range     == 0.f);
        CHECK(s.accuracy  == 1.f);
        CHECK(NEAR(s.composite, 0.67f, 0.001f));
    }
}

static void test_scores_confidence()
{
    SUITE("compute_mt1_scores: confidence component (ideal=1/(1+(d/r)²), score=1-(conf4-ideal)²)");

    // Perfect prediction (d=0): ideal=1.0
    // conf4=sigmoid(10)≈1.0  → diff≈0  → sc_cfd≈1.0
    // conf4=sigmoid(0)=0.5   → diff=−0.5 → sc_cfd=1−0.25=0.75
    {
        float raw4_hi[4] = {0.f, 0.f, 1.f, 10.f};  // conf4≈1
        auto s_hi = compute_mt1_scores(0.f, raw4_hi, 1.f, 1e30f);
        CHECK(NEAR(s_hi.confidence, 1.f, 0.002f));

        float raw4_lo[4] = {0.f, 0.f, 1.f, 0.f};   // conf4=0.5
        auto s_lo = compute_mt1_scores(0.f, raw4_lo, 1.f, 1e30f);
        CHECK(NEAR(s_lo.confidence, 0.75f, 0.002f));
    }

    // Error on range boundary (d=r): ideal=1/(1+1)=0.5
    // conf4=sigmoid(0)=0.5 → diff=0 → sc_cfd=1.0
    // We need actual_d such that err==r.  Use raw4={0, x, 0, 0}: delta_d=tanh(x)*10000,
    // r=softplus(0)*max(|delta_d|,1)=ln(2)*1≈0.693.  Set actual_d=0.693+delta_d (err=r).
    {
        float r_val = log1pf(expf(0.f));           // softplus(0)*eff_delta=1 ≈ 0.6931
        float raw4[4] = {0.f, 0.f, 0.f, 0.f};     // delta_d=0, r=0.6931, conf4=0.5
        auto s = compute_mt1_scores(r_val, raw4, 1.f, 1e30f);  // actual_d=r → err=r → dor=1
        CHECK(NEAR(s.confidence, 1.f, 0.002f));  // conf4=0.5==ideal=0.5 → sc_cfd=1
    }

    // Large miss (d>>r, err>r): scores compressed to [0.5, 0.75]
    // d=100, r≈0.693: dor≈144 → ideal≈0.000048
    // conf4≈0: diff≈0 → raw sc_cfd≈1.0 → compressed: 0.5+0.25*1.0=0.75
    // conf4=0.5: diff≈0.5 → raw sc_cfd=0.75 → compressed: 0.5+0.25*0.75=0.6875
    {
        float raw4_lo[4] = {0.f, 0.f, 0.f, -10.f};  // conf4≈0
        auto s_lo = compute_mt1_scores(100.f, raw4_lo, 1.f, 1e30f);
        CHECK(NEAR(s_lo.confidence, 0.75f, 0.002f));   // compressed: 0.5+0.25*1.0

        float raw4_mid[4] = {0.f, 0.f, 0.f, 0.f};   // conf4=0.5
        auto s_mid = compute_mt1_scores(100.f, raw4_mid, 1.f, 1e30f);
        CHECK(NEAR(s_mid.confidence, 0.6875f, 0.002f)); // compressed: 0.5+0.25*0.75
    }

    // sc_cfd always in [0,1]: outside-range cases compress to [0.5,0.75], still in [0,1]
    float raw4_set[][4] = {
        {10.f, 0.f, 0.f, 10.f},   // conf4≈1, perfect pred (d=0) → inside range → sc_cfd≈1
        {10.f, 0.f, 0.f, -10.f},  // conf4≈0, big miss (err>r) → compressed → sc_cfd∈[0.5,0.75]
        {10.f, 0.f, 5.f, 0.f},    // conf4=0.5, hit case
        {0.f,  0.f, 0.f, 0.f},    // boundaries
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
    SUITE("component pool: mutation parent round-robin (7 children × 25 parents)");

    // Slots 0..MT1_COMP_PARENTS-1 are parents (elites + wavg + inject, no mutation)
    // Slot s >= MT1_COMP_PARENTS: mut_i = s - MT1_COMP_PARENTS, parent = mut_i % MT1_COMP_PARENTS
    auto parent_of = [](int slot) -> int {
        if (slot < MT1_COMP_PARENTS) return slot;  // is a parent
        int mut_i = slot - MT1_COMP_PARENTS;
        return mut_i % MT1_COMP_PARENTS;
    };

    // First mutation slot → parent 0
    CHECK(parent_of(MT1_COMP_PARENTS) == 0);
    // Second → parent 1
    CHECK(parent_of(MT1_COMP_PARENTS + 1) == 1);
    // 25th mutation slot (index 24) → parent 24 (last parent)
    CHECK(parent_of(MT1_COMP_PARENTS + 24) == 24);
    // 26th mutation slot (index 25) → wraps to parent 0
    CHECK(parent_of(MT1_COMP_PARENTS + MT1_COMP_PARENTS) == 0);
    // Last slot overall: slot 199, mut_i = 199-25 = 174, parent = 174 % 25 = 24
    CHECK(parent_of(MT1_COMP_SLOTS - 1) == (MT1_COMP_SLOTS - 1 - MT1_COMP_PARENTS) % MT1_COMP_PARENTS);
    CHECK(parent_of(MT1_COMP_SLOTS - 1) == 24);

    // Total mutations = MT1_COMP_SLOTS - MT1_COMP_PARENTS = 175
    int mutation_count = 0;
    for (int s = MT1_COMP_PARENTS; s < MT1_COMP_SLOTS; s++) mutation_count++;
    CHECK(mutation_count == 175);
    CHECK(mutation_count == MT1_COMP_PARENTS * MT1_COMP_CHILDREN);

    // Every parent (0..24) is used at least once
    bool used[MT1_COMP_PARENTS] = {};
    for (int s = MT1_COMP_PARENTS; s < MT1_COMP_SLOTS; s++)
        used[(s - MT1_COMP_PARENTS) % MT1_COMP_PARENTS] = true;
    for (int p = 0; p < MT1_COMP_PARENTS; p++)
        CHECK(used[p]);
}

static void test_elite_slot_layout()
{
    SUITE("component pool: elite slot layout (17 direct + 3 wavg + 5 injection)");

    // Direct elites: slots 0..ELITE_COUNT-1 = 0..16
    CHECK(ELITE_COUNT == 17);
    // Wavg blend slots: ELITE_COUNT + 0..WAVG_COUNT-1 = 17, 18, 19
    for (int b = 0; b < WAVG_COUNT; b++) {
        int slot = ELITE_COUNT + b;
        CHECK(slot >= 17 && slot <= 19);
    }
    // Injection slots: ELITE_COUNT + WAVG_COUNT + 0..MT1_COMP_INJECT-1 = 20..24
    for (int k = 0; k < MT1_COMP_INJECT; k++) {
        int slot = ELITE_COUNT + WAVG_COUNT + k;
        CHECK(slot >= 20 && slot <= 24);
    }
    // Last injection slot = 24 = MT1_COMP_PARENTS - 1
    CHECK(ELITE_COUNT + WAVG_COUNT + MT1_COMP_INJECT - 1 == 24);
    CHECK(ELITE_COUNT + WAVG_COUNT + MT1_COMP_INJECT - 1 == MT1_COMP_PARENTS - 1);
    // Mutations start at slot 25
    CHECK(MT1_COMP_PARENTS == 25);
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
    // actual_d=5: sc_dir=conf≈1 (actual positive), sc_rng≈0.1, sc_acc=0, sc_cfd varies
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
    // Top 5 saved as comp_inject (composite→direction + range slots 20–24)
    CHECK(MT1_COMP_INJECT == 5);
    // Top 10 saved to composite history
    CHECK(HIST_PER_DAY == 10);
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

static void test_direction_balanced_weighting() {
    SUITE("direction balanced weighting");
    // Skewed window: 8 up-days, 2 down-days; arbitrary positive recency weights.
    int   up[10] = {1,1,1,1,1,1,1,1,0,0};
    float w[10]  = {1.0f,1.1f,1.2f,1.3f,1.4f,1.5f,1.6f,1.7f,1.8f,1.9f};
    float W = 0.f; for (int i = 0; i < 10; i++) W += w[i];

    // Constant "always up" at confidence c → no-skill baseline W/2 for ANY c (the whole point).
    const float cs[4] = {0.5f, 0.7f, 0.9f, 0.99f};
    for (int ci = 0; ci < 4; ci++) { float c = cs[ci];
        float ds[10]; for (int i = 0; i < 10; i++) ds[i] = up[i] ? c : (1.f - c);
        CHECK(NEAR(balanced_dir_total(up, w, ds, 10), W / 2.f, 1e-3f));
    }
    // Perfect → W, worst → 0.
    { float ds[10]; for (int i=0;i<10;i++) ds[i]=1.f; CHECK(NEAR(balanced_dir_total(up,w,ds,10), W, 1e-3f)); }
    { float ds[10]; for (int i=0;i<10;i++) ds[i]=0.f; CHECK(NEAR(balanced_dir_total(up,w,ds,10), 0.f, 1e-3f)); }
    // Genuine skill (correct on both classes at 0.9) → 0.9·W, strictly above no-skill.
    { float ds[10]; for (int i=0;i<10;i++) ds[i]=0.9f; float t=balanced_dir_total(up,w,ds,10);
      CHECK(NEAR(t, 0.9f*W, 1e-3f)); CHECK(t > W/2.f); }
    // Single-class window (all up) → balancing disabled, plain weighted sum.
    { int u[3]={1,1,1}; float ww[3]={1.f,1.f,1.f}; float ds[3]={0.8f,0.8f,0.8f};
      CHECK(NEAR(balanced_dir_total(u, ww, ds, 3), 0.8f*3.f, 1e-3f)); }
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
    test_drift_rel_metric();
    test_direction_balanced_weighting();

    printf("\n==================\n");
    printf("%d passed, %d failed\n", pass_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
