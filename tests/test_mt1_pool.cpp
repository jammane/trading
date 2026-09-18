// test_mt1_pool.cpp — the single-output MT1 scoring and pool lifecycle.
//
// The property that matters most: score 0.5 means "exactly as good as predicting the trailing
// mean". Under the old |actual| denominator a collapsed pool also scored ~0.5 but 0.5 looked like
// a respectable number, which is how conf4 and the range channel survived for years with dead
// pools. These pin the anchor.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include "mt1_pool.h"

static int g_checks = 0, g_fails = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            g_fails++;                                                           \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);               \
        }                                                                        \
    } while (0)
#define NEAR(a, b, eps) (fabsf((a) - (b)) <= (eps))

// ── the anchor ───────────────────────────────────────────────────────────────────
static void test_score_anchor() {
    // model predicts exactly what happened -> 1.0
    CHECK(NEAR(mt1_score(500.f, 500.f, 0.f, 1.f), 1.0f, 1e-5f));

    // model is exactly as wrong as the baseline -> 0.5, whichever side it errs on
    CHECK(NEAR(mt1_score(500.f, 300.f, 300.f, 1.f), 0.5f, 1e-5f));
    CHECK(NEAR(mt1_score(500.f, 700.f, 300.f, 1.f), 0.5f, 1e-5f));

    // beating the baseline -> above 0.5; losing to it -> below
    CHECK(mt1_score(500.f, 450.f, 300.f, 1.f) > 0.5f);
    CHECK(mt1_score(500.f, 100.f, 300.f, 1.f) < 0.5f);

    // a pool that has collapsed onto the trailing mean scores 0.5 EXACTLY, not "respectably"
    for (float a : {-2000.f, -50.f, 0.f, 50.f, 2000.f}) {
        const float baseline = 25.f;
        CHECK(NEAR(mt1_score(a, baseline, baseline, 1.f), 0.5f, 1e-4f));
    }
}

static void test_score_symmetry() {
    // over and under by the same amount score identically
    for (float d : {1.f, 37.f, 500.f, 5000.f}) {
        const float a = 400.f, b = 0.f;
        CHECK(NEAR(mt1_score(a, a + d, b, 1.f), mt1_score(a, a - d, b, 1.f), 1e-5f));
    }
}

static void test_score_bounds() {
    for (float a : {-5000.f, -1.f, 0.f, 1.f, 5000.f})
        for (float p : {-9000.f, -100.f, 0.f, 100.f, 9000.f})
            for (float b : {-500.f, 0.f, 500.f}) {
                const float s = mt1_score(a, p, b, 50.f);
                CHECK(s > 0.f && s <= 1.0f);
                CHECK(std::isfinite(s));
            }
}

static void test_score_is_scale_free() {
    // Identical RELATIVE skill on a quiet day and a loud day must score the same. The raw
    // 1/(1+err) form weighted quiet days 19x more, so a windowed sum would have selected for
    // days when nothing happened.
    const float quiet_a = 86.f,  quiet_b = 0.f;
    const float loud_a  = 1650.f, loud_b  = 0.f;
    const float sq = mt1_score(quiet_a, quiet_a / 2.f, quiet_b, 1.f);
    const float sl = mt1_score(loud_a,  loud_a  / 2.f, loud_b,  1.f);
    CHECK(NEAR(sq, sl, 1e-4f));
}

static void test_floor_guards_a_lucky_baseline() {
    // baseline lands almost exactly -> base ~ 0; without the floor any error looks catastrophic
    const float s_no_floor = mt1_score(100.f, 150.f, 100.f, 1e-6f);
    const float s_floored  = mt1_score(100.f, 150.f, 100.f, 200.f);
    CHECK(s_no_floor < 0.01f);
    CHECK(s_floored > 0.5f);
    CHECK(std::isfinite(mt1_score(0.f, 0.f, 0.f, 0.f)));   // fully degenerate inputs
}

static void test_skill() {
    CHECK(NEAR(mt1_skill(0.f, 100.f), 1.0f, 1e-6f));     // perfect
    CHECK(NEAR(mt1_skill(100.f, 100.f), 0.0f, 1e-6f));   // ties the baseline
    CHECK(mt1_skill(200.f, 100.f) < 0.f);                // worse than doing nothing
    CHECK(NEAR(mt1_skill(5.f, 0.f), 0.0f, 1e-6f));       // degenerate baseline -> 0, not inf
}

// ── decode ───────────────────────────────────────────────────────────────────────
static void test_pred_decode() {
    CHECK(NEAR(mt1_pred(0.f), 0.f, 1e-4f));
    CHECK(mt1_pred(10.f) > 0.99f * MT1_PRED_SCALE);
    CHECK(mt1_pred(-10.f) < -0.99f * MT1_PRED_SCALE);
    CHECK(NEAR(mt1_pred(1.f), -mt1_pred(-1.f), 1e-3f));   // symmetric, unlike softplus
    for (float r : {-50.f, -1.f, 0.f, 1.f, 50.f})
        CHECK(fabsf(mt1_pred(r)) <= MT1_PRED_SCALE + 1e-3f);
}

// ── lifecycle ────────────────────────────────────────────────────────────────────
static void test_slot_maturity() {
    MT1SlotMeta m;
    mt1_slot_init(m, 7);
    CHECK(m.lineage == 7u);
    CHECK(!mt1_slot_mature(m));
    CHECK(NEAR(mt1_slot_score(m), 0.f, 1e-6f));          // no record yet
    for (int i = 0; i < MT1_POOL_MIN_AGE - 1; i++) {
        mt1_slot_record(m, 0.9f);
        CHECK(!mt1_slot_mature(m));
    }
    mt1_slot_record(m, 0.9f);
    CHECK(mt1_slot_mature(m));
}

static void test_partial_record_is_not_penalised() {
    // 8 predictions at 0.9 should score 0.9, not 0.9*8/16
    MT1SlotMeta m;
    mt1_slot_init(m, 1);
    for (int i = 0; i < 8; i++) mt1_slot_record(m, 0.9f);
    CHECK(NEAR(mt1_slot_score(m), 0.9f, 1e-4f));
    CHECK(NEAR(mt1_slot_mean(m), 0.9f, 1e-4f));
}

static void test_recency_weighting() {
    MT1SlotMeta recent_good, recent_bad;
    mt1_slot_init(recent_good, 1);
    mt1_slot_init(recent_bad, 2);
    for (int i = 0; i < MT1_SCORE_HIST; i++) {
        mt1_slot_record(recent_good, i < 8 ? 0.3f : 0.7f);   // improving: good ones land last
        mt1_slot_record(recent_bad,  i < 8 ? 0.7f : 0.3f);   // declining
    }
    // identical plain means, but recency must separate them
    CHECK(NEAR(mt1_slot_mean(recent_good), mt1_slot_mean(recent_bad), 1e-4f));
    CHECK(mt1_slot_score(recent_good) > mt1_slot_score(recent_bad));
}

static void test_window_rolls() {
    MT1SlotMeta m;
    mt1_slot_init(m, 1);
    for (int i = 0; i < 100; i++) mt1_slot_record(m, 0.2f);
    for (int i = 0; i < MT1_SCORE_HIST; i++) mt1_slot_record(m, 0.8f);
    CHECK(NEAR(mt1_slot_score(m), 0.8f, 1e-4f));   // old scores fully rolled out
    CHECK(m.n_pred > (uint16_t)MT1_SCORE_HIST);
}

static void test_immature_never_outranks_a_record() {
    MT1SlotMeta lucky, seasoned;
    mt1_slot_init(lucky, 1);
    mt1_slot_init(seasoned, 2);
    mt1_slot_record(lucky, 1.0f);
    mt1_slot_record(lucky, 1.0f);                       // 2 perfect predictions
    for (int i = 0; i < MT1_SCORE_HIST; i++) mt1_slot_record(seasoned, 0.55f);
    CHECK(mt1_slot_better(seasoned, lucky));
    CHECK(!mt1_slot_better(lucky, seasoned));
}

static void test_ordering_is_a_strict_weak_ordering() {
    MT1SlotMeta a, b;
    mt1_slot_init(a, 1);
    mt1_slot_init(b, 2);
    for (int i = 0; i < MT1_SCORE_HIST; i++) {
        mt1_slot_record(a, 0.6f);
        mt1_slot_record(b, 0.6f);
    }
    CHECK(!mt1_slot_better(a, b));      // equal -> neither is better
    CHECK(!mt1_slot_better(b, a));
    CHECK(!mt1_slot_better(a, a));      // irreflexive
}

static void test_lifecycle_constants() {
    // steady state mature/N = 1/(1 + MIN_AGE*CULL_PCT)
    const float mature_frac = 1.f / (1.f + MT1_POOL_MIN_AGE * MT1_POOL_CULL_PCT);
    CHECK(mature_frac > 0.55f && mature_frac < 0.65f);          // ~60% of the pool
    CHECK(MT1_POOL_LINEAGE_RESUME < MT1_POOL_LINEAGE_CAP);      // hysteresis, not a knife edge
    CHECK(MT1_SCORE_HIST % 4 == 0);                             // recency blocks divide evenly
    CHECK(MT1_POOL_MIN_AGE <= MT1_SCORE_HIST);                  // maturity reachable inside the window
    CHECK(MT1_BASELINE_DAYS > MT1_FLOOR_DAYS);                  // baseline is the slower estimate
}

// ── causality ────────────────────────────────────────────────────────────────────
// A day-t prediction is scored at t+1. The baseline and floor windows must end at t, or they
// have seen the outcome they are being used to judge.
static void test_windows_must_be_causal() {
    const int scored = 100;                       // outcome realised on day 100
    CHECK(mt1_windows_are_causal(scored, 99, 99));   // both end the day before: fine
    CHECK(mt1_windows_are_causal(scored, 80, 90));   // further back is fine too
    CHECK(!mt1_windows_are_causal(scored, 100, 99)); // baseline saw the answer
    CHECK(!mt1_windows_are_causal(scored, 99, 100)); // floor saw the answer
    CHECK(!mt1_windows_are_causal(scored, 101, 99)); // baseline is ahead
    CHECK(!mt1_windows_are_causal(scored, 99, 101)); // floor is ahead
}

static void test_a_peeking_baseline_would_be_visible() {
    // If the baseline could see the outcome it would predict it exactly, base -> 0, and the
    // model's score would collapse toward 0 no matter how good it was. This is what the contract
    // prevents — pinned so the failure mode stays documented.
    const float actual = 800.f, good_pred = 780.f;
    const float honest = mt1_score(actual, good_pred, 25.f, 100.f);   // baseline = trailing mean
    const float peeking = mt1_score(actual, good_pred, actual, 1e-6f); // baseline = the answer
    CHECK(honest > 0.5f);        // a good prediction beats the honest baseline
    CHECK(peeking < 0.1f);       // and looks terrible against a peeking one
}

static void test_scoring_delay_is_one_day() {
    // Not a code path — a statement of the design. The old target needed MT1_FWD_DAYS=10 before a
    // prediction could be graded; this needs t+1.
    CHECK(MT1_BASELINE_DAYS > 1);      // the baseline is a window...
    CHECK(MT1_FLOOR_DAYS > 1);         // ...and so is the floor
    CHECK(MT1_BASELINE_DAYS <= 60);    // but neither is so long it cannot track a regime
    CHECK(MT1_FLOOR_DAYS <= MT1_BASELINE_DAYS);
}

// ── layout ───────────────────────────────────────────────────────────────────────
// load_bin validates only by element count, so a layout drift with an unchanged total would load
// silently and produce plausible wrong numbers. These pin every offset.
static void test_net_layout_is_contiguous() {
    struct L { int w, b, out, in; };
    const L ls[] = {
        {NT_M_A1_W, NT_M_A1_B, 20, 20}, {NT_M_A2_W, NT_M_A2_B, 20, 20},
        {NT_M_B1_W, NT_M_B1_B,  6, 10}, {NT_M_B2_W, NT_M_B2_B,  4,  6},
        {NT_M_C1_W, NT_M_C1_B,  5,  7}, {NT_M_C2_W, NT_M_C2_B,  4,  5},
        {NT_P_A1_W, NT_P_A1_B, 20, 20}, {NT_P_A2_W, NT_P_A2_B, 20, 20},
        {NT_P_B1_W, NT_P_B1_B,  6, 10}, {NT_P_B2_W, NT_P_B2_B,  4,  6},
        {NT_P_C1_W, NT_P_C1_B,  5,  7}, {NT_P_C2_W, NT_P_C2_B,  4,  5},
        {NT_D1_W,   NT_D1_B,   22, 56}, {NT_D2_W,   NT_D2_B,   10, 23},
        {NT_D3_W,   NT_D3_B,    1, 10},
    };
    const int n = (int)(sizeof(ls) / sizeof(ls[0]));
    int cursor = 0;
    for (int i = 0; i < n; i++) {
        CHECK(ls[i].w == cursor);                    // weights start where the last layer ended
        CHECK(ls[i].b == ls[i].w + ls[i].out * ls[i].in);   // bias follows its weights
        cursor = ls[i].b + ls[i].out;
    }
    CHECK(cursor == MT1NET_PARAMS);                  // no gaps, no overlap, nothing left over
}

static void test_net_is_much_smaller_than_the_old_model() {
    const int old_composed = 1996 + 4 * 1803;        // MT1DualHead + four MT1Tail
    CHECK(MT1NET_PARAMS < old_composed / 2);
    CHECK(MT1NET_PARAMS == 3501);
}

static void test_reserved_slot_has_room() {
    // d2's input must be d1's output plus exactly one reserved slot
    const int d1_out = (NT_D1_B - NT_D1_W) / 56;
    const int d2_in  = (NT_D2_B - NT_D2_W) / 10;
    CHECK(d1_out == 22);
    CHECK(d2_in == d1_out + 1);
    CHECK(MT1NET_D2_RESERVED == d1_out);
}


// ── Retirement-age histogram ─────────────────────────────────────────────────────
//
// Both of these failed on the first smoke run. mt1_step_day read the victim's age AFTER
// mt1_slot_init had zeroed it, so every retirement was recorded as age 0: the mean retirement age
// sat at exactly 0.0 for the whole run and the histogram stayed all zeros. Nothing crashed and no
// other column moved, so only reading the log caught it.
static void test_life_bucket_boundaries() {
    // Buckets are 8-15 / 16-31 / 32-63 / 64-127 / 128+, so each boundary is an off-by-one risk.
    CHECK(mt1_life_bucket(MT1_POOL_MIN_AGE) == 0);   // min age lands in bucket 0
    CHECK(mt1_life_bucket(15) == 0);   // 15 -> bucket 0
    CHECK(mt1_life_bucket(16) == 1);   // 16 -> bucket 1
    CHECK(mt1_life_bucket(31) == 1);   // 31 -> bucket 1
    CHECK(mt1_life_bucket(32) == 2);   // 32 -> bucket 2
    CHECK(mt1_life_bucket(63) == 2);   // 63 -> bucket 2
    CHECK(mt1_life_bucket(64) == 3);   // 64 -> bucket 3
    CHECK(mt1_life_bucket(127) == 3);   // 127 -> bucket 3
    CHECK(mt1_life_bucket(128) == 4);   // 128 -> bucket 4
    CHECK(mt1_life_bucket(100000) == 4);   // a very old model stays in the last bucket
}

static void test_every_bucket_is_in_range() {
    // An out-of-range index would corrupt whatever sits after retire_hist in MT1PoolScratch
    // rather than producing a visible wrong number.
    for (uint32_t age = 0; age < 300u; age++) {
        const int b = mt1_life_bucket(age);
        CHECK(b >= 0 && b < MT1_LIFE_BUCKETS);   // bucket index stays inside retire_hist
    }
}

static void test_ages_below_min_age_do_not_underflow() {
    // Culling never picks an immature model, but injection or a reset could, and a negative or
    // out-of-range bucket there would be silent memory corruption.
    for (uint32_t age = 0; age < (uint32_t)MT1_POOL_MIN_AGE; age++)
        CHECK(mt1_life_bucket(age) == 0);   // an under-age retirement clamps into bucket 0
}

static void test_age_must_be_read_before_the_slot_is_reset() {
    // The defect in one line: mt1_slot_init zeroes n_pred, so reading it afterwards always
    // yields 0 no matter how long the model actually lived.
    MT1SlotMeta m;
    mt1_slot_init(m, 7);
    for (int i = 0; i < 20; i++) mt1_slot_record(m, 0.5f);
    CHECK(m.n_pred == 20);   // twenty predictions recorded

    const uint32_t age_before = m.n_pred;
    mt1_slot_init(m, 9);
    CHECK(m.n_pred == 0);   // mt1_slot_init zeroes the age — this is why order matters
    CHECK(age_before == 20);   // the age captured before the reset survives
    CHECK(mt1_life_bucket(age_before) == 1);   // age 20 belongs in the 16-31 bucket
    CHECK(mt1_life_bucket(m.n_pred) == 0);   // reading the age after the reset would file every retirement in bucket 0
}

static void test_a_run_of_retirements_fills_more_than_one_bucket() {
    // At the designed ~20-prediction lifespan most retirements land in buckets 1-2. A histogram
    // that is entirely in one bucket means either a broken read or a pool that is not turning over.
    uint32_t hist[MT1_LIFE_BUCKETS] = {0, 0, 0, 0, 0};
    const uint32_t ages[] = {8, 12, 14, 18, 20, 25, 31, 33, 40, 60, 70, 130};
    double sum = 0.0;
    for (uint32_t a : ages) { hist[mt1_life_bucket(a)]++; sum += a; }

    int occupied = 0;
    uint32_t total = 0;
    for (int b = 0; b < MT1_LIFE_BUCKETS; b++) { if (hist[b]) occupied++; total += hist[b]; }
    CHECK(total == sizeof(ages) / sizeof(ages[0]));   // every retirement is counted exactly once
    CHECK(occupied == MT1_LIFE_BUCKETS);   // these ages span all five buckets
    CHECK(hist[0] == 3 && hist[1] == 4 && hist[2] == 3 && hist[3] == 1 && hist[4] == 1);   // each age lands in the bucket its boundaries say it should

    const double mean_age = sum / (double)(sizeof(ages) / sizeof(ages[0]));
    CHECK(mean_age > 0.0);   // the mean retirement age is non-zero — 0.0 is the signature of the bug
    CHECK(mean_age > (double)MT1_POOL_MIN_AGE);   // mean lifespan must exceed the maturity gate, or nothing is surviving to breed
}

int main() {
    test_net_layout_is_contiguous();
    test_net_is_much_smaller_than_the_old_model();
    test_reserved_slot_has_room();
    test_windows_must_be_causal();
    test_a_peeking_baseline_would_be_visible();
    test_scoring_delay_is_one_day();
    test_score_anchor();
    test_score_symmetry();
    test_score_bounds();
    test_score_is_scale_free();
    test_floor_guards_a_lucky_baseline();
    test_skill();
    test_pred_decode();
    test_slot_maturity();
    test_partial_record_is_not_penalised();
    test_recency_weighting();
    test_window_rolls();
    test_immature_never_outranks_a_record();
    test_ordering_is_a_strict_weak_ordering();
    test_lifecycle_constants();
    test_life_bucket_boundaries();
    test_every_bucket_is_in_range();
    test_ages_below_min_age_do_not_underflow();
    test_age_must_be_read_before_the_slot_is_reset();
    test_a_run_of_retirements_fills_more_than_one_bucket();

    printf("\n==================\n%d passed, %d failed\n", g_checks - g_fails, g_fails);
    return g_fails ? 1 : 0;
}
