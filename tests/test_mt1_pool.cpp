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

int main() {
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

    printf("\n==================\n%d passed, %d failed\n", g_checks - g_fails, g_fails);
    return g_fails ? 1 : 0;
}
