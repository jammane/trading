// tests/test_pass_seeding.cpp — Unit tests for the pass-boundary seeding arithmetic.
//
// No BLAS, no file I/O. pass_share_new / pass_interleave come from the shared pass_seeding.h that
// training_v4.cpp also includes, so these grade production's arithmetic rather than a replica.
//
// Build:  cmake --build build --target test_pass_seeding_cpp
// Run:    ./build/test_pass_seeding_cpp   (or: ctest --test-dir build)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include "pass_seeding.h"

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

#define NEAR(a, b, tol) (std::fabs((a) - (b)) < (tol))

static constexpr int ELITE_POOL = 20;

// ── pass_share_new ─────────────────────────────────────────────────────────

static void test_share_both_positive()
{
    current_suite = "share/both+";
    CHECK(NEAR(pass_share_new(0.02, 0.02), 0.5, 1e-9));       // equal -> even split
    CHECK(NEAR(pass_share_new(0.0333, 0.0462), 0.0462 / (0.0333 + 0.0462), 1e-9));
    CHECK(pass_share_new(0.01, 0.09) > 0.85);                 // challenger dominant
    CHECK(pass_share_new(0.09, 0.01) < 0.15);                 // champion dominant
    // Always a proportion.
    CHECK(pass_share_new(0.001, 0.9) <= 1.0);
    CHECK(pass_share_new(0.9, 0.001) >= 0.0);
}

static void test_share_one_negative()
{
    current_suite = "share/one-";
    CHECK(NEAR(pass_share_new(-0.05, 0.02), 1.0, 1e-12));     // only challenger positive
    CHECK(NEAR(pass_share_new(0.02, -0.05), 0.0, 1e-12));     // only champion positive
    // The zero crossing is a cliff by construction — this is the branch measured at 51.6%,
    // barely above chance, and the only one that can still discard a whole set on a near-tie.
    CHECK(NEAR(pass_share_new(-1e-9, 1e-9), 1.0, 1e-12));
    CHECK(NEAR(pass_share_new(1e-9, -1e-9), 0.0, 1e-12));
}

static void test_share_both_negative()
{
    current_suite = "share/both-";
    // Smaller loss must take the LARGER share.
    CHECK(pass_share_new(-0.05, -0.02) > 0.5);                // challenger lost less
    CHECK(pass_share_new(-0.02, -0.05) < 0.5);                // champion lost less
    CHECK(NEAR(pass_share_new(-0.05, -0.02), 0.05 / 0.07, 1e-9));
    CHECK(NEAR(pass_share_new(-0.03, -0.03), 0.5, 1e-9));     // equal losses -> even
    CHECK(NEAR(pass_share_new(0.0, 0.0), 0.5, 1e-9));         // degenerate, no divide by zero
}

static void test_share_direction_is_always_the_same_function()
{
    current_suite = "share/direction";
    // Every branch must agree on WHO wins: share > 0.5 iff chal > champ. Only magnitude differs.
    const double v[] = {-0.09, -0.03, -1e-9, 0.0, 1e-9, 0.03, 0.09};
    for (double a : v) {
        for (double b : v) {
            double s = pass_share_new(a, b);
            if (b > a)      CHECK(s > 0.5 || NEAR(s, 0.5, 1e-9));
            else if (b < a) CHECK(s < 0.5 || NEAR(s, 0.5, 1e-9));
            else            CHECK(NEAR(s, 0.5, 1e-9));
            CHECK(s >= 0.0 && s <= 1.0);
        }
    }
}

// ── pass_interleave ────────────────────────────────────────────────────────

static int count_from(const int* src, int n, int which)
{
    int c = 0;
    for (int k = 0; k < n; k++) if (src[k] == which) c++;
    return c;
}

static void test_interleave_counts()
{
    current_suite = "interleave/counts";
    int src[ELITE_POOL], idx[ELITE_POOL];

    pass_interleave(0.5, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 1) == 10);

    pass_interleave(0.653, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 1) == 13);          // 13.06 -> 13

    pass_interleave(0.347, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 1) == 7);           // 6.94 -> 7

    pass_interleave(0.0, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 1) == 0);           // all champion
    CHECK(count_from(src, ELITE_POOL, 0) == ELITE_POOL);

    pass_interleave(1.0, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 1) == ELITE_POOL);  // all challenger
}

static void test_interleave_preserves_rank_order()
{
    current_suite = "interleave/order";
    int src[ELITE_POOL], idx[ELITE_POOL];
    pass_interleave(0.347, ELITE_POOL, src, idx);
    int last_a = -1, last_b = -1;
    for (int k = 0; k < ELITE_POOL; k++) {
        if (src[k] == 0) { CHECK(idx[k] == last_a + 1); last_a = idx[k]; }
        else             { CHECK(idx[k] == last_b + 1); last_b = idx[k]; }
    }
    // Every output slot filled exactly once from a contiguous prefix of each set.
    CHECK(last_a + last_b + 2 == ELITE_POOL);
}

static void test_interleave_indices_in_range()
{
    current_suite = "interleave/bounds";
    int src[ELITE_POOL], idx[ELITE_POOL];
    for (int pct = 0; pct <= 100; pct++) {
        pass_interleave(pct / 100.0, ELITE_POOL, src, idx);
        for (int k = 0; k < ELITE_POOL; k++) {
            CHECK(src[k] == 0 || src[k] == 1);
            CHECK(idx[k] >= 0 && idx[k] < ELITE_POOL);
        }
    }
}

static void test_interleave_best_of_both_land_early()
{
    current_suite = "interleave/top";
    // The point of preserving order: rank-0 of each set should reach a low slot, because slot 0 is
    // the production model and every parent seeds 9 mutation children.
    int src[ELITE_POOL], idx[ELITE_POOL];
    pass_interleave(0.347, ELITE_POOL, src, idx);
    int first_a = -1, first_b = -1;
    for (int k = 0; k < ELITE_POOL && (first_a < 0 || first_b < 0); k++) {
        if (src[k] == 0 && first_a < 0) first_a = k;
        if (src[k] == 1 && first_b < 0) first_b = k;
    }
    CHECK(first_a >= 0 && first_a <= 2);
    CHECK(first_b >= 0 && first_b <= 2);
}

static void test_interleave_minority_below_resolution_gets_nothing()
{
    current_suite = "interleave/resolution";
    // 20 slots means 5% resolution: a share under ~2.5% rounds the minority set out entirely.
    // Honest consequence of the slot count, not a flaw — but it makes the split categorical at
    // the extremes, which is what the proportional form otherwise avoids.
    int src[ELITE_POOL], idx[ELITE_POOL];
    pass_interleave(0.01, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 1) == 0);
    pass_interleave(0.99, ELITE_POOL, src, idx);
    CHECK(count_from(src, ELITE_POOL, 0) == 0);
}

static void test_regime_labels()
{
    current_suite = "regime";
    CHECK(std::string(pass_regime(0.01, 0.02)) == "both+");
    CHECK(std::string(pass_regime(0.01, -0.02)) == "one-");
    CHECK(std::string(pass_regime(-0.01, 0.02)) == "one-");
    CHECK(std::string(pass_regime(-0.01, -0.02)) == "both-");
    CHECK(std::string(pass_regime(0.0, 0.0)) == "both-");
}

static void test_constants()
{
    current_suite = "constants";
    CHECK(PASS_JUDGE_DAYS == 15);
    CHECK(PASS_REF_VERSION == 1);
}

int main()
{
    printf("Pass-seeding C++ unit tests\n");
    printf("===========================\n");

    test_constants();
    test_share_both_positive();
    test_share_one_negative();
    test_share_both_negative();
    test_share_direction_is_always_the_same_function();
    test_interleave_counts();
    test_interleave_preserves_rank_order();
    test_interleave_indices_in_range();
    test_interleave_best_of_both_land_early();
    test_interleave_minority_below_resolution_gets_nothing();
    test_regime_labels();

    printf("\n===========================\n");
    printf("%d passed, %d failed\n", pass_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
