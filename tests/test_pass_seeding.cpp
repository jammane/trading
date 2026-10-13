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
    CHECK(PASS_JUDGE_MIN_DAYS == 15);
    CHECK(PASS_REF_VERSION == 2);
    CHECK(PASS_JUDGE_RECENCY == 0.995);
    // half-life ~138 days -- the figure CLAUDE.md and the race header quote
    CHECK(NEAR(std::log(0.5) / std::log(PASS_JUDGE_RECENCY), 138.3, 0.5));
}

// ── PassJudge: the champion score ─────────────────────────────────────────

static void test_judge_constant_return_scores_that_return()
{
    current_suite = "judge/constant";
    PassJudge j;
    double book = 25000.0;
    for (int age = 600; age >= 0; age--) { j.add(book, book * 1.001, age); book *= 1.001; }
    CHECK(NEAR(j.score(), 0.001, 1e-12));    // any weighting of a constant is that constant
    CHECK(j.have());
    CHECK(j.n == 601);
}

static void test_judge_reset_day_scores_zero_not_a_jump()
{
    current_suite = "judge/reset";
    // The trainer's reset path sets book_prev == slot0_score at $25,000. v1 measured the book's
    // level, so a reset from $22,400 read as +11.6%. v2 sums returns: the day must be exactly 0.
    PassJudge a, b;
    for (int age = 9; age >= 1; age--) { a.add(25000.0, 25025.0, age); b.add(25000.0, 25025.0, age); }
    a.add(25000.0, 25000.0, 0);              // reset day
    b.add(22400.0, 25000.0, 0);              // what a level-based metric would have seen
    CHECK(a.score() < 0.001);                // diluted by one zero, never inflated
    CHECK(a.score() > 0.0009 * 0.9);
    CHECK(b.score() > 10.0 * a.score());     // the contamination the fix removes
}

static void test_judge_recent_days_count_more()
{
    current_suite = "judge/recency";
    // Same returns, opposite order: the pass that did well LATE must outscore the one that did
    // well early, and by a lot when the gap is ~a year.
    PassJudge late, early;
    for (int age = 399; age >= 0; age--) {
        const bool recent = age < 200;
        late.add (25000.0, 25000.0 * (recent ? 1.002 : 0.999), age);
        early.add(25000.0, 25000.0 * (recent ? 0.999 : 1.002), age);
    }
    CHECK(late.score() > 0.0);
    CHECK(early.score() < 0.0);
    // weight at age 138 is half that at age 0
    PassJudge h;
    h.add(25000.0, 25000.0 * 1.01, 0);
    h.add(25000.0, 25000.0 * 0.99, 138);
    CHECK(h.score() > 0.0);                  // the newer day dominates
    CHECK(NEAR(h.score(), (0.01 - 0.01 * std::pow(0.995, 138)) / (1 + std::pow(0.995, 138)), 1e-12));
}

static void test_judge_effective_days()
{
    current_suite = "judge/eff";
    PassJudge j;
    for (int age = 1237; age >= 0; age--) j.add(25000.0, 25010.0, age);
    // (1+g)/(1-g) = 399 for an infinite series at g = 0.995; a 1,238-day pass is near that
    CHECK(j.eff_days() > 380.0 && j.eff_days() < 400.0);
    PassJudge one; one.add(25000.0, 25010.0, 0);
    CHECK(NEAR(one.eff_days(), 1.0, 1e-12));
}

static void test_judge_ignores_invalid_days()
{
    current_suite = "judge/invalid";
    PassJudge j;
    CHECK(!j.have());
    CHECK(j.score() == 0.0);
    j.add(0.0, 25000.0, 0);                  // no book
    j.add(25000.0, std::nan(""), 0);         // non-finite outcome
    CHECK(!j.have());
    j.add(25000.0, 25250.0, 5);
    CHECK(j.have());
    CHECK(NEAR(j.score(), 0.01, 1e-12));
    // negative age (cannot happen; clamped rather than overweighted)
    PassJudge k; k.add(25000.0, 25250.0, -3); k.add(25000.0, 24750.0, 0);
    CHECK(NEAR(k.score(), 0.0, 1e-12));
}

static void test_judge_share_stays_a_proportion()
{
    current_suite = "judge/share";
    // pass_share_new is fed v2 scores (~1e-3) now; it only uses sign and ratio, so the scale
    // change must not move it.
    CHECK(NEAR(pass_share_new(0.0004, 0.0012), pass_share_new(0.04, 0.12), 1e-12));
    CHECK(NEAR(pass_share_new(-0.0004, -0.0012), pass_share_new(-0.04, -0.12), 1e-12));
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
    test_judge_constant_return_scores_that_return();
    test_judge_reset_day_scores_zero_not_a_jump();
    test_judge_recent_days_count_more();
    test_judge_effective_days();
    test_judge_ignores_invalid_days();
    test_judge_share_stays_a_proportion();

    printf("\n===========================\n");
    printf("%d passed, %d failed\n", pass_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
