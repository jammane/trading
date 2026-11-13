// jnl_score / jnl_composite (jnl_score.h): the ORDER entries' daily score.
//
// Hand-worked cases for each level, both ends, the tie rules, and the property the composite
// exists for: ranking by the register mean is lexicographic (primary, then secondary, then
// tertiary) over a full 16-day register.
#include <cmath>
#include <cstdio>
#include <random>

#include "jnl_score.h"

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; }
}
static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

int main() {
    const int N = 12;
    float real[N], pred[N];
    for (int i = 0; i < N; i++) real[i] = (float)i;   // industry 11 best, 0 worst

    // Perfect order: 4 of 4 at either end, all 66 pairs, correlation 1.
    for (int i = 0; i < N; i++) pred[i] = 10.f * i + 3.f;
    JNLScore s = jnl_score(pred, real, N, true);
    check(near(s.primary, 4) && near(s.secondary, 66) && near(s.tertiary, 1), "perfect, top");
    s = jnl_score(pred, real, N, false);
    check(near(s.primary, 4) && near(s.secondary, 66), "perfect, bottom");

    // Reversed: 0 of 4, 0 pairs, correlation -1.
    for (int i = 0; i < N; i++) pred[i] = (float)(N - i);
    s = jnl_score(pred, real, N, true);
    check(near(s.primary, 0) && near(s.secondary, 0) && near(s.tertiary, -1), "reversed");

    // Top right, bottom wrong: the two ends score differently, the pairs do not.
    // pred: 8..11 on top in order; 0..7 reversed below them.
    for (int i = 0; i < 8; i++) pred[i] = (float)(7 - i);
    for (int i = 8; i < N; i++) pred[i] = (float)(i + 100);
    const JNLScore top = jnl_score(pred, real, N, true), bot = jnl_score(pred, real, N, false);
    check(near(top.primary, 4), "top end right");
    // bottom 4 realised = {0,1,2,3}; predicted bottom 4 = {4,5,6,7}: none
    check(near(bot.primary, 0), "bottom end wrong");
    check(near(top.secondary, bot.secondary), "secondary is direction-free");
    // pairs: within 8..11 all 6 hold, 8..11 vs 0..7 all 32 hold, within 0..7 all 28 fail
    check(near(top.secondary, 38), "secondary hand count");

    // Three of the top 4.
    for (int i = 0; i < N; i++) pred[i] = (float)i;
    pred[8] = -1.f;   // realised 4th best pushed to the bottom; 7 moves into the predicted top 4
    s = jnl_score(pred, real, N, true);
    check(near(s.primary, 3), "three of four");

    // Constant output: the random-guess values, not a universe-order answer.
    for (int i = 0; i < N; i++) pred[i] = 0.f;
    s = jnl_score(pred, real, N, true);
    check(near(s.primary, 4.0 * 4.0 / 12.0), "all tied: primary = 16/12");
    check(near(s.secondary, 33), "all tied: secondary = 33");
    check(near(s.tertiary, 0), "all tied: tertiary 0");

    // A tie straddling 4th place: top 3 right, then two tied for 4th, one of which is right.
    for (int i = 0; i < N; i++) pred[i] = (float)i;
    pred[8] = 7.5f; pred[7] = 7.5f;   // 11,10,9 then {8,7} tied
    s = jnl_score(pred, real, N, true);
    check(near(s.primary, 3.5), "tie across 4th place counts half");

    // Lexicographic composite over a 16-day register: one more primary point on one day beats
    // the best possible secondary and tertiary on every day; one more secondary half-pair on
    // one day beats the best possible tertiary on every day.
    {
        auto mean16 = [](JNLScore a, JNLScore last) {
            double m = 0;
            for (int d = 0; d < 15; d++) m += jnl_composite(a);
            m += jnl_composite(last);
            return m / 16;
        };
        JNLScore lo{2, 66, 1}, hi_last{3, 0, -1}, hi{2, 0, -1};
        check(mean16(hi, hi_last) > mean16(lo, lo), "primary dominates secondary and tertiary");
        JNLScore a{2, 30, 1}, b{2, 30, -1}, b_last{2, 30.5, -1};
        check(mean16(b, b_last) > mean16(a, a), "secondary dominates tertiary");
        JNLScore c{2, 30, 0.2}, d{2, 30, 0.1};
        check(mean16(c, c) > mean16(d, d), "tertiary breaks the remaining tie");
    }

    // Random checks against a brute-force pair count and a permutation-invariance property.
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.f, 1.f);
    for (int t = 0; t < 2000; t++) {
        for (int i = 0; i < N; i++) { real[i] = g(rng); pred[i] = g(rng); }
        const JNLScore a = jnl_score(pred, real, N, true);
        int conc = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (i < j && ((pred[i] - pred[j]) > 0) == ((real[i] - real[j]) > 0)) conc++;
        if (!near(a.secondary, conc)) { check(false, "secondary vs brute force"); break; }
        if (a.primary < 0 || a.primary > 4 || a.tertiary < -1 - 1e-9 || a.tertiary > 1 + 1e-9) {
            check(false, "ranges"); break;
        }
        // Negating both sides swaps the ends.
        float np[N], nr[N];
        for (int i = 0; i < N; i++) { np[i] = -pred[i]; nr[i] = -real[i]; }
        if (!near(jnl_score(np, nr, N, false).primary, a.primary)) {
            check(false, "bottom end of the negation = top end"); break;
        }
    }

    if (failures) { printf("%d failure(s)\n", failures); return 1; }
    printf("test_jnl_score: all checks passed\n");
    return 0;
}
