// sort_index_prefix() must order EXACTLY like std::sort would.
//
// It replaced std::sort at five sites in training_v4.cpp to kill a -Warray-bounds false positive
// at -O3. That is a hand-written sort now deciding MT2's tier assignment: a subtly wrong ordering
// would misallocate capital across industries without crashing, logging, or failing any existing
// test. So it is checked against std::sort itself over randomized inputs, including every
// degenerate size and heavy ties.
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "sort_util.h"

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; }
}

template <int N>
static bool matches_std_sort(int buf[N], int n, const std::vector<float>& key) {
    int mine[N], theirs[N];
    for (int i = 0; i < N; i++) { mine[i] = buf[i]; theirs[i] = buf[i]; }
    auto cmp = [&key](int a, int b) { return key[a] < key[b]; };
    sort_index_prefix(mine, n, cmp);
    int m = n < 0 ? 0 : (n > N ? N : n);
    std::stable_sort(theirs, theirs + m, cmp);
    for (int i = 0; i < m; i++) if (mine[i] != theirs[i]) return false;
    for (int i = m; i < N; i++) if (mine[i] != buf[i]) return false;   // tail untouched
    return true;
}

int main() {
    constexpr int N = 12;
    std::mt19937 rng(12345);

    // Distinct keys, every length including 0 and 1.
    for (int n = 0; n <= N; n++) {
        for (int trial = 0; trial < 200; trial++) {
            int buf[N];
            std::vector<float> key(N);
            for (int i = 0; i < N; i++) { buf[i] = i; key[i] = (float)(rng() % 10000) / 100.f; }
            if (!matches_std_sort<N>(buf, n, key)) {
                printf("FAIL: distinct keys, n=%d trial=%d\n", n, trial); failures++; break;
            }
        }
    }
    check(failures == 0, "distinct keys match std::stable_sort at every n");

    // Heavy ties -- where an unstable or mis-shifted insertion sort diverges. MT2 sorts by TIER,
    // an int in 0..3 across 12 industries, so ties are the normal case, not an edge case.
    int before = failures;
    for (int trial = 0; trial < 500; trial++) {
        int buf[N];
        std::vector<float> key(N);
        for (int i = 0; i < N; i++) { buf[i] = i; key[i] = (float)(rng() % 3); }
        if (!matches_std_sort<N>(buf, N, key)) { printf("FAIL: tie trial=%d\n", trial); failures++; break; }
    }
    check(failures == before, "heavy ties match, and ordering is stable");

    // All-equal keys must leave the order untouched.
    {
        int buf[N]; std::vector<float> key(N, 7.f);
        for (int i = 0; i < N; i++) buf[i] = i;
        sort_index_prefix(buf, N, [&key](int a, int b) { return key[a] < key[b]; });
        bool ok = true;
        for (int i = 0; i < N; i++) ok = ok && (buf[i] == i);
        check(ok, "all-equal keys preserve input order");
    }

    // Out-of-range n is clamped, not obeyed: n > N must not read or write past the buffer, and
    // negative n must be a no-op. This is the invariant the call sites rely on.
    {
        int buf[N]; std::vector<float> key(N);
        for (int i = 0; i < N; i++) { buf[i] = N - 1 - i; key[i] = (float)i; }
        int copy[N]; for (int i = 0; i < N; i++) copy[i] = buf[i];
        sort_index_prefix(buf, 9999, [&key](int a, int b) { return key[a] < key[b]; });
        bool sorted = true;
        for (int i = 1; i < N; i++) sorted = sorted && (key[buf[i - 1]] <= key[buf[i]]);
        check(sorted, "n > N is clamped to N and still sorts the whole buffer");

        for (int i = 0; i < N; i++) buf[i] = copy[i];
        sort_index_prefix(buf, -3, [&key](int a, int b) { return key[a] < key[b]; });
        bool same = true;
        for (int i = 0; i < N; i++) same = same && (buf[i] == copy[i]);
        check(same, "negative n is a no-op");
    }

    // Descending comparator -- step_mt2 sorts one buffer the other way round.
    {
        int buf[N]; std::vector<float> key(N);
        for (int i = 0; i < N; i++) { buf[i] = i; key[i] = (float)((i * 7) % 12); }
        sort_index_prefix(buf, N, [&key](int a, int b) { return key[a] > key[b]; });
        bool ok = true;
        for (int i = 1; i < N; i++) ok = ok && (key[buf[i - 1]] >= key[buf[i]]);
        check(ok, "descending comparator orders descending");
    }

    printf(failures ? "\n%d FAILURE(S)\n" : "\nsort_index_prefix matches std::sort\n", failures);
    return failures ? 1 : 0;
}
