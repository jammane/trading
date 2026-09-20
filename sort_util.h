#pragma once
// Sorting the first `n` entries of a small fixed-size index buffer.
//
// Extracted into a header so it can be tested against std::sort directly (tests/test_sort_util.cpp).
// It is used for MT2 tier assignment, where a silently wrong ordering would misallocate capital
// without crashing -- the failure mode this repo keeps meeting.
//
// Why not std::sort:
//   Every call site provably has n <= N (a counter incremented at most once per industry), but the
//   optimiser cannot see it. At -O3 GCC inlines introsort, including its HEAP fallback, which is
//   unreachable for n <= 16 yet cannot be bounded by value-range propagation. GCC then reports
//       "array subscript [15, 2305843009213693949] is outside array bounds of int [12]"
//   four to six times in step_mt2 and drift_score_mt2_pts. Clamping n was not enough; the heap
//   path survived it.
//
//   Blanket -Wno-array-bounds was rejected: it would equally hide a REAL out-of-bounds in
//   training_v4.cpp, and warnings on that target are new as of v0.8.1.8 precisely because none had
//   ever run there.
//
// Insertion sort is also simply the right algorithm here: n <= 12, and std::sort's own introsort
// degenerates to insertion sort below its _S_threshold of 16 anyway.
template <int N, typename Cmp>
inline void sort_index_prefix(int (&buf)[N], int n, Cmp cmp) {
    if (n < 0) n = 0;
    if (n > N) n = N;
    for (int i = 1; i < n; i++) {
        const int v = buf[i];
        int j = i - 1;
        while (j >= 0 && cmp(v, buf[j])) { buf[j + 1] = buf[j]; j--; }
        buf[j + 1] = v;
    }
}
