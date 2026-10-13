// Floating-point guards must actually run.
//
// The trainer compiles with -ffast-math, which implies -ffinite-math-only, under which GCC is
// free to fold std::isfinite()/std::isnan() to compile-time constants. That is exactly what it
// did, and it silently disabled EVERY finiteness guard in training_v4.cpp -- seven call sites,
// including the OHLCV bar validity check and cn_ok(), MT1CNet's input sanitiser. The visible
// symptom was oi_sell_aggr arriving NaN on 98.55% of industry-days in the v0.8.1.7 collection
// run, while its buy-side twin was clean because that branch also carried a `> 0.f` comparison
// (comparisons still reject NaN; only the isfinite() call was folded).
//
// This file is deliberately built with the SAME flags as the trainer. If -fno-finite-math-only
// is ever dropped from CMakeLists.txt, these assertions fail and the build stops -- which is the
// only reliable way to notice, because the failure mode is silent wrong numbers, not a crash.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; }
    else     { printf("ok  : %s\n", what); }
}

// Runtime-produced NaN/Inf. `volatile` stops constant folding, so these reach the guards the
// way a bad network output or a malformed bar would.
static volatile float vzero = 0.0f;
static volatile float vbig  = 3.0e38f;

int main() {
    const float nan_v = vzero / vzero;
    const float inf_v = vbig * vbig;

    // Sanity: the values really are what we think, tested by bit pattern, which no optimisation
    // flag can reinterpret.
    uint32_t un, ui;
    std::memcpy(&un, (const void*)&nan_v, 4);
    std::memcpy(&ui, (const void*)&inf_v, 4);
    check((un & 0x7F800000u) == 0x7F800000u && (un & 0x007FFFFFu) != 0, "runtime NaN has NaN bits");
    check((ui & 0x7F800000u) == 0x7F800000u && (ui & 0x007FFFFFu) == 0, "runtime Inf has Inf bits");

    // The guards themselves.
    check(!std::isfinite(nan_v), "isfinite(NaN) is false");
    check(!std::isfinite(inf_v), "isfinite(Inf) is false");
    check( std::isnan(nan_v),    "isnan(NaN) is true");
    check(!std::isnan(1.5f),     "isnan(finite) is false");

    // cn_ok()'s exact shape: the sanitiser MT1CNet's 146 inputs are built through.
    auto cn_ok = [](float v, float fallback) { return std::isfinite(v) ? v : fallback; };
    check(cn_ok(nan_v, 0.5f) == 0.5f, "cn_ok substitutes the fallback for NaN");
    check(cn_ok(inf_v, 0.5f) == 0.5f, "cn_ok substitutes the fallback for Inf");
    check(cn_ok(2.25f, 0.5f) == 2.25f, "cn_ok passes a finite value through");

    // cn_clamp() cannot rescue a NaN on its own -- both comparisons are false, so it RETURNS the
    // NaN. This is why cn_ok must work: the clamp is not a second line of defence.
    auto raw_clamp = [](float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); };
    check(std::isnan(raw_clamp(nan_v, 0.f, 1.f)), "a bare clamp passes NaN through (cn_ok is load-bearing)");

    // The accumulator pattern from the order-intent block: a guarded += must not poison a sum.
    double acc = 0.0;
    if (std::isfinite(nan_v)) acc += (double)nan_v;
    acc += 1.0;
    check(acc == 1.0 && !std::isnan(acc), "guarded accumulator stays finite against a NaN input");

    // safe_int()'s shape (training_v4.cpp:994) -- casting NaN to int is undefined behaviour.
    auto safe_int = [](float v) { return std::isfinite(v) ? (int)v : 0; };
    check(safe_int(nan_v) == 0, "safe_int returns 0 for NaN");
    check(safe_int(inf_v) == 0, "safe_int returns 0 for Inf");
    check(safe_int(-3.7f) == -3, "safe_int truncates a finite value");

    // OHLCV validity (training_v4.cpp:1053) -- a bar with any non-finite leg must be invalid.
    auto bar_valid = [](float o, float h, float l, float c) {
        return std::isfinite(o) && std::isfinite(h) && std::isfinite(l) && std::isfinite(c);
    };
    check(!bar_valid(1.f, nan_v, 1.f, 1.f), "a bar with a NaN high is invalid");
    check(!bar_valid(1.f, 1.f, 1.f, inf_v), "a bar with an Inf close is invalid");
    check( bar_valid(1.f, 2.f, 0.5f, 1.5f), "a clean bar is valid");

    // fminf's NaN rule, relied on implicitly by build_mt1c_input: it returns the OTHER operand.
    // Pinned because it is the reason cfeat measured 100% finite even while cn_ok was dead --
    // luck, not design, and a future switch to std::min would turn that luck into NaNs.
    check(fminf(nan_v, 4.f) == 4.f, "fminf(NaN, x) returns x");

    printf(failures ? "\n%d FAILURE(S)\n" : "\nall fp guards live\n", failures);
    return failures ? 1 : 0;
}
