// tests/test_mt1_grad.cpp — every backward in mt1_grad.h against central finite differences.
//
// A wrong backward pass does not crash and does not look wrong: it trains to a worse optimum and
// reads as "this architecture is weak". Since the whole point of the race is to compare
// architectures, an unverified gradient would produce a confident wrong ranking. Hence one
// finite-difference check per architecture, over every parameter.
//
// Build:  cmake --build build --target test_mt1_grad_cpp
// Run:    ./build/test_mt1_grad_cpp   (or: ctest --test-dir build)

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>

#include "mt1_grad.h"

static int failures = 0;
static void check(bool ok, const char* msg) {
    printf("%s: %s\n", ok ? "ok  " : "FAIL", msg);
    if (!ok) failures++;
}

static uint64_t rs = 12345;
static float rnd() {
    rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rs >> 33) / 2147483648.0) * 2.f - 1.f;
}

// Central differences over EVERY parameter, SKIPPING ReLU kinks.
//
// A ReLU net is not differentiable where a pre-activation crosses zero. If the +h and -h probes
// land on opposite sides of such a crossing, the central difference averages two different
// one-sided derivatives and reports a number the true gradient was never supposed to match. That
// is a property of the function, not an error in the backward pass -- so each probe also returns
// a hash of which units are active, and the parameter is skipped when the two probes disagree.
//
// Without this the check reports large errors on a CORRECT gradient, which would send you looking
// for a bug that is not there. The teeth test at the end proves skipping has not defanged it.
static constexpr double FD_MIN_GRAD = 2e-3;

struct FdProbe { double val; uint64_t mask; };

static uint64_t fnv_bits(const float* v, int n, uint64_t h) {
    for (int i = 0; i < n; i++) {
        h ^= (uint64_t)(v[i] > 0.f);
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename FwdFn>
static void fd_check(const char* name, FwdFn fwd, std::vector<float>& W,
                     const float* analytic, float tol)
{
    double worst = 0.0; int worst_i = -1; int checked = 0, kinks = 0;
    for (size_t i = 0; i < W.size(); i++) {
        const float w0 = W[i];
        // A LARGE step, deliberately. These nets are piecewise LINEAR, so when the mask check
        // below confirms both probes sit on the same linear piece the central difference is
        // exact -- there is no truncation error to trade against. What a small step does buy is
        // catastrophic cancellation: gradients here run ~1e-4, so at h=1e-3 the difference
        // fp-fm is ~2e-7, which IS float epsilon at an output of magnitude 0.4, and the quotient
        // comes back visibly quantized. That read as a 1.4 relative error on a correct gradient.
        const float h = std::fmax(5e-2f * std::fabs(w0), 5e-2f);
        W[i] = w0 + h; const FdProbe fp = fwd();
        W[i] = w0 - h; const FdProbe fm = fwd();
        W[i] = w0;
        if (fp.mask != fm.mask) { kinks++; continue; }   // straddles a ReLU crossing
        const double num = (fp.val - fm.val) / (2.0 * h);
        const double ana = analytic[i];
        const double den = std::fmax(std::fabs(num), std::fabs(ana));
        // Only check gradients large enough for a FLOAT forward to resolve. The error in the
        // quotient is about |out| * eps * sqrt(2) / 2h; at |out| ~ 2, eps 1.2e-7 and h 0.05 that
        // is ~3.4e-6, so a relative error under 2e-3 is only meaningful for |grad| > ~1.7e-3.
        // Below that the check would be grading float noise -- measured, every residual failure
        // sat at |grad| < 5e-5 while nothing above 1e-3 failed at all.
        if (den < FD_MIN_GRAD) continue;
        const double rel = std::fabs(num - ana) / den;
        checked++;
        if (rel > worst) { worst = rel; worst_i = (int)i; }
        if (getenv("FD_DUMP") && rel > 0.01)
            printf("      p%-6zu num=%+.8f ana=%+.8f rel=%.4f\n", i, num, ana, rel);
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: worst rel err %.3g at param %d (%d probed, %d kinks skipped)",
             name, worst, worst_i, checked, kinks);
    // The coverage floor exists to catch a DEGENERATE probe, not to demand breadth: an early
    // version of this test had a broken RNG that emitted only negative values, which killed every
    // ReLU past the first layer and left the net a constant. It reported "worst rel err 2e-06"
    // off a single probed parameter and looked like a pass. MT1S-sym legitimately probes few
    // parameters -- only one of twelve symbols carries gradient there -- so the floor is low.
    check(worst < tol && checked >= 25, buf);
}

int main()
{
    // ── MT1Net: 74 -> 1, two trunks with three parallel sub-paths each ──────────
    {
        std::vector<float> W(MT1NET_PARAMS), g(MT1NET_PARAMS, 0.f);
        for (auto& w : W) w = 0.35f * rnd();
        float in74[74];
        for (int i = 0; i < 74; i++) in74[i] = rnd();
        const float extra = 0.f;
        MT1NetCache c;
        mt1net_forward_cached(W.data(), in74, extra, c);
        mt1net_backward(W.data(), c, 1.0f, g.data());
        fd_check("MT1Net  ", [&]() -> FdProbe {
            MT1NetCache t; const double v = mt1net_forward_cached(W.data(), in74, extra, t);
            uint64_t m = 1469598103934665603ULL;
            for (int q = 0; q < 2; q++) {
                m = fnv_bits(t.t_a1[q], 20, m); m = fnv_bits(t.t_a2[q], 20, m);
                m = fnv_bits(t.t_b1[q],  6, m); m = fnv_bits(t.t_b2[q],  4, m);
                m = fnv_bits(t.t_c1[q],  5, m); m = fnv_bits(t.t_c2[q],  4, m);
            }
            m = fnv_bits(t.d1, 22, m); m = fnv_bits(t.d2, 10, m);
            return {v, m};
        }, W, g.data(), 2e-3f);

        // the cached forward must agree with mt1_pool.h's own forward, or the backward is
        // differentiating a different function than the evolutionary side evaluates
        MT1NetCache c2;
        const float a = mt1net_forward_cached(W.data(), in74, extra, c2);
        const float b = mt1net_forward(W.data(), in74, extra);
        check(std::fabs(a - b) < 1e-5f, "MT1Net cached forward == mt1_pool.h forward");
    }

    // ── MT1CNet: 146 -> 64 -> 24 -> 8 -> 1 ──────────────────────────────────────
    {
        std::vector<float> W(MT1CNET_PARAMS), g(MT1CNET_PARAMS, 0.f);
        for (auto& w : W) w = 0.25f * rnd();
        float in[MT1C_IN];
        for (int i = 0; i < MT1C_IN; i++) in[i] = rnd();
        MT1CNetCache k;
        mt1cnet_forward_cached(W.data(), in, k);
        mt1cnet_backward(W.data(), k, 1.0f, g.data());
        fd_check("MT1CNet ", [&]() -> FdProbe {
            MT1CNetCache t; const double v = mt1cnet_forward_cached(W.data(), in, t);
            uint64_t m = 1469598103934665603ULL;
            m = fnv_bits(t.a, 64, m); m = fnv_bits(t.b, 24, m); m = fnv_bits(t.c, 8, m);
            return {v, m};
        }, W, g.data(), 2e-3f);

        MT1CNetCache k2;
        const float a = mt1cnet_forward_cached(W.data(), in, k2);
        const float b = mt1cnet_forward(W.data(), in, 0.f);
        check(std::fabs(a - b) < 1e-5f, "MT1CNet cached forward == mt1_pool.h forward");
    }

    // ── MT1S summed: one loss on the industry total ─────────────────────────────
    {
        std::vector<float> W(MT1SNET_PARAMS), g(MT1SNET_PARAMS, 0.f);
        for (auto& w : W) w = 0.4f * rnd();
        float in[MT1C_IN];
        for (int i = 0; i < MT1C_IN; i++) in[i] = rnd();
        MT1SCache k;
        mt1s_forward_cached(W.data(), in, k);
        float d[MT1C_SYMS];
        for (int j = 0; j < MT1C_SYMS; j++) d[j] = 1.0f;     // d(total)/d(per_sym) == 1
        mt1s_backward(W.data(), k, d, g.data());
        fd_check("MT1S-sum", [&]() -> FdProbe {
            MT1SCache t; const double v = mt1s_forward_cached(W.data(), in, t);
            uint64_t m = 1469598103934665603ULL;
            for (int j = 0; j < MT1C_SYMS; j++) {
                m = fnv_bits(t.h1[j], MT1S_H1, m); m = fnv_bits(t.h2[j], MT1S_H2, m);
            }
            return {v, m};
        }, W, g.data(), 2e-3f);
    }

    // ── MT1S per-stock: only ONE symbol carries gradient ────────────────────────
    // This is the mode that makes an industry-day worth twelve rows, and it is a different
    // computation from the summed one -- worth its own check rather than assuming it follows.
    {
        std::vector<float> W(MT1SNET_PARAMS), g(MT1SNET_PARAMS, 0.f);
        for (auto& w : W) w = 0.4f * rnd();
        float in[MT1C_IN];
        for (int i = 0; i < MT1C_IN; i++) in[i] = rnd();
        const int target_sym = 5;
        MT1SCache k;
        mt1s_forward_cached(W.data(), in, k);
        float d[MT1C_SYMS] = {};
        d[target_sym] = 1.0f;
        mt1s_backward(W.data(), k, d, g.data());
        fd_check("MT1S-sym", [&]() -> FdProbe {
            MT1SCache t; mt1s_forward_cached(W.data(), in, t);
            uint64_t m = 1469598103934665603ULL;
            for (int j = 0; j < MT1C_SYMS; j++) {
                m = fnv_bits(t.h1[j], MT1S_H1, m); m = fnv_bits(t.h2[j], MT1S_H2, m);
            }
            return {(double)t.per_sym[target_sym], m};
        }, W, g.data(), 2e-3f);
    }

    // ── the shared encoder is genuinely shared ──────────────────────────────────
    {
        std::vector<float> W(MT1SNET_PARAMS);
        for (auto& w : W) w = 0.3f * rnd();
        float in[MT1C_IN] = {};
        in[CN_CASH] = 0.4f; in[CN_BOOK] = 1.1f;
        // give symbols 2 and 7 identical features; their outputs must be identical
        for (int q = 0; q < MT1C_PER_SYM; q++) {
            const float v = rnd();
            in[2 * MT1C_PER_SYM + q] = v;
            in[7 * MT1C_PER_SYM + q] = v;
        }
        MT1SCache k;
        mt1s_forward_cached(W.data(), in, k);
        check(std::fabs(k.per_sym[2] - k.per_sym[7]) < 1e-6f,
              "MT1S: identical symbols give identical outputs (weights really are shared)");
        // and the industry output is the sum of the parts
        float s = 0.f;
        for (int j = 0; j < MT1C_SYMS; j++) s += k.per_sym[j];
        check(std::fabs(s - k.out) < 1e-4f, "MT1S: industry output == sum of symbol outputs");
    }

    // ── a wrong backward must FAIL this harness (the test has teeth) ────────────
    {
        std::vector<float> W(MT1SNET_PARAMS), g(MT1SNET_PARAMS, 0.f);
        for (auto& w : W) w = 0.4f * rnd();
        float in[MT1C_IN];
        for (int i = 0; i < MT1C_IN; i++) in[i] = rnd();
        MT1SCache k;
        mt1s_forward_cached(W.data(), in, k);
        float d[MT1C_SYMS];
        for (int j = 0; j < MT1C_SYMS; j++) d[j] = 1.0f;
        mt1s_backward(W.data(), k, d, g.data());
        for (auto& v : g) v *= 1.5f;                       // deliberately wrong by 50%
        int bad = 0;
        for (size_t i = 0; i < W.size(); i++) {
            const float w0 = W[i], h = std::fmax(1e-3f * std::fabs(w0), 1e-3f);
            W[i] = w0 + h; const double fp = mt1s_forward(W.data(), in);
            W[i] = w0 - h; const double fm = mt1s_forward(W.data(), in);
            W[i] = w0;
            const double num = (fp - fm) / (2.0 * h);
            if (std::fabs(num) > 1e-4 && std::fabs(num - g[i]) / std::fabs(num) > 2e-3) bad++;
        }
        check(bad > 10, "a 1.5x-scaled gradient is rejected (the FD check has teeth)");
    }

    printf(failures ? "\n%d FAILURE(S)\n" : "\nall gradients verified\n", failures);
    return failures ? 1 : 0;
}
