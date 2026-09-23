// mt1_grad.h — backward passes for the EXISTING MT1 architectures, plus the per-stock model.
//
// Why backprop through mt1_pool.h's own layouts instead of a generic MLP: the point of the
// evolutionary-vs-gradient race is to isolate the SEARCH METHOD, so the architecture has to be
// held constant across the two. Substituting a plain MLP on the gradient side would confound the
// comparison, and it would make the weights non-comparable element-wise -- which is the whole
// reason for keeping the final model of every pass.
//
// So every backward here mirrors a forward in mt1_pool.h exactly: same layer shapes, same ReLU,
// same flat (out x in) row-major weight ordering followed by the bias vector. A gradient-trained
// net and a mutation-trained net of the same candidate differ ONLY in how their weights were
// found, and `W[k]` means the same thing in both.
//
// Every backward is pinned against central finite differences in tests/test_mt1_grad.cpp.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mt1_pool.h"

// ── one ReLU dense layer ─────────────────────────────────────────────────────────
//
// forward:  out[o] = relu(B[o] + sum_i W[o*n_in + i] * x[i])
//
// The ReLU mask is recovered from the OUTPUT (out[o] > 0), not from a cached pre-activation:
// relu is zero exactly where the pre-activation was non-positive, so the post-activation carries
// the mask already. One less array to keep in sync.
static inline void mt1_dense_relu_bwd(const float* W, const float* x, const float* out,
                                      const float* d_out, float* dW, float* dB, float* d_x,
                                      int n_out, int n_in)
{
    if (d_x) for (int i = 0; i < n_in; i++) d_x[i] = 0.f;
    for (int o = 0; o < n_out; o++) {
        if (!(out[o] > 0.f)) continue;              // relu gate: no gradient flows
        const float g = d_out[o];
        dB[o] += g;
        float* drow = dW + (size_t)o * n_in;
        const float* wrow = W + (size_t)o * n_in;
        for (int i = 0; i < n_in; i++) {
            drow[i] += g * x[i];
            if (d_x) d_x[i] += g * wrow[i];
        }
    }
}

// ── MT1Net (74 -> 1, 3501 params) ────────────────────────────────────────────────

struct MT1NetCache {
    float in74[74];
    float t_a1[2][20], t_a2[2][20], t_b1[2][6], t_b2[2][4], t_c1[2][5], t_c2[2][4];
    float concat56[56], d1[22], d2_in[23], d2[10];
    float extra = 0.f;
    float out = 0.f;
};

// Mirrors mt1net_trunk, keeping every intermediate for the backward pass.
static inline void mt1net_trunk_fwd(const float* W, int base, const float* in37,
                                    MT1NetCache& c, int t, float* out28)
{
    const float* xb = in37;        // daily    [0:10]
    const float* xc = in37 + 10;   // decade   [10:17]
    const float* xa = in37 + 17;   // vol+poly [17:37]
    mt1net_matvec_relu(W + base + NT_M_A1_W, W + base + NT_M_A1_B, xa, c.t_a1[t], 20, 20);
    mt1net_matvec_relu(W + base + NT_M_A2_W, W + base + NT_M_A2_B, c.t_a1[t], c.t_a2[t], 20, 20);
    mt1net_matvec_relu(W + base + NT_M_B1_W, W + base + NT_M_B1_B, xb, c.t_b1[t],  6, 10);
    mt1net_matvec_relu(W + base + NT_M_B2_W, W + base + NT_M_B2_B, c.t_b1[t], c.t_b2[t],  4,  6);
    mt1net_matvec_relu(W + base + NT_M_C1_W, W + base + NT_M_C1_B, xc, c.t_c1[t],  5,  7);
    mt1net_matvec_relu(W + base + NT_M_C2_W, W + base + NT_M_C2_B, c.t_c1[t], c.t_c2[t],  4,  5);
    for (int i = 0; i < 20; i++) out28[i]      = c.t_a2[t][i];
    for (int i = 0; i < 4;  i++) out28[20 + i] = c.t_b2[t][i];
    for (int i = 0; i < 4;  i++) out28[24 + i] = c.t_c2[t][i];
}

static inline float mt1net_forward_cached(const float* W, const float* in74, float extra,
                                          MT1NetCache& c)
{
    memcpy(c.in74, in74, 74 * sizeof(float));
    c.extra = extra;
    mt1net_trunk_fwd(W, 0,         in74,      c, 0, c.concat56);
    mt1net_trunk_fwd(W, NT_P_A1_W, in74 + 37, c, 1, c.concat56 + 28);
    mt1net_matvec_relu(W + NT_D1_W, W + NT_D1_B, c.concat56, c.d1, 22, 56);
    for (int i = 0; i < 22; i++) c.d2_in[i] = c.d1[i];
    c.d2_in[MT1NET_D2_RESERVED] = extra;
    mt1net_matvec_relu(W + NT_D2_W, W + NT_D2_B, c.d2_in, c.d2, 10, 23);
    float acc = W[NT_D3_B];
    for (int i = 0; i < 10; i++) acc += W[NT_D3_W + i] * c.d2[i];
    c.out = acc;
    return acc;
}

static inline void mt1net_trunk_bwd(const float* W, int base, const MT1NetCache& c, int t,
                                    const float* d_out28, float* g)
{
    float d_a2[20], d_b2[4], d_c2[4];
    for (int i = 0; i < 20; i++) d_a2[i] = d_out28[i];
    for (int i = 0; i < 4;  i++) d_b2[i] = d_out28[20 + i];
    for (int i = 0; i < 4;  i++) d_c2[i] = d_out28[24 + i];
    float d_a1[20], d_b1[6], d_c1[5];
    mt1_dense_relu_bwd(W + base + NT_M_A2_W, c.t_a1[t], c.t_a2[t], d_a2,
                       g + base + NT_M_A2_W, g + base + NT_M_A2_B, d_a1, 20, 20);
    mt1_dense_relu_bwd(W + base + NT_M_B2_W, c.t_b1[t], c.t_b2[t], d_b2,
                       g + base + NT_M_B2_W, g + base + NT_M_B2_B, d_b1, 4, 6);
    mt1_dense_relu_bwd(W + base + NT_M_C2_W, c.t_c1[t], c.t_c2[t], d_c2,
                       g + base + NT_M_C2_W, g + base + NT_M_C2_B, d_c1, 4, 5);
    // first layers read the input directly; nothing downstream needs d_x, so pass nullptr
    const float* in37 = c.in74 + (t == 0 ? 0 : 37);
    mt1_dense_relu_bwd(W + base + NT_M_A1_W, in37 + 17, c.t_a1[t], d_a1,
                       g + base + NT_M_A1_W, g + base + NT_M_A1_B, nullptr, 20, 20);
    mt1_dense_relu_bwd(W + base + NT_M_B1_W, in37,      c.t_b1[t], d_b1,
                       g + base + NT_M_B1_W, g + base + NT_M_B1_B, nullptr, 6, 10);
    mt1_dense_relu_bwd(W + base + NT_M_C1_W, in37 + 10, c.t_c1[t], d_c1,
                       g + base + NT_M_C1_W, g + base + NT_M_C1_B, nullptr, 5, 7);
}

// Accumulates dL/dW into g for a given dL/d(raw output). g must be zeroed by the caller.
static inline void mt1net_backward(const float* W, const MT1NetCache& c, float d_out, float* g)
{
    float d_d2[10];
    g[NT_D3_B] += d_out;
    for (int i = 0; i < 10; i++) {
        g[NT_D3_W + i] += d_out * c.d2[i];
        d_d2[i] = d_out * W[NT_D3_W + i];
    }
    float d_d2_in[23];
    mt1_dense_relu_bwd(W + NT_D2_W, c.d2_in, c.d2, d_d2,
                       g + NT_D2_W, g + NT_D2_B, d_d2_in, 10, 23);
    // the reserved slot is a constant input, so its gradient is discarded
    float d_d1[22];
    for (int i = 0; i < 22; i++) d_d1[i] = d_d2_in[i];
    float d_concat[56];
    mt1_dense_relu_bwd(W + NT_D1_W, c.concat56, c.d1, d_d1,
                       g + NT_D1_W, g + NT_D1_B, d_concat, 22, 56);
    mt1net_trunk_bwd(W, 0,         c, 0, d_concat,      g);
    mt1net_trunk_bwd(W, NT_P_A1_W, c, 1, d_concat + 28, g);
}

// ── MT1CNet (146 -> 1, 11177 params) ─────────────────────────────────────────────

struct MT1CNetCache {
    float in[MT1C_IN];
    float a[64], b[24], c[8];
    float out = 0.f;
};

static inline float mt1cnet_forward_cached(const float* W, const float* in, MT1CNetCache& k)
{
    memcpy(k.in, in, MT1C_IN * sizeof(float));
    mt1net_matvec_relu(W + CN_L1_W, W + CN_L1_B, in,  k.a, 64, MT1C_IN);
    mt1net_matvec_relu(W + CN_L2_W, W + CN_L2_B, k.a, k.b, 24, 64);
    mt1net_matvec_relu(W + CN_L3_W, W + CN_L3_B, k.b, k.c,  8, 24);
    float acc = W[CN_L4_B];
    for (int i = 0; i < 8; i++) acc += W[CN_L4_W + i] * k.c[i];
    k.out = acc;
    return acc;
}

static inline void mt1cnet_backward(const float* W, const MT1CNetCache& k, float d_out, float* g)
{
    float d_c[8];
    g[CN_L4_B] += d_out;
    for (int i = 0; i < 8; i++) {
        g[CN_L4_W + i] += d_out * k.c[i];
        d_c[i] = d_out * W[CN_L4_W + i];
    }
    float d_b[24], d_a[64];
    mt1_dense_relu_bwd(W + CN_L3_W, k.b, k.c, d_c, g + CN_L3_W, g + CN_L3_B, d_b, 8, 24);
    mt1_dense_relu_bwd(W + CN_L2_W, k.a, k.b, d_b, g + CN_L2_W, g + CN_L2_B, d_a, 24, 64);
    mt1_dense_relu_bwd(W + CN_L1_W, k.in, k.a, d_a, g + CN_L1_W, g + CN_L1_B, nullptr,
                       64, MT1C_IN);
}

// ── MT1S — the per-stock model (385 params) ──────────────────────────────────────
//
// MT1CNet reads all 12 symbols at once through a 64x146 first layer, so it has to learn the same
// "what does a bar plus an order look like" mapping twelve times over, once per input block, and
// it learns a fixed ordering of the symbols that is in fact arbitrary. MT1S applies ONE shared
// encoder to each symbol in turn and sums the twelve outputs.
//
// Two consequences, and they are the reason this is a candidate:
//   - 385 parameters against MT1CNet's 11,177. The race showed the smallest net winning, so the
//     binding constraint is data, not capacity.
//   - it is permutation-invariant across symbols by construction, which is correct: nothing about
//     an industry depends on which slot a ticker happens to occupy.
//
// Each symbol sees its own 12 features plus the two industry-level context values (cash and book),
// because "buy 100 shares" means something different against a $25k book than a $45k one.
//
// The sum is the industry prediction, so MT1S is trainable two ways and BOTH are raced:
//   MT1S-sum    one loss per industry-day on the summed output    -- architecture benefit only
//   MT1S-stock  one loss per SYMBOL-day on that symbol's own P&L  -- 12x the rows
// They share this code; only the target differs.
static constexpr int MT1S_PER_SYM = MT1C_PER_SYM + 2;         // 12 features + cash + book
static constexpr int MT1S_H1 = 16, MT1S_H2 = 8;
static constexpr int SS_L1_W = 0,
                     SS_L1_B = MT1S_H1 * MT1S_PER_SYM,                    // 224
                     SS_L2_W = SS_L1_B + MT1S_H1,                         // 240
                     SS_L2_B = SS_L2_W + MT1S_H2 * MT1S_H1,               // 368
                     SS_L3_W = SS_L2_B + MT1S_H2,                         // 376
                     SS_L3_B = SS_L3_W + MT1S_H2;                         // 384
static constexpr int MT1SNET_PARAMS = SS_L3_B + 1;                        // 385

struct MT1SCache {
    float in[MT1C_SYMS][MT1S_PER_SYM];
    float h1[MT1C_SYMS][MT1S_H1], h2[MT1C_SYMS][MT1S_H2];
    float per_sym[MT1C_SYMS];      // this symbol's own contribution, in raw units
    float out = 0.f;               // their sum = the industry prediction
};

// Builds one symbol's slice from an MT1C input vector, so both candidates read the SAME numbers
// and a difference between them is a difference in architecture, not in features.
static inline void mt1s_slice(const float* in146, int j, float* out /*MT1S_PER_SYM*/)
{
    memcpy(out, in146 + (size_t)j * MT1C_PER_SYM, MT1C_PER_SYM * sizeof(float));
    out[MT1C_PER_SYM]     = in146[CN_CASH];
    out[MT1C_PER_SYM + 1] = in146[CN_BOOK];
}

static inline float mt1s_forward_cached(const float* W, const float* in146, MT1SCache& k)
{
    float total = 0.f;
    for (int j = 0; j < MT1C_SYMS; j++) {
        mt1s_slice(in146, j, k.in[j]);
        mt1net_matvec_relu(W + SS_L1_W, W + SS_L1_B, k.in[j], k.h1[j], MT1S_H1, MT1S_PER_SYM);
        mt1net_matvec_relu(W + SS_L2_W, W + SS_L2_B, k.h1[j], k.h2[j], MT1S_H2, MT1S_H1);
        float acc = W[SS_L3_B];
        for (int i = 0; i < MT1S_H2; i++) acc += W[SS_L3_W + i] * k.h2[j][i];
        k.per_sym[j] = acc;
        total += acc;
    }
    k.out = total;
    return total;
}

static inline float mt1s_forward(const float* W, const float* in146)
{
    MT1SCache k; return mt1s_forward_cached(W, in146, k);
}

// d_per_sym[j] is dL/d(that symbol's output). For MT1S-sum every entry is the same dL/d(total),
// since the industry prediction is their sum; for MT1S-stock each symbol carries its own.
// The shared encoder means all twelve accumulate into the SAME weights -- that is what makes one
// industry-day worth twelve rows.
static inline void mt1s_backward(const float* W, const MT1SCache& k, const float* d_per_sym,
                                 float* g)
{
    for (int j = 0; j < MT1C_SYMS; j++) {
        const float d_out = d_per_sym[j];
        if (d_out == 0.f) continue;
        float d_h2[MT1S_H2];
        g[SS_L3_B] += d_out;
        for (int i = 0; i < MT1S_H2; i++) {
            g[SS_L3_W + i] += d_out * k.h2[j][i];
            d_h2[i] = d_out * W[SS_L3_W + i];
        }
        float d_h1[MT1S_H1];
        mt1_dense_relu_bwd(W + SS_L2_W, k.h1[j], k.h2[j], d_h2, g + SS_L2_W, g + SS_L2_B,
                           d_h1, MT1S_H2, MT1S_H1);
        mt1_dense_relu_bwd(W + SS_L1_W, k.in[j], k.h1[j], d_h1, g + SS_L1_W, g + SS_L1_B,
                           nullptr, MT1S_H1, MT1S_PER_SYM);
    }
}
