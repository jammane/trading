// The hand-coded backward pass must match numerical differentiation.
//
// This is the test that matters. A backward pass with a transposed index, a missing tanh', or an
// off-by-one in the layer offsets still compiles, still runs, still produces a number that falls
// and looks like learning -- and trains a model that is quietly wrong forever. Comparing against
// finite differences is the only way to know the chain rule was applied correctly.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "mt1_backprop.h"

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; } else { printf("ok  : %s\n", what); }
}

int main() {
    // Several shapes, including the deployed 146-wide one and a deliberately deep/narrow case.
    const std::vector<std::vector<int>> shapes = {
        {4, 3, 1}, {8, 5, 3, 1}, {146, 32, 8, 1}, {12, 4, 4, 4, 1}
    };
    for (const auto& d : shapes) {
        MT1Backprop net;
        net.init(d, 12345u + d.size());
        std::vector<float> in(d[0]);
        for (int i = 0; i < d[0]; i++) in[i] = 0.3f * std::sin(0.7f * i) + 0.1f;
        const float target = 0.42f;

        net.forward(in.data());
        net.backward(target);
        std::vector<float> analytic = net.grad;

        // Numerical gradient: central difference on every weight (or a sample, for the big one).
        const float h = 1e-3f;
        double worst = 0.0;
        const size_t nstep = analytic.size() > 600 ? analytic.size() / 200 : 1;
        int compared = 0;
        for (size_t i = 0; i < analytic.size(); i += nstep) {
            const float w0 = net.W[i];
            net.W[i] = w0 + h; const float lp = net.loss_for(in.data(), target);
            net.W[i] = w0 - h; const float lm = net.loss_for(in.data(), target);
            net.W[i] = w0;
            const double num = (lp - lm) / (2.0 * h);
            const double den = std::max(1.0, std::max(std::fabs(num), (double)std::fabs(analytic[i])));
            worst = std::max(worst, std::fabs(num - analytic[i]) / den);
            compared++;
        }
        char msg[160];
        snprintf(msg, sizeof(msg), "gradient matches finite differences, shape %zu layers, "
                 "%d weights checked, worst rel err %.2e", d.size(), compared, worst);
        check(worst < 2e-2, msg);
    }

    // Training must actually reduce the loss on a learnable relationship.
    {
        MT1Backprop net; net.init({3, 8, 1}, 7);
        auto f = [](const float* x) { return 0.5f * x[0] - 0.3f * x[1] + 0.2f * x[0] * x[2]; };
        float first = 0.f, last = 0.f;
        uint64_t s = 99;
        auto rnd = [&s]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                            return (float)((s >> 33) / 4294967296.0) * 2.f - 1.f; };
        for (int it = 0; it < 4000; it++) {
            float x[3] = {rnd(), rnd(), rnd()};
            net.forward(x);
            const float l = net.train_step(f(x));
            if (it < 200) first += l / 200.f;
            if (it >= 3800) last += l / 200.f;
        }
        char msg[120];
        snprintf(msg, sizeof(msg), "loss falls on a learnable target (%.4f -> %.4f)", first, last);
        check(last < first * 0.5f, msg);
    }

    // Weight decay must pull an unused weight toward zero rather than leaving it adrift.
    {
        MT1Backprop net; net.init({2, 2, 1}, 3);
        net.decay = 0.5f; net.lr = 1e-2f;
        const float before = std::fabs(net.W[0]);
        float x[2] = {0.f, 0.f};                       // zero input -> zero data gradient
        for (int i = 0; i < 500; i++) { net.forward(x); net.train_step(0.f); }
        check(std::fabs(net.W[0]) < before, "weight decay shrinks a weight with no data gradient");
    }

    // Parameter count must match the flat layout convert_weights.py expects.
    {
        MT1Backprop net; net.init({146, 32, 8, 1}, 1);
        check(net.n_params() == 146*32 + 32 + 32*8 + 8 + 8*1 + 1, "n_params matches the layout");
        check((int)net.W.size() == net.n_params(), "W is sized to n_params");
    }

    // Weights must round-trip through the flat array convert_weights.py and save_bin read, or the
    // paired MT1 that travels into champion/ comes back as something else.
    {
        MT1Backprop a, b;
        a.init({146, 32, 8, 1}, 4242);
        b.init({146, 32, 8, 1}, 777);
        std::vector<float> in(146);
        for (int i = 0; i < 146; i++) in[i] = 0.2f * std::cos(0.3f * i);
        for (int i = 0; i < 300; i++) { a.forward(in.data()); a.train_step(0.31f); }
        const float want = a.forward(in.data());
        check(std::fabs(b.forward(in.data()) - want) > 1e-6f,
              "a differently-seeded net really does differ before the copy");
        b.W = a.W;                                  // the save/load path is a flat float copy
        check(std::fabs(b.forward(in.data()) - want) < 1e-6f,
              "copying the flat weight vector reproduces the output exactly");
    }

    // Two industries trained on different targets must not converge to the same network -- if they
    // did, the per-industry pairing would be decoration.
    {
        MT1Backprop p, q;
        p.init({6, 8, 1}, 11); q.init({6, 8, 1}, 11);   // SAME seed, so any divergence is training
        uint64_t s2 = 5;
        auto rnd = [&s2]() { s2 = s2 * 6364136223846793005ULL + 1442695040888963407ULL;
                             return (float)((s2 >> 33) / 2147483648.0) * 2.f - 1.f; };
        for (int it = 0; it < 2000; it++) {
            float x[6]; for (int i = 0; i < 6; i++) x[i] = rnd();
            p.forward(x); p.train_step(0.4f * x[0]);
            q.forward(x); q.train_step(-0.4f * x[0]);
        }
        float x[6]; for (int i = 0; i < 6; i++) x[i] = 0.5f;
        check(std::fabs(p.forward(x) - q.forward(x)) > 0.05f,
              "nets trained on opposite targets diverge from an identical seed");
    }

    // ── the race's arbitrary-gradient injection: backward(out - g/2) == g * dOut/dW ───
    {
        MT1Backprop net; net.init({6, 5, 3, 1}, 77);
        float x[6] = {0.2f, -0.7f, 0.4f, 0.9f, -0.1f, 0.3f};
        const float out = net.forward(x);
        net.backward(out - 0.5f);                  // d_out = 1: the plain dOut/dW
        std::vector<float> base = net.grad;
        const float g = -0.37f;
        net.forward(x);
        net.backward(out - 0.5f * g);
        double worst = 0.0;
        for (size_t k = 0; k < base.size(); k++)
            worst = std::fmax(worst, std::fabs(net.grad[k] - g * base[k]));
        check(worst < 1e-6, "backward(out - g/2) injects exactly g as the output gradient");
    }

    // ── train_batch: ONE Adam step on the MEAN gradient ─────────────────────────────
    // n = 1 must be exactly train_step, and the gradient it steps on must be the mean of the
    // per-row gradients -- a sum would quietly scale the step with the day's row count.
    {
        MT1Backprop a, b; a.init({6, 5, 3, 1}, 21); b.init({6, 5, 3, 1}, 21);
        float x[6] = {0.2f, -0.7f, 0.4f, 0.9f, -0.1f, 0.3f};
        const float* rows[1] = {x}; const float t[1] = {0.25f};
        a.forward(x); a.train_step(0.25f);
        b.train_batch(rows, t, 1);
        double worst = 0.0;
        for (size_t k = 0; k < a.W.size(); k++) worst = std::fmax(worst, std::fabs(a.W[k] - b.W[k]));
        check(worst == 0.0, "train_batch over one row is exactly train_step");
    }
    {
        MT1Backprop net; net.init({6, 5, 3, 1}, 33);
        float r0[6] = {0.1f, 0.2f, -0.3f, 0.4f, 0.0f, -0.5f};
        float r1[6] = {-0.6f, 0.3f, 0.2f, -0.1f, 0.8f, 0.1f};
        float r2[6] = {0.5f, -0.4f, 0.7f, 0.2f, -0.2f, 0.6f};
        const float* rows[3] = {r0, r1, r2}; const float t[3] = {0.3f, -0.2f, 0.05f};
        std::vector<double> mean(net.W.size(), 0.0);
        for (int r = 0; r < 3; r++) {
            net.forward(rows[r]); net.backward(t[r]);
            for (size_t k = 0; k < mean.size(); k++) mean[k] += net.grad[k] / 3.0;
        }
        net.train_batch(rows, t, 3);                 // grad is left holding what it stepped on
        double worst = 0.0;
        for (size_t k = 0; k < mean.size(); k++) worst = std::fmax(worst, std::fabs(net.grad[k] - mean[k]));
        check(worst < 1e-6, "train_batch steps on the mean of the per-row gradients");
    }
    // The regression itself. A race-shaped day -- 200 rows sharing most of their O(1) inputs, a
    // demeaned noise target -- trained per-row collapses to a handful of tied predictions; one
    // batched step per day must not. Both arms are run so the test also proves it can see the
    // failure: if per-row stops collapsing, the scenario no longer exercises what it guards.
    {
        auto run = [](bool batched) {
            MT1Backprop net; net.init({146, 32, 8, 1}, 15);
            uint64_t s = 0x5EED;
            auto u = [&s]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                              return (float)((s >> 11) * (1.0 / 9007199254740992.0)); };
            auto gauss = [&u]() { return std::sqrt(-2.f * std::log(u() + 1e-12f)) *
                                         std::cos(6.2831853f * u()); };
            const int R = 200;
            static float rows[200][146]; static float tgt[200]; const float* rp[200];
            float base[146];
            int dist = 0;
            for (int d = 0; d < 150; d++) {
                for (int q = 0; q < 146; q++) base[q] = (q % 12 < 4) ? 1.f + 0.02f * gauss() : 0.3f * gauss();
                double mu = 0.0;
                for (int m = 0; m < R; m++) {
                    for (int q = 0; q < 146; q++)
                        rows[m][q] = base[q] + ((q % 12 >= 8 && u() < 0.33f) ? 0.5f * gauss() : 0.f);
                    tgt[m] = 0.0166f * gauss(); mu += tgt[m]; rp[m] = rows[m];
                }
                for (int m = 0; m < R; m++) tgt[m] -= (float)(mu / R);
                if (batched) net.train_batch(rp, tgt, R);
                else for (int m = 0; m < R; m++) { net.forward(rows[m]); net.train_step(tgt[m]); }
            }
            std::vector<float> p(R);
            for (int m = 0; m < R; m++) p[m] = net.forward(rows[m]);
            std::sort(p.begin(), p.end());
            dist = (int)(std::unique(p.begin(), p.end()) - p.begin());
            return dist;
        };
        const int per_row = run(false), batched = run(true);
        char msg[160];
        snprintf(msg, sizeof(msg), "per-row training collapses on a race-shaped day "
                 "(%d distinct of 200)", per_row);
        check(per_row < 60, msg);
        snprintf(msg, sizeof(msg), "one batched step per day does not (%d distinct of 200)", batched);
        check(batched >= 190, msg);
    }

    printf(failures ? "\n%d FAILURE(S)\n" : "\nbackprop verified\n", failures);
    return failures ? 1 : 0;
}
