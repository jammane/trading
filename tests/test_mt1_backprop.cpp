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

    printf(failures ? "\n%d FAILURE(S)\n" : "\nbackprop verified\n", failures);
    return failures ? 1 : 0;
}
