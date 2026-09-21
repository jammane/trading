#pragma once
// Gradient training for MT1, hand-coded because the trainer links no autodiff library.
//
// WHY THIS EXISTS. Every MT1 to date was trained by mutation and selection, which feeds back one
// scalar per generation ("this mutant beat its parent") to adjust thousands of weights. Measured
// offline on identical inputs and target, the same MT1CNet architecture scored +0.0033 evolved and
// +0.0077 to +0.0161 gradient-fitted -- the search method, not the architecture, was the binding
// constraint. StockNN keeps evolution because its objective runs through fill simulation, whole-
// share rounding and limit matching and is genuinely non-differentiable. MT1 emits a number and a
// known number arrives the next day; there was never a reason to evolve it.
//
// LAYOUT matches models.py so weights stay portable through convert_weights.py: each layer's
// (out x in) matrix in row-major order, then its bias. Activation is tanh on hidden layers, linear
// on the output; the caller applies tanh(out) * MT1_PRED_SCALE as before.
//
// Adam with weight decay. The target is close to pure noise (mean/sd ~ 0.02), so an unregularised
// fit memorises rather than learns -- offline, the 11,177-parameter shape scored WORSE than a
// 4,993-parameter one on the same data. Capacity is a liability here, not an asset.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

struct MT1Backprop {
    std::vector<int>   dims;      // e.g. {146, 32, 8, 1}
    std::vector<float> W;         // flat weights, models.py order
    std::vector<float> m, v;      // Adam moments, same shape as W
    float lr    = 3e-3f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float decay = 1e-3f;
    long  step  = 0;

    // scratch, sized on init so a training step allocates nothing
    std::vector<std::vector<float>> act;   // act[0] = input, act[k] = layer k output
    std::vector<std::vector<float>> del;   // per-layer error
    std::vector<float> grad;               // same shape as W; exposed so it can be gradient-checked

    int n_params() const {
        int n = 0;
        for (size_t k = 0; k + 1 < dims.size(); k++) n += dims[k] * dims[k+1] + dims[k+1];
        return n;
    }

    void init(const std::vector<int>& d, uint64_t seed) {
        dims = d;
        W.assign(n_params(), 0.f);
        m.assign(W.size(), 0.f);
        v.assign(W.size(), 0.f);
        step = 0;
        act.resize(dims.size());
        del.resize(dims.size());
        grad.assign(W.size(), 0.f);
        for (size_t k = 0; k < dims.size(); k++) { act[k].assign(dims[k], 0.f); del[k].assign(dims[k], 0.f); }
        // Xavier/Glorot uniform, the same scale models.py's nn.Linear defaults to.
        uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
        auto nextf = [&s]() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            return (float)((s >> 11) * (1.0 / 9007199254740992.0));   // [0,1)
        };
        int off = 0;
        for (size_t k = 0; k + 1 < dims.size(); k++) {
            const int fin = dims[k], fout = dims[k+1];
            const float lim = std::sqrt(6.0f / (float)(fin + fout));
            for (int i = 0; i < fin * fout; i++) W[off + i] = (2.f * nextf() - 1.f) * lim;
            off += fin * fout;
            for (int i = 0; i < fout; i++) W[off + i] = 0.f;      // biases start at zero
            off += fout;
        }
    }

    // Forward pass. Returns the scalar output; activations are kept for the backward pass.
    float forward(const float* in) {
        std::memcpy(act[0].data(), in, dims[0] * sizeof(float));
        int off = 0;
        for (size_t k = 0; k + 1 < dims.size(); k++) {
            const int fin = dims[k], fout = dims[k+1];
            const float* Wl = W.data() + off;
            const float* bl = Wl + fin * fout;
            const bool last = (k + 2 == dims.size());
            for (int o = 0; o < fout; o++) {
                float z = bl[o];
                const float* row = Wl + (size_t)o * fin;
                for (int i = 0; i < fin; i++) z += row[i] * act[k][i];
                act[k+1][o] = last ? z : std::tanh(z);
            }
            off += fin * fout + fout;
        }
        return act.back()[0];
    }

    // Backward pass: fills `grad` with d(loss)/d(W) for the input most recently forwarded.
    // Separated from the update so it can be checked against numerical differentiation -- a
    // backward pass that compiles but is subtly wrong produces plausible numbers forever.
    float backward(float target, float weight = 1.f) {
        const float pred = act.back()[0];
        const float err  = pred - target;
        const size_t L = dims.size();
        del[L-1][0] = 2.f * err * weight;                 // linear output
        for (size_t k = L - 1; k >= 2; k--) {
            int off = 0;
            for (size_t q = 0; q + 1 < k; q++) off += dims[q] * dims[q+1] + dims[q+1];
            const int fin = dims[k-1], fout = dims[k];
            const float* Wl = W.data() + off;
            for (int i = 0; i < fin; i++) {
                float g = 0.f;
                for (int o = 0; o < fout; o++) g += Wl[(size_t)o * fin + i] * del[k][o];
                const float a = act[k-1][i];
                del[k-1][i] = g * (1.f - a * a);          // tanh'
            }
        }
        int off = 0;
        for (size_t k = 0; k + 1 < L; k++) {
            const int fin = dims[k], fout = dims[k+1];
            for (int o = 0; o < fout; o++) {
                const float d = del[k+1][o];
                for (int i = 0; i < fin; i++)
                    grad[off + (size_t)o * fin + i] = d * act[k][i];
                grad[off + (size_t)fin * fout + o] = d;
            }
            off += fin * fout + fout;
        }
        return err * err * weight;
    }

    // Adam update from whatever is currently in `grad`, plus decoupled weight decay.
    void adam_step() {
        step++;
        const float b1t = 1.f - std::pow(beta1, (float)step);
        const float b2t = 1.f - std::pow(beta2, (float)step);
        for (size_t i = 0; i < W.size(); i++) {
            const float g = grad[i] + decay * W[i];
            m[i] = beta1 * m[i] + (1.f - beta1) * g;
            v[i] = beta2 * v[i] + (1.f - beta2) * g * g;
            W[i] -= lr * (m[i] / b1t) / (std::sqrt(v[i] / b2t) + eps);
        }
    }

    float train_step(float target, float weight = 1.f) {
        const float loss = backward(target, weight);
        adam_step();
        return loss;
    }

    float loss_for(const float* in, float target) {
        const float p = forward(in);
        return (p - target) * (p - target);
    }
};
