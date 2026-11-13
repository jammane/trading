// jn_bridge.cpp -- see jn_bridge.h. Built with j_neuro's flags (-ffp-contract=off -fno-fast-math),
// never with the trainer's -ffast-math.
#include "jn_bridge.h"

#include <jneuro/jneuro.hpp>
#include <jneuro/modify.hpp>

#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using D   = std::uint32_t;
using JNet = jneuro::Net<D>;

struct JNPool {
    int n_in = 0, n_out = 0, ring_cap = 0;
    uint64_t seed = 0;
    jneuro::Profile profile = jneuro::Profile::defaults();
    std::vector<JNet> nets;
    std::deque<std::vector<double>> ring;     // oldest first; the newest row is today's
    std::vector<double> in;                   // scratch
    // A saved net with mig_from outputs is widened on load: a fresh output (spec/12,
    // add_output_fresh) is inserted at each of mig_at, ascending final indices.
    int mig_from = -1;
    std::vector<int> mig_at;
};

// Widen n to the pool's outputs when it has the registered older width. False if it fits neither.
static bool fit_outputs(const JNPool* p, JNet& n, uint64_t salt) {
    if ((int)n.n_inputs != p->n_in) return false;
    if ((int)n.n_outputs == p->n_out) return true;
    if ((int)n.n_outputs != p->mig_from) return false;
    jneuro::Rng rng(p->seed ^ (salt * 0x9E3779B97F4A7C15ULL));
    for (int at : p->mig_at) n = jneuro::add_output_fresh(n, rng, (std::uint32_t)at);
    return (int)n.n_outputs == p->n_out;
}

void jn_set_output_migration(JNPool* p, int from_n_out, const std::vector<int>& insert_at) {
    p->mig_from = from_n_out;
    p->mig_at = insert_at;
}

static JNet make_net(const jneuro::AnyNet& any) { return std::get<JNet>(any); }

JNPool* jn_pool_create(int n_in, int n_out, int n_slots, int ring_cap,
                       const std::string& profile_path, uint64_t seed, std::string& err) {
    try {
        auto* p = new JNPool;
        p->n_in = n_in; p->n_out = n_out; p->ring_cap = ring_cap; p->seed = seed;
        std::ifstream f(profile_path);
        if (!f) { delete p; err = "cannot read j_neuro profile " + profile_path; return nullptr; }
        std::stringstream ss; ss << f.rdbuf();
        p->profile = jneuro::profile_from_json(ss.str());
        p->nets.reserve(n_slots);
        for (int s = 0; s < n_slots; s++) {
            jneuro::Rng rng(seed ^ (0xD1B54A32D192ED03ULL * (uint64_t)(s + 1)));
            p->nets.push_back(make_net(jneuro::create(n_in, n_out, jneuro::DepthType::U32, 5,
                                                      jneuro::Priority::Speed, rng)));
        }
        p->in.assign(n_in, 0.0);
        return p;
    } catch (const std::exception& e) {
        err = std::string("j_neuro: ") + e.what();
        return nullptr;
    }
}

void jn_pool_free(JNPool* p) { delete p; }
int  jn_n_slots(const JNPool* p) { return (int)p->nets.size(); }

void jn_push_row(JNPool* p, const float* row) {
    p->ring.emplace_back(row, row + p->n_in);
    while ((int)p->ring.size() > p->ring_cap) p->ring.pop_front();
}

void jn_clear_rows(JNPool* p) { p->ring.clear(); }

void jn_replace_row(JNPool* p, const float* row) {
    if (p->ring.empty()) { jn_push_row(p, row); return; }
    p->ring.back().assign(row, row + p->n_in);
}

bool jn_forward(JNPool* p, int slot, float* out) {
    for (int o = 0; o < p->n_out; o++) out[o] = 0.f;
    JNet& net = p->nets[slot];
    const size_t warm = (size_t)net.get_warmup_steps();
    if (p->ring.size() < warm + 1) return false;                // not enough history yet
    net.reset();
    std::vector<double> y;
    for (size_t r = p->ring.size() - warm - 1; r < p->ring.size(); r++) y = net.step(p->ring[r]);
    bool ok = true;
    for (int o = 0; o < p->n_out; o++) {
        const double v = y[o];
        if (!std::isfinite(v)) { ok = false; break; }
        // Clamp to float range: a finite double can still overflow the cast.
        out[o] = (float)std::fmax(-3.0e38, std::fmin(3.0e38, v));
    }
    if (!ok) for (int o = 0; o < p->n_out; o++) out[o] = 0.f;
    return ok;
}

void jn_mutate(JNPool* p, int src, int dst, uint64_t generation, uint64_t seed) {
    jneuro::Rng rng(seed);
    const bool grow = jneuro::is_growth_generation(p->seed, generation, p->profile);
    p->nets[dst] = jneuro::mutate(p->nets[src], p->profile, generation, rng, nullptr, grow);
}

void jn_permute(JNPool* p, const int* src_of_dst, int n) {
    std::vector<JNet> tmp;
    tmp.reserve(n);
    for (int k = 0; k < n; k++) tmp.push_back(p->nets[src_of_dst[k]].copy_genome());
    for (int k = 0; k < n; k++) p->nets[k] = std::move(tmp[k]);
}

void jn_inject(JNPool* p, int slot, double pct, uint64_t seed) {
    jneuro::Rng rng(seed);
    p->nets[slot] = jneuro::inject_diversity(p->nets[slot], pct, rng);
}

JNStats jn_stats(const JNPool* p) {
    JNStats s;
    s.ring_len = (int)p->ring.size();
    if (p->nets.empty()) return s;
    double nodes = 0.0, warm = 0.0;
    for (const auto& n : p->nets) {
        const int nn = (int)n.nodes.size() - (int)n.n_inputs;
        const int w  = (int)n.get_warmup_steps();
        nodes += nn; warm += w;
        if (nn > s.nodes_max) s.nodes_max = nn;
        if (w > s.warm_max) s.warm_max = w;
    }
    s.nodes_mean = nodes / (double)p->nets.size();
    s.warm_mean  = warm  / (double)p->nets.size();
    return s;
}

bool jn_save_slot(const JNPool* p, int slot, const std::string& path) {
    try { jneuro::save_net(path, p->nets[slot]); return true; }
    catch (const std::exception&) { return false; }
}

bool jn_load_slot(JNPool* p, int slot, const std::string& path) {
    try {
        if (slot < 0 || slot >= (int)p->nets.size() || !std::filesystem::exists(path)) return false;
        JNet n = make_net(jneuro::load_net(path));
        if (!fit_outputs(p, n, (uint64_t)slot + 1)) return false;
        p->nets[slot] = std::move(n);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool jn_save(const JNPool* p, const std::string& dir, const std::string& prefix) {
    for (size_t s = 0; s < p->nets.size(); s++)
        if (!jn_save_slot(p, (int)s, dir + "/" + prefix + "_" + std::to_string(s) + ".jnet"))
            return false;
    FILE* f = fopen((dir + "/" + prefix + "_ring.bin").c_str(), "wb");
    if (!f) return false;
    const int32_t hdr[2] = {(int32_t)p->ring.size(), (int32_t)p->n_in};
    bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1;
    for (const auto& r : p->ring) ok = ok && fwrite(r.data(), sizeof(double), r.size(), f) == r.size();
    fclose(f);
    return ok;
}

bool jn_load(JNPool* p, const std::string& dir, const std::string& prefix) {
    try {
        std::vector<JNet> nets;
        for (size_t s = 0; s < p->nets.size(); s++) {
            const std::string path = dir + "/" + prefix + "_" + std::to_string(s) + ".jnet";
            if (!std::filesystem::exists(path)) return false;
            JNet n = make_net(jneuro::load_net(path));
            if (!fit_outputs(p, n, s + 1)) return false;
            nets.push_back(std::move(n));
        }
        std::deque<std::vector<double>> ring;
        if (FILE* f = fopen((dir + "/" + prefix + "_ring.bin").c_str(), "rb")) {
            int32_t hdr[2] = {0, 0};
            bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[1] == p->n_in && hdr[0] >= 0;
            for (int r = 0; ok && r < hdr[0]; r++) {
                std::vector<double> row(p->n_in);
                ok = fread(row.data(), sizeof(double), row.size(), f) == row.size();
                if (ok) ring.push_back(std::move(row));
            }
            fclose(f);
            if (!ok) return false;
        }
        p->nets = std::move(nets);
        p->ring = std::move(ring);
        while ((int)p->ring.size() > p->ring_cap) p->ring.pop_front();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
