// jn_bridge.cpp -- see jn_bridge.h. Built with j_neuro's flags (-ffp-contract=off -fno-fast-math),
// never with the trainer's -ffast-math.
#include "jn_bridge.h"

#include <jneuro/jneuro.hpp>
#include <jneuro/modify.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
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
    jneuro::Profile bulk = jneuro::Profile::defaults();   // jn_set_bulk_profile
    bool has_bulk = false;
    std::vector<JNet> nets;
    int n_persist = -1;   // jn_set_persist; -1 = every slot
    // get_warmup_steps() per slot, -1 = not yet computed. It depends on a net's structure only,
    // but j_neuro recomputes it from the graph on every call (~2.4 ms on a ~420-node net, measured
    // 2026-11-12: ~6 s of a JN StockNN day). Cleared wherever a slot gets a different net.
    mutable std::vector<int> warm;
    std::vector<uint64_t> id;   // jn_id: per slot, kept by copies, renewed by every new net
    uint64_t next_id = 1;
    void renew(size_t slot) { if (id.size() < nets.size()) id.resize(nets.size(), 0); id[slot] = next_id++; }
    void renew_all() { id.assign(nets.size(), 0); for (size_t k = 0; k < nets.size(); k++) id[k] = next_id++; }
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

bool jn_set_bulk_profile(JNPool* p, const std::string& path, std::string& err) {
    try {
        std::ifstream f(path);
        if (!f) { err = "cannot read j_neuro bulk profile " + path; return false; }
        std::stringstream ss; ss << f.rdbuf();
        p->bulk = jneuro::profile_from_json(ss.str());
        if (p->bulk.growth_budget() != "fill") { err = path + ": growth_budget must be fill"; return false; }
        p->has_bulk = true;
        return true;
    } catch (const std::exception& e) {
        err = std::string("j_neuro bulk profile: ") + e.what();
        return false;
    }
}

bool jn_bulk(JNPool* p, int src, int dst, double pct, uint64_t generation, uint64_t seed) {
    if (pct <= 0.0 || !p->has_bulk) {   // a plain copy, so the same net (same id)
        jn_copy(p, src, dst);
        return pct <= 0.0;
    }
    jneuro::Profile prof = p->bulk;
    prof.set_overall("growth_max_pct", pct);
    jneuro::Rng rng(seed);
    p->nets[(size_t)dst] = jneuro::mutate(p->nets[(size_t)src], prof, generation, rng, nullptr, true);
    p->warm[(size_t)dst] = -1;
    p->renew((size_t)dst);
    return true;
}

int jn_slot_nodes(const JNPool* p, int slot) {
    const auto& n = p->nets[(size_t)slot];
    return (int)n.nodes.size() - (int)n.n_inputs;
}

void jn_set_output_migration(JNPool* p, int from_n_out, const std::vector<int>& insert_at) {
    p->mig_from = from_n_out;
    p->mig_at = insert_at;
}

// ── timing probe (jn_timing_take) ──
enum { T_FWD, T_WARM, T_RESET, T_STEP, T_MUT, T_GROW, T_PERM, T_INJ, T_STATS, T_ROWS, T_SAVE,
       C_FWD, C_STEPS, C_MUT, C_COLD, T_N };
static std::atomic<long long> g_t[T_N];
static inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct TScope {
    int k; long long t0;
    explicit TScope(int k_) : k(k_), t0(now_ns()) {}
    ~TScope() { g_t[k].fetch_add(now_ns() - t0, std::memory_order_relaxed); }
};
static inline void tcount(int k, long long n = 1) { g_t[k].fetch_add(n, std::memory_order_relaxed); }

JNTiming jn_timing_take() {
    long long v[T_N];
    for (int k = 0; k < T_N; k++) v[k] = g_t[k].exchange(0, std::memory_order_relaxed);
    auto s = [&](int k) { return (double)v[k] * 1e-9; };
    JNTiming t;
    t.fwd = s(T_FWD); t.fwd_warm = s(T_WARM); t.fwd_reset = s(T_RESET); t.fwd_step = s(T_STEP);
    t.mutate = s(T_MUT); t.mut_grow = s(T_GROW); t.permute = s(T_PERM); t.inject = s(T_INJ);
    t.stats = s(T_STATS); t.rows = s(T_ROWS); t.save = s(T_SAVE);
    t.n_fwd = v[C_FWD]; t.n_steps = v[C_STEPS]; t.n_mut = v[C_MUT]; t.n_cold = v[C_COLD];
    return t;
}

static int warm_of(const JNPool* p, int slot) {
    int& w = p->warm[(size_t)slot];
    if (w < 0) { TScope tw(T_WARM); w = (int)p->nets[(size_t)slot].get_warmup_steps(); }
    return w;
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
        p->warm.assign(p->nets.size(), -1);
        p->renew_all();
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
    TScope ts(T_ROWS);
    p->ring.emplace_back(row, row + p->n_in);
    while ((int)p->ring.size() > p->ring_cap) p->ring.pop_front();
}

void jn_clear_rows(JNPool* p) { p->ring.clear(); }

int jn_ring_len(const JNPool* p) { return (int)p->ring.size(); }

void jn_replace_row(JNPool* p, const float* row) {
    if (p->ring.empty()) { jn_push_row(p, row); return; }
    p->ring.back().assign(row, row + p->n_in);
}

bool jn_forward(JNPool* p, int slot, float* out) {
    TScope ts(T_FWD);
    tcount(C_FWD);
    for (int o = 0; o < p->n_out; o++) out[o] = 0.f;
    JNet& net = p->nets[slot];
    const size_t warm = (size_t)warm_of(p, slot);
    if (p->ring.size() < warm + 1) { tcount(C_COLD); return false; }   // not enough history yet
    { TScope tr(T_RESET); net.reset_state(); }   // keeps the plan; a slot is never edited in place
    std::vector<double> y;
    {
        TScope tst(T_STEP);
        for (size_t r = p->ring.size() - warm - 1; r < p->ring.size(); r++) y = net.step(p->ring[r]);
        tcount(C_STEPS, (long long)warm + 1);
    }
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
    TScope ts(T_MUT);
    tcount(C_MUT);
    jneuro::Rng rng(seed);
    bool grow;
    { TScope tg(T_GROW); grow = jneuro::is_growth_generation(p->seed, generation, p->profile); }
    p->nets[dst] = jneuro::mutate(p->nets[src], p->profile, generation, rng, nullptr, grow);
    p->warm[(size_t)dst] = -1;
    p->renew((size_t)dst);
}

void jn_permute(JNPool* p, const int* src_of_dst, int n) {
    TScope ts(T_PERM);
    std::vector<JNet> tmp;
    tmp.reserve(n);
    for (int k = 0; k < n; k++) tmp.push_back(p->nets[src_of_dst[k]].copy_genome());
    std::vector<int> tw(n);
    std::vector<uint64_t> ti(n);
    for (int k = 0; k < n; k++) { tw[k] = p->warm[(size_t)src_of_dst[k]]; ti[k] = p->id[(size_t)src_of_dst[k]]; }
    for (int k = 0; k < n; k++) {
        p->nets[k] = std::move(tmp[k]); p->warm[(size_t)k] = tw[k]; p->id[(size_t)k] = ti[k];
    }
}

void jn_inject(JNPool* p, int slot, double pct, uint64_t seed) {
    TScope ts(T_INJ);
    jneuro::Rng rng(seed);
    p->nets[slot] = jneuro::inject_diversity(p->nets[slot], pct, rng);
    p->warm[(size_t)slot] = -1;
    p->renew((size_t)slot);   // weights only, so unchanged in principle; recomputed to be safe
}

JNStats jn_stats(const JNPool* p) { return jn_stats_n(p, (int)p->nets.size()); }

JNStats jn_stats_n(const JNPool* p, int n_slots) {
    TScope ts(T_STATS);
    JNStats s;
    s.ring_len = (int)p->ring.size();
    const size_t n_use = std::min(p->nets.size(), (size_t)std::max(0, n_slots));
    if (n_use == 0) return s;
    double nodes = 0.0, warm = 0.0;
    for (size_t k = 0; k < n_use; k++) {
        const auto& n = p->nets[k];
        const int nn = (int)n.nodes.size() - (int)n.n_inputs;
        const int w  = warm_of(p, (int)k);
        nodes += nn; warm += w;
        if (nn > s.nodes_max) s.nodes_max = nn;
        if (w > s.warm_max) s.warm_max = w;
    }
    s.nodes_mean = nodes / (double)n_use;
    s.warm_mean  = warm  / (double)n_use;
    return s;
}

void jn_set_persist(JNPool* p, int n) { p->n_persist = n; }

static size_t persisted(const JNPool* p) {
    return p->n_persist < 0 ? p->nets.size() : std::min(p->nets.size(), (size_t)p->n_persist);
}

void jn_copy(JNPool* p, int src, int dst) {
    if (src == dst) return;
    p->nets[(size_t)dst] = p->nets[(size_t)src].copy_genome();
    p->warm[(size_t)dst] = p->warm[(size_t)src];
    p->id[(size_t)dst] = p->id[(size_t)src];
}

uint64_t jn_id(const JNPool* p, int slot) { return p->id[(size_t)slot]; }

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
        p->warm[(size_t)slot] = -1;
        p->renew((size_t)slot);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool jn_save(const JNPool* p, const std::string& dir, const std::string& prefix) {
    TScope ts(T_SAVE);
    for (size_t s = 0; s < persisted(p); s++)
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
        for (size_t s = 0; s < persisted(p); s++) {
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
        // Persisted slots are replaced; scratch slots past them keep their nets.
        for (size_t s = 0; s < nets.size(); s++) { p->nets[s] = std::move(nets[s]); p->renew(s); }
        p->warm.assign(p->nets.size(), -1);
        p->ring = std::move(ring);
        while ((int)p->ring.size() > p->ring_cap) p->ring.pop_front();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
