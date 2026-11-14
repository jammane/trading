// jn_bridge.h -- the trainer's window onto j_neuro (github.com/jammane/j_neuro).
//
// j_neuro defines every computation bit for bit (IEEE-754 binary64, fixed evaluation order) and
// its build refuses fast-math. The trainer is built with -ffast-math, and most of j_neuro lives in
// headers (Net::step, mutate), so including them here would compile its arithmetic under the
// trainer's flags. This header therefore names no j_neuro type: everything goes through an opaque
// JNPool, implemented in jn_bridge.cpp, which CMake builds with j_neuro's own flags.
//
// A pool is N nets with the same input and output widths, plus a ring of the rows they have been
// fed, one per day. j_neuro nets are stateful (s_neurons keep their last k inputs), and a net is
// fully warmed only after get_warmup_steps() steps (spec/05). Each day a net is reset and replays
// its last warmup rows before today's: since an s_neuron's state is exactly its last k inputs,
// that gives the output a net stepped every day would give, without carrying state through
// selection (a mutated child starts reset, spec/05). A net whose warmup exceeds the ring's history
// is NOT READY and produces no output that day.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct JNPool;

struct JNStats {
    double nodes_mean = 0.0;    // hidden + output nodes (non-input), mean over the pool
    int    nodes_max  = 0;
    double warm_mean  = 0.0;    // get_warmup_steps()
    int    warm_max   = 0;
    int    ring_len   = 0;
};

// Builds n_slots minimal nets (spec/03), each from its own seed, so their random m/v ops differ.
// profile_path is a spec/09 profile JSON (profiles/training.json). Returns nullptr and sets err on
// failure.
JNPool* jn_pool_create(int n_in, int n_out, int n_slots, int ring_cap,
                       const std::string& profile_path, uint64_t seed, std::string& err);
void    jn_pool_free(JNPool* p);

int     jn_n_slots(const JNPool* p);

// Append today's row (n_in values). Call once per day, before jn_forward.
void    jn_push_row(JNPool* p, const float* row);
// Forget the history (a new pass restarts the calendar).
void    jn_clear_rows(JNPool* p);
// Replace the newest row (same day, recomputed input).
void    jn_replace_row(JNPool* p, const float* row);
// Rows of history held (at most the ring capacity).
int     jn_ring_len(const JNPool* p);

// Reset slot, replay its warmup rows, step the newest row; outputs (n_out) as float.
// Returns false, writing zeros, if the slot is not ready (or an output is not finite).
bool    jn_forward(JNPool* p, int slot, float* out);

// dst = mutate(src) under the pool's profile. `generation` decides growth generations
// (is_growth_generation over the pool seed) and is recorded on new neurons; `seed` is the RNG.
void    jn_mutate(JNPool* p, int src, int dst, uint64_t generation, uint64_t seed);
// dst[k] = copy of the net now in src_of_dst[k], k in [0, n), applied simultaneously.
void    jn_permute(JNPool* p, const int* src_of_dst, int n);
// Re-randomise pct% of slot's v_neuron weights (spec/06 diversity injection), in place.
void    jn_inject(JNPool* p, int slot, double pct, uint64_t seed);

JNStats jn_stats(const JNPool* p);
// The same over slots [0, n) only (n <= the pool's size).
JNStats jn_stats_n(const JNPool* p, int n);

// Slots [0, n) are the pool proper: jn_save writes them and jn_load reads them. Slots past n are
// the caller's scratch (the trainer's champion and history candidates), never persisted.
// Default: every slot.
void    jn_set_persist(JNPool* p, int n);
// dst := an exact copy of src (structure and weights).
void    jn_copy(JNPool* p, int src, int dst);
// A net's identity: a copy (jn_copy, jn_permute) keeps it, anything that makes a new net (create,
// load, mutate, bulk, inject) gives a new one. Equal ids = the same net, so a caller can keep
// clones out of a selection.
uint64_t jn_id(const JNPool* p, int slot);

// Bulking (j_neuro profiles/bulk.json, growth_budget = fill). Load the bulk profile once per pool;
// then dst = src grown by pct% of its neurons (exactly, within rounding). pct <= 0 copies src.
// Returns false when pct > 0 but the pool has no bulk profile: dst is then a plain copy, and the
// caller must say so (race pools went unbulked for JN2-JN4 because this fell back silently).
bool    jn_set_bulk_profile(JNPool* p, const std::string& path, std::string& err);
bool    jn_bulk(JNPool* p, int src, int dst, double pct, uint64_t generation, uint64_t seed);
// Neurons (non-input nodes) in one slot's net, as jn_stats counts them.
int     jn_slot_nodes(const JNPool* p, int slot);

// Wall time spent inside the bridge, summed over every pool and thread since the last take.
// A performance probe (`--jn-timing`); costs two clock reads per call.
struct JNTiming {
    double fwd = 0, fwd_warm = 0, fwd_reset = 0, fwd_step = 0;   // seconds; fwd = whole jn_forward
    double mutate = 0, mut_grow = 0, permute = 0, inject = 0, stats = 0, rows = 0, save = 0;
    long long n_fwd = 0, n_steps = 0, n_mut = 0, n_cold = 0;
};
JNTiming jn_timing_take();

// Saved nets with from_n_out outputs load into this pool by inserting a fresh output (j_neuro
// spec/12, add_output_fresh) at each index of insert_at, ascending in the final layout. Lets a
// pool trained before an output was added carry on instead of restarting.
void    jn_set_output_migration(JNPool* p, int from_n_out, const std::vector<int>& insert_at);

// <dir>/<prefix>_<slot>.jnet for every slot, plus <prefix>_ring.bin (the history rows).
bool    jn_save(const JNPool* p, const std::string& dir, const std::string& prefix);
// Loads what jn_save wrote; false (pool unchanged) if any net is missing or does not match.
bool    jn_load(JNPool* p, const std::string& dir, const std::string& prefix);
// One slot alone, e.g. a deployed model for the champion store.
bool    jn_save_slot(const JNPool* p, int slot, const std::string& path);
// Loads one net written by jn_save_slot into slot; false (slot unchanged) if missing or the
// widths do not match. The champion store and pass seeding move nets one at a time.
bool    jn_load_slot(JNPool* p, int slot, const std::string& path);
