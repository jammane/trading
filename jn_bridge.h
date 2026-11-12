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
