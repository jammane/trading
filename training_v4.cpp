// training_v4.cpp — C++ port of training_v4.py
// Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
// Run:   ./build/training_v4_cpp --output models [--load-dir DIR] [--start-day N] [--stop-day N]
//        [--passes N] [--sigma F] [--master-sigma F] [--sigma-decay F] [--workers N]

#define TRAINER_VERSION "0.8.1.27"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <sys/stat.h>
#include <unistd.h>
#include <csignal>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cblas.h>
#include <sys/mman.h>

#include "mt1_pool.h"
#include "mt1_grad.h"   // MT1Net layout/forward + pool scoring (shared with tests/test_mt1_pool.cpp)
#include "pass_seeding.h"   // pass-boundary share + interleave (shared with tests/test_pass_seeding.cpp)
#include "sort_util.h"      // small fixed-buffer index sort (shared with tests/test_sort_util.cpp)
#include "bayes_online.h"   // online conditional models trained during the pass
#include "mt1_backprop.h"   // gradient-trained MT1 (forward/backward/Adam, no autodiff lib)

// Force OpenBLAS single-threaded: multi-threaded BLAS with N worker threads causes
// 2×N threads competing for N CPUs, multiplying overhead 2-3× per forward pass.
extern "C" void openblas_set_num_threads(int);

namespace fs = std::filesystem;

// ── Universe ───────────────────────────────────────────────────────────────────

static constexpr int N_IND   = 12;
static constexpr int IND_SYMS = 12;
static constexpr int N_SYMS  = N_IND * IND_SYMS;   // 144

// Runtime-loaded from universe.json (populated by load_universe_json in main)
static std::string g_ind_names[N_IND];
static std::string g_syms[N_IND][IND_SYMS];

// Display-only abbreviations — not affected by symbol swaps
static const char* const IND_SHORT[N_IND] = {
    "hardware ","software ","financial","discret  ","services ","health   ",
    "industrl ","staples  ","energy   ","utilitie ","land     ","materials"
};

// Load industry names and symbols from universe.json.
// The JSON has exactly N_IND * (1 + IND_SYMS) quoted strings in insertion order:
//   name0, sym[0][0..11], name1, sym[1][0..11], ...
// This matches the order produced by Python's json.dump(INDUSTRIES, f, indent=2).
static bool load_universe_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        fprintf(stderr, "FATAL: cannot open %s\n", path.c_str());
        fprintf(stderr, "       Run: python swap_symbols.py '{}' to generate it.\n");
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(ifs)), {});

    std::vector<std::string> tokens;
    for (size_t p = 0; p < text.size(); ) {
        size_t a = text.find('"', p);
        if (a == std::string::npos) break;
        size_t b = text.find('"', a + 1);
        if (b == std::string::npos) break;
        tokens.push_back(text.substr(a + 1, b - a - 1));
        p = b + 1;
    }

    const int expected = N_IND * (1 + IND_SYMS);   // 12 * 13 = 156
    // Extended format: token[0]="industries" key, then 156 name/symbol tokens, then extras
    int offset = 0;
    if ((int)tokens.size() == expected + 1 && tokens[0] == "industries") {
        offset = 1;  // skip the "industries" key itself
    } else if ((int)tokens.size() >= expected + 1 && tokens[0] == "industries") {
        offset = 1;  // extended format (all_symbols/industry_names present too)
    } else if ((int)tokens.size() != expected) {
        fprintf(stderr, "FATAL: %s has %d quoted tokens, expected %d.\n",
                path.c_str(), (int)tokens.size(), expected);
        fprintf(stderr, "       Re-run: python swap_symbols.py '{}' to regenerate.\n");
        return false;
    }
    for (int i = 0; i < N_IND; i++) {
        g_ind_names[i] = tokens[offset + i * (1 + IND_SYMS)];
        for (int j = 0; j < IND_SYMS; j++)
            g_syms[i][j] = tokens[offset + i * (1 + IND_SYMS) + 1 + j];
    }
    return true;
}

// ── Training constants ─────────────────────────────────────────────────────────

static constexpr int   N_SLOTS             = 200;
static constexpr int   ELITE_COUNT         = 17;
static constexpr int   WAVG_COUNT          = 3;
static constexpr int   ELITE_POOL          = 20;   // industry + MT2
static constexpr int   MUTATIONS_PER_PARENT = 9;

// MT1 heads/tails: one shared head pool + 4 specialized tail pools per industry, 200 slots each.
//   Slot layout: 0-16 direct elites | 17-19 wavg blends | 20-199 mutations (NO injection slots —
//   the shared head propagates cross-component learning; capacity goes into more elite mutations).
// MT1 pool geometry lives in mt1_pool.h (MT1_POOL_SLOTS, MT1_POOL_MIN_AGE, MT1_POOL_CULL_PCT,
// MT1_POOL_ELITE_PCT, MT1_POOL_LINEAGE_CAP). The five-pool constants that stood here — the 17/3/180
// replay layout, the 10-day scoring window, and the direction constant-collapse detector with its
// trip count, cooldown and injection blend — went with the pools they configured.


static constexpr int   HIST_WINDOW         = 15;

static constexpr float IND_STARTING_CASH   = 25000.0f;
static constexpr float MST_STARTING_CASH   = 300000.0f;
static constexpr float IND_UNIT_PRICE      = 25000.0f;
static constexpr float MAX_SINGLE_STOCK_PCT = 0.60f;

// Whole shares only. Every Alpaca order is submitted with qty= (never notional=), and the
// stop-loss orders forbid fractional trading outright — so production floors to int
// (production_v2.py: 'quantity': int(_qty)) while the simulator traded continuous quantities.
// That let training buy 0.7 of a share and collect exposure where production buys NOTHING,
// and because int() truncates DOWN the bias was directional: production always deployed less
// than training assumed. The error scales inversely with capital (~3% at $25,000/industry,
// ~47% at a $2,000 single-industry start), i.e. worst exactly where production starts.
// The epsilon absorbs float error so 2.9999997 does not become 2.
static inline float whole_shares(float q) {
    float w = std::floor(q + 1e-4f);
    return w > 0.f ? w : 0.f;
}
static constexpr float SEC_FEE_RATE        = 0.0000278f;
static constexpr float FINRA_TAF_PER_SHARE = 0.000166f;
static constexpr float FINRA_TAF_MAX       = 8.30f;
static constexpr float SLIPPAGE_RATE       = 0.001f;

static constexpr int   STOCKNN_PARAMS  = 928825;
// New flat 5-layer MasterNN: 444→444→444→312→180→48
// fc1: 444×444+444=197580  fc2: +197580=395160  fc3: 444×312+312=138840→534000
// fc4: 312×180+180=56340→590340  fc_out: 180×48+48=8688→599028
static constexpr int   MASTERNN_PARAMS = 599028;
static constexpr int   MASTER_START_DAY = 30;
static constexpr float TIER_WEIGHTS[4] = {0.f, 1.f, 1.5f, 2.25f};
static constexpr float NULL_DENOM      = 1.f + 1.5f + 2.25f;  // 4.75

static constexpr int HIST_DAYS    = 5;
static constexpr int HIST_PER_DAY = 10;
static constexpr int HIST_ELITE   = 7;   // top-7 direct elite slots saved per day
static constexpr int HIST_WAVG    = 3;   // wavg slots (17,18,19) saved per day

// MT1: per-industry predictor of THIS industry's next-session StockNN P&L. One network, one
// output, one 200-slot pool per industry. Sizes live in mt1_pool.h (MT1NET_PARAMS = 3501).
static constexpr int   MT1_START_DAY       = 25;
// The gradient-trained MT1 starts at day 400, not MT1_START_DAY. The first ~400 days are StockNN
// learning to trade at all -- measured, the book produces -$0.41/day over that span against
// +$19.84/day in the final third -- so the P&L MT1 would be fitting there describes a model that
// no longer exists by the time it matters.
static constexpr int   MT1_BP_START_DAY    = 400;
// How many of the 200 scored models contribute a training row each industry-day. All of them:
// with per-model order intent in the input these are genuinely distinct rows, not one row repeated.
// Slots 0..ELITE_POOL-1 are the elites and blends, the rest are this day's mutation children --
// exactly the variety a critic needs to learn what separates a good order set from a bad one.
// Cost is ~2.05M gradient steps on a 4,977-param net over a full pass, which is nothing.
static constexpr int   MT1_BP_MODELS       = N_SLOTS;
// Carried gradient entries begin TRAINING here instead of MT1_BP_START_DAY -- but only from the
// SECOND pass onward.
//
// The compensation is for a STALE optimiser, not a cold one. Adam's bias correction is done after
// ~5,000 steps and the race takes 200 steps per industry-day, so a cold start is warm inside ~25
// days and needs no help. What does need help is a CARRIED entry crossing a pass boundary: its
// moment estimates were fitted against the previous champion's order distribution, and the pass
// re-seeds StockNN from a new champion/challenger blend. The extra days let those moments
// re-adapt to the new distribution before the scoring window opens.
//
// Pass 1 therefore gets no extension, and that is the useful part: with nothing yet to carry and
// the same training start, grad and grdC are identical but for their RNG seed. Pass 1's gap
// between them IS the noise floor, measured in the same table by the same procedure -- so a gap in
// passes 2-5 can be read against it rather than against an assumption.
static constexpr int   MT1_BP_WARM_DAY     = 100;

// ── the MT1 race: 3 architectures x 2 search methods ────────────────────────────
//
// Every entry trains in the SAME daily loop on the SAME order sets, in the same order. Holding
// the architecture constant across the two search methods is what makes "evolutionary vs
// gradient" a statement about SEARCH rather than about shape -- and it is also what makes the
// final weights of each pass comparable element-wise, which is why they are kept.
//
// Two metrics, and they answer different questions:
//
//   within-day rank  Spearman across the 200 order sets scored that industry-day. They share the
//                    day's market move, so this ranks the TRADING and nothing else.
//   across-day corr  the deployed model's prediction against the realised industry P&L. This
//                    still contains the common market move, so it is the easier number, and it
//                    is the one an allocator would consume.
//
// MT1Net CANNOT rank order sets, and that is not a defect -- its 74 inputs are market and
// portfolio curves with no order intent in them, so all 200 models present it with an identical
// vector. Its within-day rank must therefore come back ~0.0000, which makes it a built-in control
// for that metric: a real architecture, trained the same way, that structurally cannot see what
// is being ranked. If it reads non-zero the measurement leaks. (The previous artificial control
// read +0.06 because ordinal ranks over tied predictions recovered the slot index; see
// bp_avg_ranks.) On the across-day metric MT1Net is a full competitor.
//
// MT1S-stock differs from MT1S-sum only on the GRADIENT side, where per-symbol labels give twelve
// rows per industry-day instead of one. Evolutionary selection sees a single scalar score per
// model per day, so the per-symbol target has no meaning there without a different score function
// -- which would need per-symbol predictions in MT1PoolScratch. Rather than leave the slot empty
// it runs as a SEED REPLICATE of MT1S-sum-evo: identical in every way but its RNG seed, so the
// gap between the two measures run-to-run variance of the evolutionary search directly. This
// project has been misled by that variance before (trade delta -$81,021 on one run against
// +$406,381 on another), so having the noise floor measured inside the same table is worth a slot.
using MT1Forward = float (*)(const float*, const float*, float);

enum RaceArch  { ARCH_MT1, ARCH_MT1C, ARCH_MT1S };
enum RaceSrch  { SRCH_EVO, SRCH_GRAD };

struct RaceEntry {
    const char* name;
    RaceArch    arch;
    RaceSrch    search;
    bool        per_symbol;   // gradient only: one row per symbol instead of one per industry
    bool        carry;        // keep the model across a pass boundary, or start fresh
    const char* tag;          // pool file prefix; must be unique per evolutionary entry
};

// Day this entry starts TRAINING. Measurement always begins at MT1_BP_START_DAY.
// pass is 0-based: the warm-up applies only where there is carried state to re-adapt.
static inline int race_train_start(const RaceEntry& e, int pass) {
    return (e.search == SRCH_GRAD && e.carry && pass > 0) ? MT1_BP_WARM_DAY : MT1_BP_START_DAY;
}
// CARRY is the third axis, and it is deliberate rather than incidental.
//
// The evolutionary pools have always carried: the champion store seeds the next pass, so a pool
// that reset would throw away the search that produced the champion it is paired with. Gradient
// entries appear BOTH ways -- restart and carry -- for two reasons. It keeps the evo-vs-grad
// comparison honest, since comparing a carried pool against a restarted net would measure five
// passes of accumulated search against one rather than anything about search method. And the
// grad-restart vs grad-carry pair answers a question nothing here has tested: whether accumulating
// across passes helps at all. It is not obvious that it does -- measured on StockNN, pass 5
// performs at 0.96-1.02x of pass 1, which is no detectable carryover whatsoever.
//
// Carried gradient entries keep their Adam moments along with their weights. The moments are
// somewhat stale across a boundary, since the pass re-seeds StockNN and the target distribution
// shifts, but Adam re-adapts within a few hundred steps and zeroing them would make "carry" a
// different thing than simply continuing to train.
static constexpr RaceEntry RACE[] = {
    {"MT1      evo ", ARCH_MT1,  SRCH_EVO,  false, true,  "r_mt1"   },  // control for rank
    {"MT1      grad", ARCH_MT1,  SRCH_GRAD, false, false, nullptr   },  // control for rank
    {"MT1      grdC", ARCH_MT1,  SRCH_GRAD, false, true,  nullptr   },
    {"MT1C     evo ", ARCH_MT1C, SRCH_EVO,  false, true,  "r_mt1c"  },
    {"MT1C     grad", ARCH_MT1C, SRCH_GRAD, false, false, nullptr   },
    {"MT1C     grdC", ARCH_MT1C, SRCH_GRAD, false, true,  nullptr   },
    {"MT1S-sum evo ", ARCH_MT1S, SRCH_EVO,  false, true,  "r_mt1s"  },
    {"MT1S-sum grad", ARCH_MT1S, SRCH_GRAD, false, false, nullptr   },
    {"MT1S-sum grdC", ARCH_MT1S, SRCH_GRAD, false, true,  nullptr   },
    {"MT1S-stk evo ", ARCH_MT1S, SRCH_EVO,  true,  true,  "r_mt1s2" },  // seed replicate
    {"MT1S-stk grad", ARCH_MT1S, SRCH_GRAD, true,  false, nullptr   },
    {"MT1S-stk grdC", ARCH_MT1S, SRCH_GRAD, true,  true,  nullptr   },
};
static constexpr int RACE_N = (int)(sizeof(RACE) / sizeof(RACE[0]));

// Entries read by position outside the race loop. Named and asserted because a bare index went
// stale when the table grew from 5 variants to 12: race[3] had been MT1C grad and became MT1C evo,
// whose gradient vectors are never allocated, so .w[i] dereferenced null and the smoke segfaulted.
// Reordering the table now fails the build instead.
static constexpr int RACE_DEPLOYED = 4;   // MT1C grad (restart): publishes bp_pred
static constexpr int RACE_LOG_POOL = 0;   // MT1 evo: the pool mt_training_log.bin records
static_assert(RACE[RACE_DEPLOYED].arch == ARCH_MT1C && RACE[RACE_DEPLOYED].search == SRCH_GRAD &&
              !RACE[RACE_DEPLOYED].per_symbol && !RACE[RACE_DEPLOYED].carry,
              "RACE_DEPLOYED must name MT1C grad (restart)");
static_assert(RACE[RACE_LOG_POOL].arch == ARCH_MT1 && RACE[RACE_LOG_POOL].search == SRCH_EVO,
              "RACE_LOG_POOL must name MT1 evo");

static inline int race_params(RaceArch a) {
    return a == ARCH_MT1 ? MT1NET_PARAMS : a == ARCH_MT1C ? MT1CNET_PARAMS : MT1SNET_PARAMS;
}
// MT1Forward-compatible wrapper so MT1S can use the existing evolutionary pool machinery.
static inline float mt1s_forward_mt1f(const float* W, const float* in, float /*extra*/) {
    return mt1s_forward(W, in);
}
using MT1Init2 = void (*)(float*, uint64_t);

static inline MT1Forward race_forward(RaceArch a) {
    return a == ARCH_MT1 ? mt1net_forward : a == ARCH_MT1C ? mt1cnet_forward : mt1s_forward_mt1f;
}
// 146 -> 32 -> 8 -> 1. Deliberately smaller than MT1CNet's 11,177: measured offline, the larger
// shape scored WORSE on the same data (+0.0077 vs +0.0184). The target is close to pure noise,
// so capacity buys memorisation rather than skill.
static constexpr int   MT1_BP_HIDDEN_A     = 32;
static constexpr int   MT1_BP_HIDDEN_B     = 8;
// Horizon of the MARKET forward return, which is now only an MT2 read-only diagnostic and the
// drift study's runway. MT1's own target is the next session, full stop.
static constexpr int   MT1_FWD_DAYS        = 10;
// MT1 steps once per session. The constant is 1 and stays 1 — it survives only as the extent of
// the per-day staging arrays, which the log writer and the CSV writer index. The 25-day
// block-alternating T1/H/T2 schedule it used to name is gone.
static constexpr int   MT1_DAYS            = 1;

// MT2 injection: fire when ≥75% of pool scores below threshold (worst ~15% of days)
static constexpr float MT2_INJ_THRESHOLD = -7.0f;
static constexpr int   MT2_INJ_MIN_BELOW = (int)(N_SLOTS * 0.75f);  // 150/200
// MT2 pool-consensus diagnostic: look-behind window for the reliability-weighted ensemble
static constexpr int   MT2_LB_DAYS       = 5;

// MT2NN: FC[12→36→36] ‖ LSTM[1→36×2layers] → concat72 → 66→60→54→48
//
// BREAKING as of the MT1 rebuild: the input was 48 = 12 industries × 4 MT1 channels. MT1 now emits
// ONE number per industry — its dollar prediction of that industry's next-session P&L — so the
// input is 12, and the LSTM walks the 12 industries one scalar at a time instead of four.
// MT2NN_PARAMS changes, so every mt2_*.bin from before this change is unloadable; load_bin
// validates by exact element count and falls back to random init silently, so start clean.
static constexpr int   MT2NN_PARAMS   = 32844;
// FC branch
static constexpr int MT2_FC1_W  =     0;  static constexpr int MT2_FC1_B  =   432;  // 36×12
static constexpr int MT2_FC2_W  =   468;  static constexpr int MT2_FC2_B  =  1764;  // +36, 36×36
// LSTM L1 (input_size=1, hidden=36): wih[144×1], whh[144×36], bih[144], bhh[144]
static constexpr int MT2_L1_WIH =  1800;  static constexpr int MT2_L1_WHH =  1944;
static constexpr int MT2_L1_BIH =  7128;  static constexpr int MT2_L1_BHH =  7272;
// LSTM L2 (input_size=36, hidden=36): wih[144×36], whh[144×36], bih[144], bhh[144]
static constexpr int MT2_L2_WIH =  7416;  static constexpr int MT2_L2_WHH = 12600;
static constexpr int MT2_L2_BIH = 17784;  static constexpr int MT2_L2_BHH = 17928;
// Taper (biases follow weights immediately for each layer → kaiming_init works)
static constexpr int MT2_T1_W  = 18072;  static constexpr int MT2_T1_B  = 22824;  // 72×66
static constexpr int MT2_T2_W  = 22890;  static constexpr int MT2_T2_B  = 26850;  // +66, 66×60
static constexpr int MT2_T3_W  = 26910;  static constexpr int MT2_T3_B  = 30150;  // +60, 60×54
static constexpr int MT2_OUT_W = 30204;  static constexpr int MT2_OUT_B = 32796;  // +54, 54×48+48=32844
static_assert(MT2_OUT_B + 48 == MT2NN_PARAMS, "MT2NN layout drifted from MT2NN_PARAMS");

// --no-save (v0.6.6.0): train into a scratch directory and delete it at exit.
//
// It used to mean "skip all model writes", which silently DISABLED StockNN training entirely:
// step_industry reloads elite_buf and hist_buf from disk at the top of EVERY day, so with nothing
// written they were re-random-initialised every day from the same seed. Measured on the v0.6.0.0-A
// run: 297,120 random-init lines == 20 slots x 12 industries x 1238 days. Nothing ever learned;
// only the portfolio carried forward, growing purely from picking the best of 200 fresh random
// models each day. Every "diagnostic" run taken under --no-save measured that, not training.
//
// The point of the flag was never "do not write" — it was "do not leave a 3 GB run directory
// behind". So it now writes to <output>.nosave and removes it on exit.
// --control-untrained (v0.6.9.0): a deliberate NO-LEARNING control. StockNN elites are
// re-randomised every day instead of loaded, so the pool never accumulates anything, while the
// market data, selection, scoring and logging stay identical. It answers a question this codebase
// cannot currently answer: how much of the observed performance comes from TRAINING versus from
// the selection mechanism operating on arbitrary models.
//
// The question is live because an accidental version of this ran for months as the --no-save
// defect, and it scored comparably to trained runs (+151.4% against ~+127-142%). If a control
// matches a trained run, the problem is not the passes — it is that the scoring cannot tell
// models apart at all.
//
// Deliberately LOUD: a banner at startup, a marker in the CSV header, and a per-pass reminder.
// The whole hazard of this mode is that its output looks exactly like a real run.
// --control-random (v0.6.10.0): the no-learning control, run IN-PROCESS as a paired
// same-day comparison instead of as a separate run.
//
// After each day's scoring and mutation, spawn N genuinely random models, score them against the
// SAME reference portfolio on the SAME day, record best/mean, and throw them away. Nothing is
// persisted and nothing selects on them.
//
// Why paired rather than a second training run: two independent runs differ by the seed noise
// floor (~30% sd per industry, 4.3% on totals), which is the same order as the effect being
// measured — so a standalone control could not resolve it. Scored against the same reference
// portfolio on the same day, the comparison is paired and that noise cancels. It also needs no
// second process (the box has under 2 GB of RAM and one trainer already mlocks ~720 MB), no
// extra disk, and one model buffer instead of a 928 MB pool.
//
// Pass 1 only: the control never learns, so passes 2-5 would be identical replicates at full cost.
static bool g_control_random = false;
static int  g_cur_pass = 0;      // set once per pass before workers start; read-only in workers
static FILE* g_ctl_csv = nullptr;  // control_log.csv, long format, opened only with --control-random
static FILE* g_hold_csv = nullptr; // holdings_log.csv: slot-0's composition, one row per industry-day

static bool g_control_untrained = false;
// Skip EVERY neural allocator for this run: MT1Net, MT1CNet, MT2INet and MT2 itself. Frees
// ~369 MB of pools and scratch, and their forward and evolve passes, leaving StockNN plus the
// online conditional models. MT2 goes with them because it consumes MT1's predictions as in12 --
// running it against a dead MT1 would train it on twelve zeros and quietly corrupt its pool.
// StockNN is untouched, so the P&L series the conditional models learn from is the same one a
// normal run produces.
static bool g_no_nn_race = false;
static bool g_no_save = false;

// ── Trading lock (v0.6.5.0) ───────────────────────────────────────────────────
// The droplet has 2 cores and a training run uses ~150% CPU. When training and a production cycle
// compete, the loser slows 3-5x — measured: training fell from 34 s/day to 1-3 min/day with two
// short jobs alongside it. Production is the time-sensitive side (fetch, decide, submit orders),
// so the trainer yields: it polls this lock between training days and sleeps while it is held.
//
// Written by trading_lock() in production_v2.py — THE DEFAULT PATH MUST MATCH THE ONE THERE.
// Absolute because the two run from different worktrees (/root/trading-ht vs /root/trading).
// Line 1 of the file is the holder's PID.
static std::string g_trade_lock = "/run/trading/trading_active.lock";
static constexpr int TRADE_LOCK_STALE_SEC = 1800;   // 30 min; a production cycle takes minutes
static constexpr int TRADE_LOCK_POLL_SEC  = 15;


// ── Run seed (v0.6.4.0) ────────────────────────────────────────────────────────
// Every PCG32 in this file used to be seeded from hardcoded constants plus loop indices, with no
// entropy anywhere — so "random init" produced byte-identical weights on every run and two runs of
// the same binary could not differ. That made a noise floor impossible to measure: a 55-point gap
// between versions and a gap caused by one lucky initialisation were indistinguishable.
//
// g_run_seed defaults to the clock and is mixed into every seed site through mix_seed(). It is
// LOGGED at startup, and --seed N reproduces a run exactly when that is what you want.
// RE-SEEDED AT THE TOP OF EVERY PASS, not once per run. The per-day/per-industry seeds carry no
// pass component — seed_rng.seed(actual_day * 1000007 + ind_i * 13) — so before v0.6.4.0 pass 1
// day 17 and pass 5 day 17 applied the IDENTICAL perturbation vector. Every pass re-walked the
// same noise sequence, differing only in which parents it was applied to. That is a plausible
// contributor to extra passes failing to help.
static uint64_t g_run_seed = 0;
static uint64_t g_seed_arg = 0;   // --seed N; 0 = unset, derive each pass from the clock

// splitmix64 finaliser: decorrelates the run seed from the per-day / per-industry structure, so
// adjacent seeds do not yield correlated streams.
static inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t mix_seed(uint64_t base) { return splitmix64(base + g_run_seed); }
// --dir-reps: how many times the direction pool replays each block. The gentle 8.3% cull only turns
// over ~10 of 200 slots a day, so reps are what restore evolutionary throughput. They also mean N
// epochs over the same 25 days, which is an overfitting risk — hence a flag, so it can be swept and
// judged on the out-of-sample instrument instead of assumed.
static int  g_dir_reps = 20;

// ── Layer dimensions ───────────────────────────────────────────────────────────

// today vector width, and the fc_today input width it implies. The `cat` scratch buffer in
// stock_forward is reused for BOTH the inject-layer concatenations (max in_sz 245) and the today
// concatenation, so it must be sized for the larger of the two — it was a bare literal (420) and
// a 232-wide today vector overflowed it by two floats, segfaulting only once the width grew.
static constexpr int STOCK_TODAY_DIM = 232;                      // today_arr width
static constexpr int STOCK_TODAY_IN  = 190 + STOCK_TODAY_DIM;    // 422 = fc_today inputs

static constexpr int STOCK_INJ_IN [14] = {180,185,190,195,200,205,210,215,220,225,230,235,240,245};
static constexpr int STOCK_INJ_OUT[14] = {125,130,135,140,145,150,155,160,165,170,175,180,185,190};

// Float offsets into the flat weight array for StockNN
static constexpr int STOCK_SEED_W  = 0;
static constexpr int STOCK_SEED_B  = 7200;
static constexpr int STOCK_INJ_W[14] = {
     7320, 29945, 54125, 79910,107350,136495,167395,200100,
    234660,271125,309545,349970,392450,437035
};
static constexpr int STOCK_INJ_B[14] = {
    29820, 53995, 79775,107210,136350,167245,199945,234500,
    270960,309375,349795,392270,436850,483585
};
static constexpr int STOCK_TODAY_W = 483775;
static constexpr int STOCK_TODAY_B = 610375;
static constexpr int STOCK_FLAT1_W = 610675;
static constexpr int STOCK_FLAT1_B = 700675;
static constexpr int STOCK_FLAT2_W = 700975;
static constexpr int STOCK_FLAT2_B = 790975;
static constexpr int STOCK_FC1_W   = 791275;
static constexpr int STOCK_FC1_B   = 862375;
static constexpr int STOCK_FC2_W   = 862612;
static constexpr int STOCK_FC2_B   = 903850;
static constexpr int STOCK_FC3_W   = 904024;
static constexpr int STOCK_FC3_B   = 923338;
static constexpr int STOCK_OUT_W   = 923449;
static constexpr int STOCK_OUT_B   = 928777;

// Float offsets into the flat weight array for MasterNN (5-layer flat FC)
static constexpr int MAST_FC1_W   = 0;
static constexpr int MAST_FC1_B   = 197136;   // 444*444
static constexpr int MAST_FC2_W   = 197580;   // +444
static constexpr int MAST_FC2_B   = 394716;   // +444*444
static constexpr int MAST_FC3_W   = 395160;   // +444
static constexpr int MAST_FC3_B   = 533688;   // +444*312
static constexpr int MAST_FC4_W   = 534000;   // +312
static constexpr int MAST_FC4_B   = 590160;   // +312*180
static constexpr int MAST_OUT_W   = 590340;   // +180
static constexpr int MAST_OUT_B   = 598980;   // +180*48
// Verify: 598980+48 = 599028 = MASTERNN_PARAMS

// ── PCG32 fast RNG ─────────────────────────────────────────────────────────────

struct PCG32 {
    uint64_t state{0}, inc{1};

    void seed(uint64_t s, uint64_t seq = 1) {
        state = 0; inc = (seq << 1) | 1;
        next(); state += s; next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xs >> rot) | (xs << ((-rot) & 31u));
    }
    float next_float() { return (next() >> 8) * (1.0f / (1 << 24)); }
};

// Two Box-Muller normal variates into out[0] and out[1]
static inline void bm_pair(PCG32& rng, float sigma, float* out) {
    float u1 = rng.next_float() + 1e-7f;
    float u2 = rng.next_float();
    float r   = sigma * sqrtf(-2.0f * logf(u1));
    float th  = 6.2831853f * u2;
    out[0] = r * cosf(th);
    out[1] = r * sinf(th);
}

// ── Forward pass helpers ────────────────────────────────────────────────────────

static inline void sgemv_relu(const float* W, const float* b,
                               const float* x, float* y, int out, int in) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in, 1.f, W, in, x, 1, 0.f, y, 1);
    for (int i = 0; i < out; i++) { y[i] += b[i]; if (y[i] < 0.f) y[i] = 0.f; }
}

static inline void sgemv_only(const float* W, const float* b,
                               const float* x, float* y, int out, int in) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in, 1.f, W, in, x, 1, 0.f, y, 1);
    for (int i = 0; i < out; i++) y[i] += b[i];
}

static inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }


// Close relative to the day's weighted average price (v0.6.1.0, StockNN today feature 16).
//   A   = (2O + 3C + H + L) / 7
//   out = (C - A)/A = (4C - 2O - H - L) / (2O + 3C + H + L)
// Shares its NUMERATOR with stock_close_pos below; only the denominator differs (7A vs 7(H-L)).
// A sits close to C, so this is effectively the /C normalization — measured weaker on the
// tradeable next-intraday leg (within-industry t +0.93 vs +2.54). Carried anyway because the two
// together encode (H-L)/A, the range as a fraction of price, which a linear layer cannot form
// from either alone. Occupies the slot held reserved in v0.6.0.0, so STOCKNN_PARAMS, every
// offset and every .bin layout are UNCHANGED from v0.6.0.0 — that was the point of reserving it.
static inline float stock_close_vs_wap(float o, float h, float l, float c) {
    float denom = 2.f * o + 3.f * c + h + l;
    if (!(denom > 1e-9f)) return 0.f;
    return (4.f * c - 2.f * o - h - l) / denom;
}

// Intraday position of the close within today's bar (v0.6.0.0, StockNN today feature 15).
//   close_pos = (4C - 2O - H - L) / (7(H - L))  ==  2*(C-O)/(H-L) + CLV
// Measured within-industry rank IC vs next-day targets (144 syms x 1254 days, one obs/day):
//   overnight gap C->O  -0.0371 (t -10.90);  next intraday O->C  +0.0081 (t +2.54),
//   and +0.0129 (t +4.49) after orthogonalizing against the prior-day return, which carries
//   none of it (t -0.28). A strong close gaps down, then reverts during the next session —
//   the O->C leg is the one the fill model can reach.
// The raw O/H/L/C inputs are dollar amounts, so fc_today can already form this numerator;
// it cannot divide, so the scale-free normalization is what this feature actually adds.
// Denominator is (H-L), not C: with /C the O->C leg is not significant (t +0.93).
static inline float stock_close_pos(float o, float h, float l, float c) {
    float rng = h - l;
    if (!(rng > 1e-9f)) return 0.f;          // zero-range or invalid bar
    return (4.f * c - 2.f * o - h - l) / (7.f * rng);
}

// StockNN forward — weights[] is STOCKNN_PARAMS floats in the offset layout above.
// hist15x60 is row-major [15][60], row 0 = oldest.  today232 is [232].
// Output out48 is [48] = reshape of [12][4]; activations applied per column.
static void stock_forward(const float* W, const float* hist15x60,
                          const float* today232, float* out48) {
    float x[300], y[300], cat[STOCK_TODAY_IN], fc1[237], fc2[174], fc3[111];
    static_assert(STOCK_TODAY_IN >= STOCK_INJ_IN[13], "cat too small for inject concat");

    // Seed: hist[0][60] → 120
    sgemv_relu(W + STOCK_SEED_W, W + STOCK_SEED_B, hist15x60, x, 120, 60);

    // Inject layers i=0..13
    int xsz = 120;
    for (int i = 0; i < 14; i++) {
        int in_sz  = STOCK_INJ_IN[i];
        int out_sz = STOCK_INJ_OUT[i];
        memcpy(cat, x, xsz * sizeof(float));
        memcpy(cat + xsz, hist15x60 + (i + 1) * 60, 60 * sizeof(float));
        sgemv_relu(W + STOCK_INJ_W[i], W + STOCK_INJ_B[i], cat, y, out_sz, in_sz);
        memcpy(x, y, out_sz * sizeof(float));
        xsz = out_sz;  // 125, 130, ..., 190
    }

    // Today: cat(x[190], today[STOCK_TODAY_DIM]) = STOCK_TODAY_IN → 300
    memcpy(cat, x, 190 * sizeof(float));
    memcpy(cat + 190, today232, STOCK_TODAY_DIM * sizeof(float));
    sgemv_relu(W + STOCK_TODAY_W, W + STOCK_TODAY_B, cat, x, 300, STOCK_TODAY_IN);

    // Flat layers
    sgemv_relu(W + STOCK_FLAT1_W, W + STOCK_FLAT1_B, x, y, 300, 300);
    sgemv_relu(W + STOCK_FLAT2_W, W + STOCK_FLAT2_B, y, x, 300, 300);

    // Funnel
    sgemv_relu(W + STOCK_FC1_W, W + STOCK_FC1_B, x,   fc1, 237, 300);
    sgemv_relu(W + STOCK_FC2_W, W + STOCK_FC2_B, fc1, fc2, 174, 237);
    sgemv_relu(W + STOCK_FC3_W, W + STOCK_FC3_B, fc2, fc3, 111, 174);

    // Output (no activation yet)
    sgemv_only(W + STOCK_OUT_W, W + STOCK_OUT_B, fc3, out48, 48, 111);

    // Per-column activations: [buy_qty=ReLU, buy_price=Sig, sell_all=Sig, sell_qty=ReLU]
    for (int j = 0; j < 12; j++) {
        out48[4*j+0] = fmaxf(0.f, out48[4*j+0]);
        out48[4*j+1] = sigmoidf(out48[4*j+1]);
        out48[4*j+2] = sigmoidf(out48[4*j+2]);
        out48[4*j+3] = fmaxf(0.f, out48[4*j+3]);
    }
}

// MasterNN forward — weights[] is MASTERNN_PARAMS floats, today444 is (1,444) flat vector.
// Output out48: raw logits [12][4]; caller decodes tier via argmax over each group of 4.
// master_forward() removed in v0.8.1.10: dead since the 888-wide feature vector landed (-Wunused-function). MasterNN inference lives in production_v2.py, not the trainer.

// Single LSTM time step (one layer). gates[4*hidden] is caller-provided scratch.
static inline void lstm_step(const float* W_ih, const float* W_hh,
                              const float* b_ih, const float* b_hh,
                              const float* x_t, const float* h_prev, const float* c_prev,
                              float* h_out, float* c_out,
                              float* gates, int hidden, int input_sz) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4*hidden, input_sz,
                1.f, W_ih, input_sz, x_t, 1, 0.f, gates, 1);
    for (int j = 0; j < 4*hidden; j++) gates[j] += b_ih[j];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4*hidden, hidden,
                1.f, W_hh, hidden, h_prev, 1, 1.f, gates, 1);
    for (int j = 0; j < 4*hidden; j++) gates[j] += b_hh[j];
    for (int j = 0; j < hidden; j++) {
        float i_g = sigmoidf(gates[j]);
        float f_g = sigmoidf(gates[  hidden + j]);
        float g_g = tanhf   (gates[2*hidden + j]);
        float o_g = sigmoidf(gates[3*hidden + j]);
        c_out[j] = f_g * c_prev[j] + i_g * g_g;
        h_out[j] = o_g * tanhf(c_out[j]);
    }
}

// MT2NN forward: FC‖LSTM parallel fork → concat72 → taper → 48 raw logits.
// in12: one MT1 dollar prediction per industry (raw, no normalization — the magnitude IS the
// allocation signal). The LSTM reads them as a 12-step sequence of scalars.
static void mt2_forward(const float* W, const float* in12, float* out48) {
    // FC branch
    float fc1[36], fc2[36];
    sgemv_relu(W + MT2_FC1_W, W + MT2_FC1_B, in12, fc1, 36, 12);
    sgemv_relu(W + MT2_FC2_W, W + MT2_FC2_B, fc1,  fc2, 36, 36);

    // LSTM branch: 12 steps × 4 features, 2 layers, hidden=36
    float h1[36]={}, c1[36]={}, hn1[36], cn1[36];
    float h2[36]={}, c2[36]={}, hn2[36], cn2[36];
    float gates[4*36];
    for (int t = 0; t < 12; t++) {
        const float* x_t = in12 + t;
        lstm_step(W+MT2_L1_WIH, W+MT2_L1_WHH, W+MT2_L1_BIH, W+MT2_L1_BHH,
                  x_t, h1, c1, hn1, cn1, gates, 36, 1);
        memcpy(h1, hn1, 36*sizeof(float)); memcpy(c1, cn1, 36*sizeof(float));
        lstm_step(W+MT2_L2_WIH, W+MT2_L2_WHH, W+MT2_L2_BIH, W+MT2_L2_BHH,
                  hn1, h2, c2, hn2, cn2, gates, 36, 36);
        memcpy(h2, hn2, 36*sizeof(float)); memcpy(c2, cn2, 36*sizeof(float));
    }

    // Concatenate FC + LSTM final hidden
    float cat72[72];
    memcpy(cat72,      fc2, 36*sizeof(float));
    memcpy(cat72 + 36, h2,  36*sizeof(float));

    // Taper: 72→66→60→54→48
    float t1[66], t2[60], t3[54];
    sgemv_relu(W + MT2_T1_W, W + MT2_T1_B, cat72, t1, 66, 72);
    sgemv_relu(W + MT2_T2_W, W + MT2_T2_B, t1,    t2, 60, 66);
    sgemv_relu(W + MT2_T3_W, W + MT2_T3_B, t2,    t3, 54, 60);
    sgemv_only(W + MT2_OUT_W, W + MT2_OUT_B, t3, out48, 48, 54);
}

// ── Data structures ─────────────────────────────────────────────────────────────

struct OHLCV {
    float open, high, low, close, volume;
    bool  valid;
};

struct DayData {
    OHLCV sym[N_IND][IND_SYMS];
};

struct Portfolio {
    float cash;
    float holdings[IND_SYMS];
    float stop_prices[IND_SYMS];
};

struct MasterPortfolio {
    float cash;
    float holdings[N_IND];
};

// Circular history buffer: data[10] = [open,close,high,low,vol,d_open,d_close,d_high,d_low,d_vol]
struct HistEntry { float data[10]; };

struct SymHist {
    HistEntry buf[15];
    int       len{0}, head{0};

    const HistEntry& get(int t) const { return buf[(head + t) % 15]; }
    // get_from_newest(0)=newest, get_from_newest(t)=t steps ago
    const HistEntry* newest() const { return len > 0 ? &get(len - 1) : nullptr; }

    void push(const HistEntry& e) {
        if (len < 15) buf[len++] = e;
        else { buf[head] = e; head = (head + 1) % 15; }
    }
};

struct IndustryState {
    Portfolio portfolios[N_SLOTS];
    SymHist   hist[IND_SYMS];
    int       streak{0};
    // elites removed — stored in per-worker WorkerScratch to avoid OOM
};

// IND_HIST_CAP: maximum entries in ind_val_hist rolling buffer.
// Need up to lookback 91 days of history (delta at t=90 needs hist[90] and hist[91]).
static constexpr int IND_HIST_CAP = 92;

struct MasterState {
    MasterPortfolio portfolios[N_SLOTS];
    // Per-industry rolling value history: oldest-first, length ind_hist_count (≤ IND_HIST_CAP)
    float           ind_val_hist[N_IND][IND_HIST_CAP];
    // Parallel market cumulative-return index (starts at IND_STARTING_CASH each pass)
    float           mkt_val_hist[N_IND][IND_HIST_CAP];
    int             ind_hist_count{0};
    // Consecutive tier-0 counter per slot per industry; slot 0 persists across days
    int             zero_counts[N_SLOTS][N_IND];
    // Days remaining before another diversity injection is allowed (counts down each day)
    int             mt2_injection_hold{0};
    // elites removed — stored in MasterScratch (heap-allocated in main)
};

// ── Per-worker scratch (heap-allocated once per thread, ~162 MB total) ──────────

struct WorkerScratch {
    float*    elite_buf;    // [ELITE_POOL * STOCKNN_PARAMS] — loaded from disk per industry
    float*    new_elites;   // [ELITE_POOL * STOCKNN_PARAMS] — temp for selection reorder
    float*    wavg_buf;     // [3 * STOCKNN_PARAMS]
    float*    mut_buf;      // [STOCKNN_PARAMS]
    float*    hist_buf;     // [HIST_DAYS * HIST_PER_DAY * STOCKNN_PARAMS] — per-industry history (disk-backed)
    int       hist_head{0}; // circular write index into hist_buf, 0..HIST_DAYS-1
    int       hist_count{0};// days of valid history populated, 0..HIST_DAYS
    uint64_t  mut_seeds[N_SLOTS - ELITE_POOL];

    WorkerScratch() {
        size_t ep   = (size_t)ELITE_POOL * STOCKNN_PARAMS;
        size_t hist = (size_t)HIST_DAYS * HIST_PER_DAY * STOCKNN_PARAMS;
        elite_buf  = new float[ep]();
        new_elites = new float[ep]();
        wavg_buf   = new float[3 * STOCKNN_PARAMS]();
        mut_buf    = new float[STOCKNN_PARAMS]();
        hist_buf   = new float[hist]();
        // Pin only elite_buf (hot inference read path) — 70 MB per worker.
        // new_elites and hist_buf are write-heavy/sequential; swap-friendly.
        mlock(elite_buf, ep * sizeof(float));
    }
    ~WorkerScratch() {
        delete[] elite_buf; delete[] new_elites;
        delete[] wavg_buf;  delete[] mut_buf;
        delete[] hist_buf;
    }
    float* elite(int i)      { return elite_buf  + (size_t)i * STOCKNN_PARAMS; }
    float* new_elite(int i)  { return new_elites + (size_t)i * STOCKNN_PARAMS; }
    float* wavg(int i)       { return wavg_buf   + (size_t)i * STOCKNN_PARAMS; }
    float* hist(int d, int p){ return hist_buf + ((size_t)d * HIST_PER_DAY + p) * STOCKNN_PARAMS; }
};

struct MasterScratch {
    float*    elite_buf;
    float*    new_elites;
    float*    wavg_buf;
    float*    mut_buf;
    uint64_t  mut_seeds[N_SLOTS - ELITE_POOL];

    MasterScratch() {
        size_t ep = (size_t)ELITE_POOL * MASTERNN_PARAMS;
        elite_buf  = new float[ep]();
        new_elites = new float[ep]();
        wavg_buf   = new float[3 * MASTERNN_PARAMS]();
        mut_buf    = new float[MASTERNN_PARAMS]();
        // Master runs sequentially (not concurrent with workers); not mlock'd.
    }
    ~MasterScratch() {
        delete[] elite_buf; delete[] new_elites;
        delete[] wavg_buf;  delete[] mut_buf;
    }
    float* elite(int i)     { return elite_buf  + (size_t)i * MASTERNN_PARAMS; }
    float* new_elite(int i) { return new_elites + (size_t)i * MASTERNN_PARAMS; }
    float* wavg(int i)      { return wavg_buf   + (size_t)i * MASTERNN_PARAMS; }
};

// ── MT1/MT2 structures ──────────────────────────────────────────────────────────

// ── MT1 rebuild: one pool of persistent individuals per industry ─────────────────
//
// Replaces MT1Scratch's five pools, head/tail split, history rings, OOS snapshots and the
// direction pool's parallel machinery. 119 lines of state become ~30.
//
// CAUSALITY: a prediction made on day t is scored at t+1 against the realised t->t+1 P&L, using
// a baseline and floor computed from days <= t. mt1_windows_are_causal() asserts it. That is a
// one-day delay, the minimum possible; the old design needed 10 (and 20 for vol), which is what
// left a 10-day scoring window holding ~1.6 independent observations.
struct MT1PoolScratch {
    // Parameterised on the model width so MT1Net (3,501) and its competitor MT1CNet (11,177) share
    // one pool implementation. They must: the comparison is only meaningful if the two run under
    // identical selection, lifecycle and scoring, so the ONLY difference is the network and its
    // inputs. Allocation is deferred to alloc() because make_unique<T[]> cannot pass constructor
    // arguments.
    int         n_params{MT1NET_PARAMS};
    float*      pool{nullptr};              // [MT1_POOL_SLOTS x n_params], persistent
    float*      mut_buf{nullptr};           // one model, breeding scratch
    MT1SlotMeta meta[MT1_POOL_SLOTS]{};
    int         best_slot{0};               // deployed model; ranked in place, never shuffled
    uint32_t    next_lineage{1};
    uint32_t    barred[16]{};               // lineages over the cap, barred from breeding
    int         barred_count{0};

    // Yesterday's predictions, awaiting today's outcome. This IS the out-of-sample mechanism:
    // every model made its call before the outcome existed, so no snapshot machinery is needed.
    float       pending[MT1_POOL_SLOTS]{};
    bool        has_pending{false};
    int         pending_day{-1};

    // Trailing realised P&L. Feeds both the baseline (the predictor MT1 must beat) and the score
    // floor. Never contains the day being scored — see the causality contract.
    float       actual_buf[MT1_BASELINE_DAYS]{};
    int         actual_head{0}, actual_count{0};
    int         last_pushed_day{-1};

    uint32_t    retire_hist[5]{};           // 8-15 / 16-31 / 32-63 / 64-127 / 128+
    double      retire_age_sum{0.0};
    uint32_t    retire_n{0};
    int         culled_today{0}, births_today{0};

    MT1PoolScratch() { alloc(MT1NET_PARAMS); }
    void alloc(int np) {
        delete[] pool; delete[] mut_buf;
        n_params = np;
        pool    = new float[(size_t)MT1_POOL_SLOTS * np]();
        mut_buf = new float[np]();
    }
    ~MT1PoolScratch() { delete[] pool; delete[] mut_buf; }
    MT1PoolScratch(const MT1PoolScratch&) = delete;
    MT1PoolScratch& operator=(const MT1PoolScratch&) = delete;

    float* slot(int i) { return pool + (size_t)i * n_params; }
    const float* slot(int i) const { return pool + (size_t)i * n_params; }

    // Trailing mean of realised P&L — the naive predictor. Score 0.5 means tying this.
    float baseline() const {
        if (actual_count <= 0) return 0.f;
        float s = 0.f;
        for (int i = 0; i < actual_count; i++) s += actual_buf[i];
        return s / (float)actual_count;
    }
    // Floor on the score denominator. p10 of |actual| is ~$86 and the baseline's error can be
    // near zero on a day the trailing mean happens to land, which would make the score
    // hypersensitive. Uses the most recent MT1_FLOOR_DAYS entries.
    float floor_v() const {
        const int n = actual_count < MT1_FLOOR_DAYS ? actual_count : MT1_FLOOR_DAYS;
        if (n <= 0) return 1.f;
        float s = 0.f;
        for (int k = 0; k < n; k++) {
            const int idx = (actual_head - 1 - k + 2 * MT1_BASELINE_DAYS) % MT1_BASELINE_DAYS;
            s += fabsf(actual_buf[idx]);
        }
        return fmaxf(s / (float)n * MT1_FLOOR_FRAC, 1.f);
    }
    void push_actual(float v, int day) {
        actual_buf[actual_head] = v;
        actual_head = (actual_head + 1) % MT1_BASELINE_DAYS;
        if (actual_count < MT1_BASELINE_DAYS) actual_count++;
        last_pushed_day = day;
    }
};

// What one MT1 day produced. Reported as MEAN and slot-0, never as the pool max: max-of-N is
// in-sample even when every individual prediction is out-of-sample, which is exactly the artifact
// that made the StockNN control look like it won 94% of days.
struct MT1DayResult {
    float pred0{0};                       // the deployed model's call for tomorrow
    float actual{0};                      // realised P&L scored TODAY (yesterday's call)
    float baseline{0}, floor_v{0};        // what the score was measured against
    float score0{0}, score_mean{0}, score_best{0}, score_min{0};
    int   mature{0}, culled{0}, births{0};
    int   lineage_max{0}, lineage_n{0};   // monoculture read: largest lineage, distinct lineages
    bool  scored{false};                  // false on the first day of a pass (no outcome yet)
};


struct MT1Result {
    float best_score, slot0_score, mean_score, min_score;
    float slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4;  // composite slot0 activations for MT2
    float dir0_conf, dir0_delta_t, dir0_range_pct, dir0_conf4;      // direction-pool slot0 activations for MT2 (toggle)
    // per-component pool stats (direction, range, accuracy, confidence)
    float best_dir, slot0_dir, mean_dir, min_dir, mean_dir_cdbl;
    // Direction-pool lifecycle diagnostics (v0.5.0.0 forward accumulation). mean_dir_cdbl now
    // carries the pool mean TERTIARY blend; the old n_correct_dbl it held died with the replay path.
    float dir_mature, dir_culled, dir_lineage_max, dir_lineage_n, dir_mean_sec;
    float best_rng, slot0_rng, mean_rng, min_rng;
    float best_acc, slot0_acc, mean_acc, min_acc;
    float best_cfd, slot0_cfd, mean_cfd, min_cfd;
};


struct MT2Scratch {
    float*   elite_buf;   // [ELITE_POOL × MT2NN_PARAMS]
    float*   new_elites;  // [ELITE_POOL × MT2NN_PARAMS]
    float*   wavg_buf;    // [3 × MT2NN_PARAMS]
    float*   mut_buf;     // [MT2NN_PARAMS]
    float*   hist_buf;    // [HIST_DAYS * HIST_PER_DAY * MT2NN_PARAMS]
    int      hist_head{0};
    int      hist_count{0};
    uint64_t mut_seeds[N_SLOTS - ELITE_POOL];
    // Look-behind buffer for the reliability-weighted consensus diagnostic (in-RAM, transient).
    float    lb_in12[MT2_LB_DAYS][N_IND]{};
    float    lb_perf[MT2_LB_DAYS][N_IND]{};
    int      lb_head{0};
    int      lb_count{0};

    MT2Scratch() {
        size_t ep = (size_t)ELITE_POOL * MT2NN_PARAMS;
        elite_buf  = new float[ep]();
        new_elites = new float[ep]();
        wavg_buf   = new float[3 * MT2NN_PARAMS]();
        mut_buf    = new float[MT2NN_PARAMS]();
        hist_buf   = new float[(size_t)HIST_DAYS * HIST_PER_DAY * MT2NN_PARAMS]();
    }
    ~MT2Scratch() { delete[] elite_buf; delete[] new_elites; delete[] wavg_buf; delete[] mut_buf; delete[] hist_buf; }
    float* elite(int i)       { return elite_buf  + (size_t)i * MT2NN_PARAMS; }
    float* new_elite(int i)   { return new_elites + (size_t)i * MT2NN_PARAMS; }
    float* wavg(int i)        { return wavg_buf   + (size_t)i * MT2NN_PARAMS; }
    float* hist(int d, int p) { return hist_buf   + ((size_t)(d * HIST_PER_DAY + p)) * MT2NN_PARAMS; }
};

// ── Fee helpers ─────────────────────────────────────────────────────────────────

static inline float sell_net(float shares, float price) {
    float gross = shares * price;
    float fee   = gross * SEC_FEE_RATE + std::min(shares * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX);
    return gross - fee;
}

static float compute_value_ind(const Portfolio& p, const OHLCV* day, const OHLCV* fill) {
    float val = p.cash;
    for (int j = 0; j < IND_SYMS; j++) {
        if (p.holdings[j] > 0.f) {
            // Use fill price close (next-day); fall back to same-day close
            float price = (fill && fill[j].valid) ? fill[j].close :
                          (day  && day[j].valid)  ? day[j].close  : 0.f;
            val += p.holdings[j] * price;
        }
    }
    return val;
}

// ── Mutation: apply Gaussian noise in-place with PCG32 seed ───────────────────

static void apply_gaussian(float* W, int n, float sigma, uint64_t seed) {
    PCG32 rng; rng.seed(seed);
    float pair[2];
    int i = 0;
    for (; i + 1 < n; i += 2) {
        bm_pair(rng, sigma, pair);
        W[i]   += pair[0];
        W[i+1] += pair[1];
    }
    if (i < n) {
        bm_pair(rng, sigma, pair);
        W[i] += pair[0];
    }
}

// Initialize weights with Kaiming uniform: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
static void kaiming_init(float* W, int out, int in, PCG32& rng) {
    float bound = 1.0f / sqrtf((float)in);
    for (int i = 0; i < out * in; i++)
        W[i] = (rng.next_float() * 2.f - 1.f) * bound;
    // biases: same bound (PyTorch nn.Linear default)
    float bbound = 1.0f / sqrtf((float)in);
    for (int i = 0; i < out; i++)
        W[out * in + i] = (rng.next_float() * 2.f - 1.f) * bbound;
}

// Initialize all weights for a StockNN using Kaiming uniform per layer
static void init_stock_weights(float* W, PCG32& rng) {
    // seed
    kaiming_init(W + STOCK_SEED_W, 120, 60, rng);
    // inject
    for (int i = 0; i < 14; i++) {
        int in_sz = STOCK_INJ_IN[i], out_sz = STOCK_INJ_OUT[i];
        // weights then biases stored consecutively at STOCK_INJ_W[i] and STOCK_INJ_B[i]
        float bound = 1.0f / sqrtf((float)in_sz);
        for (int k = 0; k < out_sz * in_sz; k++)
            W[STOCK_INJ_W[i] + k] = (rng.next_float() * 2.f - 1.f) * bound;
        for (int k = 0; k < out_sz; k++)
            W[STOCK_INJ_B[i] + k] = (rng.next_float() * 2.f - 1.f) * bound;
    }
    kaiming_init(W + STOCK_TODAY_W, 300, STOCK_TODAY_IN, rng);
    kaiming_init(W + STOCK_FLAT1_W, 300, 300, rng);
    kaiming_init(W + STOCK_FLAT2_W, 300, 300, rng);
    kaiming_init(W + STOCK_FC1_W,   237, 300, rng);
    kaiming_init(W + STOCK_FC2_W,   174, 237, rng);
    kaiming_init(W + STOCK_FC3_W,   111, 174, rng);
    kaiming_init(W + STOCK_OUT_W,    48, 111, rng);
}

// init_master_weights() removed in v0.8.1.10: its only caller was load_or_init_master(), also removed (-Wunused-function).

static void init_mt2_weights(float* W, PCG32& rng) {
    kaiming_init(W + MT2_FC1_W, 36, 12, rng);
    kaiming_init(W + MT2_FC2_W, 36, 36, rng);
    // LSTM: PyTorch default — Uniform(-1/sqrt(hidden), 1/sqrt(hidden)) for all params
    float lb = 1.f / sqrtf(36.f);
    for (int j = 0; j < 4*36* 4; j++) W[MT2_L1_WIH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36*36; j++) W[MT2_L1_WHH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36;    j++) W[MT2_L1_BIH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36;    j++) W[MT2_L1_BHH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36*36; j++) W[MT2_L2_WIH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36*36; j++) W[MT2_L2_WHH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36;    j++) W[MT2_L2_BIH+j] = (rng.next_float()*2.f-1.f)*lb;
    for (int j = 0; j < 4*36;    j++) W[MT2_L2_BHH+j] = (rng.next_float()*2.f-1.f)*lb;
    kaiming_init(W + MT2_T1_W, 66, 72, rng);
    kaiming_init(W + MT2_T2_W, 60, 66, rng);
    kaiming_init(W + MT2_T3_W, 54, 60, rng);
    kaiming_init(W + MT2_OUT_W, 48, 54, rng);
}

// ── Model file I/O ──────────────────────────────────────────────────────────────

static bool load_bin(const std::string& path, float* W, int n_params) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    size_t got = fread(W, sizeof(float), n_params, f);
    fclose(f);
    return (int)got == n_params;
}

static bool save_bin(const std::string& path, const float* W, int n_params) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t wrote = fwrite(W, sizeof(float), n_params, f);
    fclose(f);
    return (int)wrote == n_params;
}

static void log_msg(const std::string& msg);   // defined below, in the Logging section

// A crashed production run must not stall training forever, so a lock is honoured only while its
// holder is alive AND it is recent. Overrides are logged — silently ignoring a lock would be the
// worse failure of the two.
static bool trade_lock_active() {
    struct stat st;
    if (stat(g_trade_lock.c_str(), &st) != 0) return false;      // no lock: run freely

    if (time(nullptr) - st.st_mtime > TRADE_LOCK_STALE_SEC) {
        log_msg("WARNING: trading lock " + g_trade_lock + " is older than " +
                std::to_string(TRADE_LOCK_STALE_SEC / 60) + " min — ignoring it as stale");
        return false;
    }
    long pid = 0;
    if (FILE* f = fopen(g_trade_lock.c_str(), "r")) {
        if (fscanf(f, "%ld", &pid) != 1) pid = 0;
        fclose(f);
    }
    if (pid > 0 && kill((pid_t)pid, 0) != 0 && errno == ESRCH) {
        log_msg("WARNING: trading lock holder pid " + std::to_string(pid) +
                " is gone — ignoring stale lock " + g_trade_lock);
        return false;
    }
    return true;
}

// Called between training days, never mid-day: the worker threads are parked on their semaphore
// at that point, so sleeping here idles all of them.
static void wait_while_trading() {
    bool paused = false;
    while (trade_lock_active()) {
        if (!paused) {
            log_msg("PAUSED — production is running (" + g_trade_lock + ")");
            paused = true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(TRADE_LOCK_POLL_SEC));
    }
    if (paused) log_msg("RESUMED — production finished, training continues");
}

static std::string elite_path(const std::string& dir, const char* prefix, int slot) {
    return dir + "/" + prefix + "_elite_" + std::to_string(slot) + ".bin";
}

// ── Logging ────────────────────────────────────────────────────────────────────

static std::mutex g_log_mutex;

static void log_msg(const std::string& msg) {
    auto now  = std::chrono::system_clock::now();
    auto t    = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char ts[16];
    snprintf(ts, sizeof(ts), "%02d:%02d:%02d", tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    std::lock_guard<std::mutex> lk(g_log_mutex);
    printf("[%s] %s\n", ts, msg.c_str());
    fflush(stdout);
}

// ── JSON stock data loader ──────────────────────────────────────────────────────
// Each file: {"days": [{"date":"YYYY-MM-DD","open":f,"high":f,"low":f,"close":f,"volume":f},...]}

// Bars rejected as non-finite or non-positive during load. Reported after loading so a data
// problem is visible instead of silent — being silent is what made the SCCO case expensive.
static long g_bad_bars = 0;

// (int)NaN is UB and lands on INT32_MIN here. Belt-and-braces for anything that formats a
// float as an int; the real fix is rejecting the bar above.
static inline int safe_int(float v) { return std::isfinite(v) ? (int)v : 0; }

// sort_index_prefix() lives in sort_util.h -- see there for why it is not std::sort.

static float parse_float_after(const char* buf, const char* key, float def = 0.f) {
    const char* p = strstr(buf, key);
    if (!p) return def;
    p = strchr(p, ':');
    if (!p) return def;
    return (float)atof(p + 1);
}

static bool load_sym_data(const std::string& path,
                          std::map<std::string, OHLCV>& out_map) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    const char* buf = content.c_str();

    // Find "days"
    const char* p = strstr(buf, "\"days\"");
    if (!p) return false;
    p = strchr(p, '[');
    if (!p) return false;
    p++;

    while (*p) {
        // skip to next '{'
        while (*p && *p != '{' && *p != ']') p++;
        if (!*p || *p == ']') break;
        p++;
        // find "date"
        const char* date_p = strstr(p, "\"date\"");
        if (!date_p) break;
        date_p = strchr(date_p, ':');
        if (!date_p) break;
        date_p = strchr(date_p, '"');
        if (!date_p) break;
        date_p++;
        const char* date_end = strchr(date_p, '"');
        if (!date_end) break;
        std::string date(date_p, date_end - date_p);

        // find closing '}' for this entry
        const char* entry_end = strchr(p, '}');
        if (!entry_end) break;
        std::string entry(p, entry_end - p + 1);

        OHLCV o;
        o.open   = parse_float_after(entry.c_str(), "\"open\"");
        o.high   = parse_float_after(entry.c_str(), "\"high\"");
        o.low    = parse_float_after(entry.c_str(), "\"low\"");
        o.close  = parse_float_after(entry.c_str(), "\"close\"");
        o.volume = parse_float_after(entry.c_str(), "\"volume\"");
        // A bar can arrive non-finite: yfinance returns NaN for a missing session and
        // json.dump writes it as a bare `NaN` literal, which atof() parses happily. Marking
        // such a bar valid let NaN into the portfolio valuation, and `(int)NaN` is undefined
        // behaviour — observed as `prod=$-2147483648` (INT32_MIN) for SCCO 2026-08-11. It is
        // not cosmetic: it silently destroys every statistic computed over that
        // industry-pass (a quarter return came out as -4,047,374%).
        o.valid = std::isfinite(o.open) && std::isfinite(o.high) &&
                  std::isfinite(o.low)  && std::isfinite(o.close) &&
                  o.open > 0.f && o.high > 0.f && o.low > 0.f && o.close > 0.f &&
                  o.high >= o.low;
        if (!std::isfinite(o.volume) || o.volume < 0.f) o.volume = 0.f;
        if (!o.valid) g_bad_bars++;
        out_map[date] = o;
        p = entry_end + 1;
    }
    return !out_map.empty();
}

// ── Selection + mutation ────────────────────────────────────────────────────────

// Normalize values: clip negatives to 0, divide by sum; returns equal weights if all zero.
static void normalize_weights(const float* vals, float* out, int n) {
    float total = 0.f;
    for (int i = 0; i < n; i++) total += std::max(0.f, vals[i]);
    if (total <= 0.f) { for (int i = 0; i < n; i++) out[i] = 1.f / n; return; }
    for (int i = 0; i < n; i++) out[i] = std::max(0.f, vals[i]) / total;
}

// Weighted average of elites (flat layout: elite i at elite_buf + i*n_params).
static void wavg_weights_flat(const float* elite_buf, int n_params,
                               const int* src_ranks, const float* weights, int n,
                               float* dst) {
    memset(dst, 0, n_params * sizeof(float));
    for (int k = 0; k < n; k++) {
        const float* src = elite_buf + (size_t)src_ranks[k] * n_params;
        float w = weights[k];
        for (int j = 0; j < n_params; j++) dst[j] += src[j] * w;
    }
}

static void wavg_portfolio(const Portfolio* ports[], const float* weights, int n,
                           Portfolio& dst) {
    dst = Portfolio{};
    for (int k = 0; k < n; k++) {
        dst.cash += ports[k]->cash * weights[k];
        for (int j = 0; j < IND_SYMS; j++) {
            dst.holdings[j]    += ports[k]->holdings[j]    * weights[k];
            dst.stop_prices[j] += ports[k]->stop_prices[j] * weights[k];
        }
    }
}

static void wavg_mst_portfolio(const MasterPortfolio* ports[], const float* weights,
                                int n, MasterPortfolio& dst) {
    dst = MasterPortfolio{};
    for (int k = 0; k < n; k++) {
        dst.cash += ports[k]->cash * weights[k];
        for (int j = 0; j < N_IND; j++)
            dst.holdings[j] += ports[k]->holdings[j] * weights[k];
    }
}

struct IndResult {
    // The three marks that decompose a day. All on slot 0 (the deployed model):
    //   book_prev   = reference book (yesterday's holdings, no trades) at TODAY's close
    //   baseline    = the same holdings at the NEXT day's close  -> price move, no trading
    //   slot0_score = the book AFTER today's trades at the NEXT day's close
    // so that
    //   market move = baseline    - book_prev     (prices moved, holdings fixed)
    //   trade delta = slot0_score - baseline      (holdings moved, prices fixed)
    //   book P&L    = slot0_score - book_prev     = market + trade, exactly
    // Only the trade delta was available before, and it is the component that cancels beta --
    // which is why the target built on it was zero-mean, fat-tailed and unpredictable.
    float book_prev;
    // ── slot-0 ORDER INTENT ──────────────────────────────────────────────────
    // What the DEPLOYED model decided to do, before any of it filled.
    //
    // CAUSAL, and the distinction matters: buy_exec/sell_exec count what actually FILLED, and a
    // fill is decided by `nd_open/nd_low/nd_high` — the NEXT day's bar. Using executed volume as a
    // feature is look-ahead, and in production it does not exist yet because orders are placed
    // after the allocation decision. Intent is different: the limit prices are
    // `low_t + frac * span_t`, anchored to TODAY's bar alone, and the quantities come straight off
    // StockNN's forward pass on today's data.
    //
    // This is also the only thing available to MT1 that carries per-symbol information. The 74
    // features are built from two scalar value curves; StockNN is a 928,825-parameter model that
    // has already read all 12 symbols, so its order intent is a learned compression of exactly the
    // data MT1 cannot see.
    //
    // The price fracs are additionally CAPITAL-INDEPENDENT — `buy_price_frac` is a raw network
    // output scaled by today's range — so a production run can obtain them from a forward pass at
    // any capital, before allocation, at no extra cost.
    float oi_n_buy, oi_n_sell;        // symbols (of IND_SYMS) carrying a buy / sell order
    float oi_buy_aggr, oi_sell_aggr;  // mean (limit − close_t)/span_t over ordering symbols;
                                      // > 0 = limit above today's close (wants to fill now)
    float oi_buy_disp;                // sd of buy_price_frac across ordering symbols
    float oi_buy_val, oi_sell_val;    // intended buy / sell in dollars at today's close
    // Every scored elite's post-trade value at the next close, so the online conditional
    // models can consume all of them rather than only the deployed slot. They all start
    // from slot 0's portfolio, so they share the market move and differ only in trades.
    float elite_score[ELITE_POOL];

    // ── per-model order intent and outcome, for every scored slot ────────────────
    //
    // The si_* arrays above are SLOT 0's intent. That is the right thing for the dataset log and
    // for MT1Net's feature slice, but it is the wrong thing to train a gradient MT1 on: handing it
    // slot 0's orders and asking for elite #7's realised P&L pairs an input with an outcome that a
    // DIFFERENT model produced. For 19 of 20 elites that input describes orders nobody placed.
    //
    // With per-model intent the 200 rows stop being 200 targets against one input and become 200
    // distinct pairs -- the input now varies exactly where the outcome varies. That is also what
    // makes ranking candidate order sets possible at all: the across-model outcome spread is
    // ~$166/industry-day, and nothing in a shared input can explain a single dollar of it.
    float slot_score[N_SLOTS];                 // scored book, every slot (elite_score is the first 20)
    // Per-SYMBOL decomposition of each slot's book P&L. MT1S-stock trains one row per symbol-day
    // rather than per industry-day, which is what turns ~855 industry-days into ~10,260 rows and
    // attacks the sample-size ceiling the variant race exposed. The twelve entries sum EXACTLY to
    // slot_score[s] - book_prev; tests/test_mt1_grad.cpp's twin in the C++ suite pins that, because
    // a decomposition that silently fails to close would train the per-stock model on a residual
    // that belongs to no stock.
    float slot_sym_pnl[N_SLOTS][IND_SYMS];
    float slot_bqty [N_SLOTS][IND_SYMS];
    float slot_bfrac[N_SLOTS][IND_SYMS];
    float slot_sfrac[N_SLOTS][IND_SYMS];
    float slot_sqty [N_SLOTS][IND_SYMS];
    // Per-symbol slot-0 state and intent, for the MT1CNet competitor's 146-wide input. Holdings
    // and cash are the PRE-trade reference book — what is actually held at today's close, since
    // today's orders do not fill until tomorrow.
    float ref_hold[IND_SYMS], ref_cash_v;
    float si_bqty[IND_SYMS], si_bfrac[IND_SYMS], si_sfrac[IND_SYMS], si_sqty[IND_SYMS];
    float baseline, slot0_score, best_delta;
    float top_hold, top_cash;
    int   new_streak;
    float elite_max_val, elite_min_val, elite_mean_val;
    // Rechenberg's 1/5 statistic: fraction of the 180 mutations that BEAT THEIR OWN PARENT.
    // Read-only instrumentation — nothing selects on it. It answers two open questions at once:
    // whether sigma is in the usable band on the CURRENT architecture (the 0.0055-0.009 range was
    // measured many versions and one breaking param change ago), and whether the regime-optimal
    // sigma moves. Far above 1/5 means steps are too small and the pool is degenerate; far below
    // means most mutations are damage and selection is picking survivors of noise.
    float mut_success;
    // --control-random: best/mean delta over ctl_n freshly random models scored on THIS day
    // against THIS day's reference portfolio. ctl_n == 0 means the control did not run.
    float ctl_best, ctl_mean;
    int   ctl_n;
    // Slot-0's book at the end of the day: the per-symbol share counts and cash of the model
    // that is actually deployed. We already log the portfolio's VALUE (prod=$) but never its
    // COMPOSITION, and without composition the only market-side regressor available is the
    // equal-weight industry index — while the book may hold 60% in one name. That mismatch is
    // why the P&L decomposition came back at R^2 ~ 0.0001: a broken regressor, not a finding.
    float hold[IND_SYMS];
    float hold_cash;
};

struct MasterResult {
    float best_pts, elite_max_pts, elite_min_pts, elite_mean_pts, ideal_pts;
    float consensus_flat_pts, consensus_wtd_pts;  // pool-consensus diagnostic (not fed into training)
    // Dual-graded deployed slot-0 pts (set in process_block, default 0 from aggregate returns):
    //   slot0_pts_pf  = graded on the slot-0 StockNN portfolio delta (the TRAINED objective)
    //   slot0_pts_mkt = graded on the market forward return (the old coincident proxy, diagnostic)
    float slot0_pts_pf, slot0_pts_mkt;
};

// ── Forward declarations (needed because step_industry calls load/save defined later) ──
static void load_or_init_industry(const std::string& dir, const std::string& load_dir,
                                   int ind_i, float* elite_buf, int actual_day);
static void save_industry_elites(const std::string& dir, int ind_i, const float* elite_buf);
static void load_ind_history(const std::string& dir, int ind_i, WorkerScratch& scratch);
static void save_ind_history(const std::string& dir, int ind_i, const WorkerScratch& scratch);

// ── step_industry ───────────────────────────────────────────────────────────────

static IndResult step_industry(int ind_i, IndustryState& state,
                               WorkerScratch& scratch,
                               const std::string& models_dir,
                               const std::string& load_dir,
                               const DayData& day, const DayData* fill,
                               int actual_day, int total_avail,
                               int day_num, int num_days,
                               float sigma, bool freeze, const bool* seq_flags) {
    // Load this industry's elites from disk (or random init on first day)
    load_or_init_industry(models_dir, load_dir, ind_i, scratch.elite_buf, actual_day);
    // Load per-industry elite history (or reset at pass start)
    if (day_num == 0) {
        scratch.hist_head  = 0;
        scratch.hist_count = 0;
    } else {
        load_ind_history(models_dir, ind_i, scratch);
    }
    const OHLCV* day_sym  = day.sym[ind_i];
    const OHLCV* fill_sym = fill ? fill->sym[ind_i] : day_sym;
    // Compute num_past = minimum history length across symbols
    int num_past = HIST_WINDOW;
    for (int j = 0; j < IND_SYMS; j++)
        num_past = std::min(num_past, state.hist[j].len);

    if (day_num % 10 == 0 || day_num == num_days - 1)
        log_msg(std::string("[") + IND_SHORT[ind_i] + "] Day " +
                std::to_string(actual_day + 1) + "/" + std::to_string(total_avail) +
                " — running 200 models + " + std::to_string(scratch.hist_count * HIST_PER_DAY) +
                " history (ohlcv_hist=" + std::to_string(num_past) + "/15 days warm)");

    // ── Baseline: slot 0 valued at fill prices with no trading ──────────────
    float ref_cash = state.portfolios[0].cash;
    float ref_hold[IND_SYMS], ref_stop[IND_SYMS];
    for (int j = 0; j < IND_SYMS; j++) {
        ref_hold[j] = state.portfolios[0].holdings[j];
        ref_stop[j] = state.portfolios[0].stop_prices[j];
    }
    float baseline = state.portfolios[0].cash;
    for (int j = 0; j < IND_SYMS; j++) {
        float price = fill_sym[j].valid ? fill_sym[j].close :
                      day_sym[j].valid  ? day_sym[j].close  : 0.f;
        baseline += ref_hold[j] * price;
    }
    // Same holdings, same cash, marked at TODAY's close instead of the fill close. The difference
    // between the two is the overnight price move on what is already held -- the component the
    // trade delta cancels. Falls back the other way (day -> fill) so a missing bar degrades to the
    // same price on both sides and contributes 0 to the move rather than the position's whole value.
    float book_prev = state.portfolios[0].cash;
    for (int j = 0; j < IND_SYMS; j++) {
        float price = day_sym[j].valid  ? day_sym[j].close :
                      fill_sym[j].valid ? fill_sym[j].close : 0.f;
        book_prev += ref_hold[j] * price;
    }

    // Reset all 200 portfolios to slot 0's state
    for (int s = 0; s < N_SLOTS; s++) {
        state.portfolios[s].cash = ref_cash;
        for (int j = 0; j < IND_SYMS; j++) {
            state.portfolios[s].holdings[j]    = ref_hold[j];
            state.portfolios[s].stop_prices[j] = ref_stop[j];
        }
    }

    // ── Pre-compute rolling stats ────────────────────────────────────────────
    struct SymStats {
        float hi15, lo15, avg_c, avg_v, avg_dv, volatility;
    };
    SymStats sym_stats[IND_SYMS];

    for (int j = 0; j < IND_SYMS; j++) {
        const SymHist& h = state.hist[j];
        if (h.len >= 2) {
            float hi15 = -1e30f, lo15 = 1e30f;
            float sum_c = 0.f, sum_v = 0.f, sum_dv = 0.f;
            for (int t = 0; t < h.len; t++) {
                const float* d = h.get(t).data;
                // data: [open, close, high, low, vol, ...]
                if (d[2] > hi15) hi15 = d[2];
                if (d[3] < lo15) lo15 = d[3];
                sum_c  += d[1];
                sum_v  += d[4];
                sum_dv += d[0] * d[4];
            }
            float avg_c  = sum_c  / h.len;
            float avg_v  = sum_v  / h.len;
            float avg_dv = sum_dv / h.len;
            if (avg_v  <= 0.f) avg_v  = 1.f;
            if (avg_dv <= 0.f) avg_dv = 1.f;
            float var_c = 0.f;
            for (int t = 0; t < h.len; t++) {
                float dc = h.get(t).data[1] - avg_c;
                var_c += dc * dc;
            }
            float std_c = sqrtf(var_c / h.len);
            sym_stats[j] = {hi15, lo15, avg_c > 0.f ? avg_c : 1.f,
                            avg_v, avg_dv, avg_c > 0.f ? std_c / avg_c : 0.f};
        } else {
            sym_stats[j] = {1.f, 0.f, 1.f, 1.f, 1.f, 0.f};
        }
    }

    // ── Build shared input arrays ────────────────────────────────────────────
    // history_arr: [15][60] — row 0 = oldest, row 14 = newest
    float history_arr[15 * 60] = {};
    for (int t = 14; t >= 0; t--) {
        int row_idx = 14 - t;   // row 0 when t=14 (oldest)
        for (int j = 0; j < IND_SYMS; j++) {
            const SymHist& h = state.hist[j];
            if (h.len > t) {
                // entry at position (len-1-t) from oldest = t steps back from newest
                const float* entry = h.get(h.len - 1 - t).data;
                for (int k = 0; k < 5; k++)
                    history_arr[row_idx * 60 + j * 5 + k] = entry[k];
            }
            // else: already zero from init
        }
    }

    // today_arr: [232]
    // per sym: 5 raw + 5 delta + 5 normalized = 15 × 12 = 180
    // + 15 cross-sym delta aggs + 13 state
    // Slot 16: reserved in v0.6.0.0, filled in v0.6.1.0 with close_vs_wap. Filling it changed
    // no dimension, no offset and no file format — which is exactly what reserving it bought.
    static constexpr int STOCK_CLOSE_WAP = 16;

    // Section offsets into today_arr. Per-symbol blocks occupy [0, TODAY_AGG_OFF).
    static constexpr int TODAY_PER_SYM   = 17;                          // v0.6.0.0: was 15
    static constexpr int TODAY_AGG_OFF   = IND_SYMS * TODAY_PER_SYM;    // 204
    static constexpr int TODAY_STATE_OFF = TODAY_AGG_OFF + 15;          // 219
    static_assert(TODAY_STATE_OFF + 1 + IND_SYMS == STOCK_TODAY_DIM, "today_arr layout drift");
    float today_arr[STOCK_TODAY_DIM] = {};
    float today_dl[IND_SYMS][5] = {};  // raw delta per sym, for cross-sym aggs

    for (int j = 0; j < IND_SYMS; j++) {
        const OHLCV& d = day_sym[j];
        float raw_t[5] = {d.valid ? d.open   : 0.f,
                          d.valid ? d.close  : 0.f,
                          d.valid ? d.high   : 0.f,
                          d.valid ? d.low    : 0.f,
                          d.valid ? d.volume : 0.f};
        const HistEntry* prev = state.hist[j].newest();
        float dlt_t[5] = {};
        if (prev) for (int k = 0; k < 5; k++) dlt_t[k] = raw_t[k] - prev->data[k];
        for (int k = 0; k < 5; k++) today_dl[j][k] = dlt_t[k];

        const SymStats& st = sym_stats[j];
        float rng_15 = std::max(st.hi15 - st.lo15, 1e-9f);
        int base = j * TODAY_PER_SYM;
        // raw(5)
        for (int k = 0; k < 5; k++) today_arr[base + k]     = raw_t[k];
        // delta(5)
        for (int k = 0; k < 5; k++) today_arr[base + 5 + k] = dlt_t[k];
        // normalized(7)
        today_arr[base + 10] = (raw_t[1] - st.lo15) / rng_15;
        today_arr[base + 11] = raw_t[1] / st.avg_c;
        today_arr[base + 12] = st.volatility;
        today_arr[base + 13] = raw_t[4] / st.avg_v;
        today_arr[base + 14] = (raw_t[0] * raw_t[4]) / st.avg_dv;
        today_arr[base + 15] = stock_close_pos(raw_t[0], raw_t[2], raw_t[3], raw_t[1]);
        today_arr[base + STOCK_CLOSE_WAP] = stock_close_vs_wap(raw_t[0], raw_t[2], raw_t[3], raw_t[1]);
    }
    // Cross-sym aggs: for each of 5 delta channels: max, min, mean
    for (int k = 0; k < 5; k++) {
        float mx = today_dl[0][k], mn = today_dl[0][k], sm = 0.f;
        for (int j = 0; j < IND_SYMS; j++) {
            if (today_dl[j][k] > mx) mx = today_dl[j][k];
            if (today_dl[j][k] < mn) mn = today_dl[j][k];
            sm += today_dl[j][k];
        }
        today_arr[TODAY_AGG_OFF + k * 3 + 0] = mx;
        today_arr[TODAY_AGG_OFF + k * 3 + 1] = mn;
        today_arr[TODAY_AGG_OFF + k * 3 + 2] = sm / IND_SYMS;
    }
    // State: [cash, holdings[12]]
    today_arr[TODAY_STATE_OFF] = ref_cash;
    for (int j = 0; j < IND_SYMS; j++) today_arr[TODAY_STATE_OFF + 1 + j] = ref_hold[j];

    // ── Inference + trade loop ────────────────────────────────────────────────
    float slot_scores[N_SLOTS];
    int   trade_count[N_SLOTS] = {};
    float buy_exec = 0.f, sell_exec = 0.f;

    // Use per-worker scratch buffers (no thread-local statics)
    float*    mut_buf   = scratch.mut_buf;
    uint64_t* mut_seeds = scratch.mut_seeds;

    // Assign mutation seeds at start of day
    {
        PCG32 seed_rng;
        seed_rng.seed(mix_seed((uint64_t)actual_day * 1000007ULL + (uint64_t)ind_i * 13ULL));
        for (int i = 0; i < N_SLOTS - ELITE_POOL; i++)
            mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    // slot-0 order intent, accumulated on the slot == 0 iteration only
    float si_bq[IND_SYMS] = {}, si_bf[IND_SYMS] = {};
    float si_sf[IND_SYMS] = {}, si_sq[IND_SYMS] = {};
    // ...and every slot's, for the gradient MT1. static because N_SLOTS x IND_SYMS x 4 floats is
    // 38 KB and step_industry runs on worker threads with modest stacks; one per thread.
    //
    // ZEROED EVERY CALL, and that is load-bearing. The symbol loop below skips invalid bars
    // (`if (!day_sym[j].valid) continue;`) BEFORE it writes, so a surviving static would hand the
    // next industry whatever the previous one left in that slot. CN_BQTY/CN_SQTY would survive it
    // (both are gated on `ok`), but CN_BFRAC/CN_SFRAC are not gated, so stale intent from another
    // industry would land in the feature vector. The old slot-0 arrays were plain zero-initialised
    // locals, which is where that safety came from.
    static thread_local float res_sym_pnl[N_SLOTS][IND_SYMS];
    memset(res_sym_pnl, 0, sizeof(res_sym_pnl));
    static thread_local float res_slot_bq[N_SLOTS][IND_SYMS], res_slot_bf[N_SLOTS][IND_SYMS];
    static thread_local float res_slot_sf[N_SLOTS][IND_SYMS], res_slot_sq[N_SLOTS][IND_SYMS];
    memset(res_slot_bq, 0, sizeof(res_slot_bq));
    memset(res_slot_bf, 0, sizeof(res_slot_bf));
    memset(res_slot_sf, 0, sizeof(res_slot_sf));
    memset(res_slot_sq, 0, sizeof(res_slot_sq));
    int   oi_nb = 0, oi_ns = 0;
    double oi_ba = 0.0, oi_sa = 0.0, oi_bv = 0.0, oi_sv = 0.0;
    double oi_bf = 0.0, oi_bf2 = 0.0;

    float out48[48];
    for (int slot = 0; slot < N_SLOTS; slot++) {
        Portfolio& port = state.portfolios[slot];

        // Select weights
        const float* W;
        if (slot < ELITE_POOL) {
            W = scratch.elite(slot);
        } else {
            int mut_i  = slot - ELITE_POOL;
            int parent = mut_i / MUTATIONS_PER_PARENT;
            memcpy(mut_buf, scratch.elite(parent), STOCKNN_PARAMS * sizeof(float));
            apply_gaussian(mut_buf, STOCKNN_PARAMS, sigma, mut_seeds[mut_i]);
            W = mut_buf;
        }

        stock_forward(W, history_arr, today_arr, out48);

        float local_buy = 0.f, local_sell = 0.f;
        // Cash moved BY each symbol: negative when this symbol's buy consumed cash, positive when
        // its sell returned some. Every cash mutation below happens inside the per-symbol loop, so
        // the attribution is exact rather than apportioned.
        float sym_cash[IND_SYMS] = {};

        // ── Phase 1: partial sells, gap sell_all, high-first sell_all, stops, buys ─
        for (int j = 0; j < IND_SYMS; j++) {
            if (!day_sym[j].valid) continue;
            float buy_qty            = out48[4*j+0];
            float buy_price_frac     = out48[4*j+1];
            float sell_all_price_frac = out48[4*j+2];
            float sell_qty           = out48[4*j+3];

            float low_t  = day_sym[j].low;
            float high_t = day_sym[j].high;
            float span_t = std::max(high_t - low_t, 1e-9f);
            float sell_all_price = low_t + sell_all_price_frac * span_t;
            float buy_price      = low_t + buy_price_frac * span_t;
            float stop_loss      = buy_price * 0.9f;

            // Record the deployed model's INTENT before anything is matched against next-day
            // prices. Deliberately placed here, above the fill logic, so it cannot pick up
            // nd_open/nd_low/nd_high by accident.
            // Every slot's intent is recorded; the slot-0 aggregates below are a separate
            // consumer (the dataset log and MT1Net's feature slice) and stay slot-0 only.
            res_slot_bq[slot][j] = buy_qty;              res_slot_bf[slot][j] = buy_price_frac;
            res_slot_sf[slot][j] = sell_all_price_frac;  res_slot_sq[slot][j] = sell_qty;
            if (slot == 0) {
                const float close_t = day_sym[j].close;
                si_bq[j] = buy_qty;  si_bf[j] = buy_price_frac;
                si_sf[j] = sell_all_price_frac; si_sq[j] = sell_qty;
                // The aggregates below are features too, so a non-finite raw output must not
                // propagate into them. TWO independent guards on purpose: the comparisons
                // reject NaN on their own, and the explicit isfinite() catches the infinities
                // that comparisons let through. The sell branch used to carry only the
                // isfinite(), which -ffast-math folded to `true` -- measured, oi_sell_aggr came
                // back NaN on 98.55% of industry-days while its buy twin, which also had a
                // `> 0.f`, was clean. See the -fno-finite-math-only note in CMakeLists.txt.
                const bool bar_ok = std::isfinite(close_t) && std::isfinite(span_t)
                                    && span_t > 0.f && close_t > 0.f;
                // Intent is bounded by what could actually EXECUTE -- a buy by available cash,
                // a sell by the position held -- and floored to whole shares, because Alpaca
                // forbids fractional quantities. Unbounded, these are raw network outputs:
                // measured, they summed to $1.5e12 of "intended buys" against a $25k book.
                // The COUNTS are deliberately not bounded: an order the model cannot afford was
                // still an order it chose to place, and oi_n_buy/oi_n_sell is the buy/sell
                // ratio channel. A bounded quantity of 0 against a nonzero count is the honest
                // encoding of "wanted to, could not".
                const float afford_q = bar_ok ? whole_shares(ref_cash / close_t) : 0.f;
                const float held_q   = whole_shares(ref_hold[j]);
                if (bar_ok && buy_qty > 1e-6f && buy_price > 0.f && std::isfinite(buy_price)) {
                    oi_nb++;
                    oi_ba  += (double)(buy_price - close_t) / span_t;
                    oi_bv  += (double)whole_shares(fminf(buy_qty, afford_q)) * close_t;
                    oi_bf  += buy_price_frac;
                    oi_bf2 += (double)buy_price_frac * buy_price_frac;
                }
                if (bar_ok && sell_qty > 1e-6f && sell_all_price > 0.f
                        && std::isfinite(sell_all_price)) {
                    oi_ns++;
                    oi_sa += (double)(sell_all_price - close_t) / span_t;
                    oi_sv += (double)whole_shares(fminf(sell_qty, held_q)) * close_t;
                }
            }

            // Fill day data (next-day)
            float nd_open = fill_sym[j].valid ? fill_sym[j].open  : day_sym[j].close;
            float nd_low  = fill_sym[j].valid ? fill_sym[j].low   : day_sym[j].low;
            float nd_high = fill_sym[j].valid ? fill_sym[j].high  : day_sym[j].high;

            bool low_first = seq_flags[ind_i * IND_SYMS + j];

            // Partial sell at open
            if (sell_qty > 1e-6f && port.holdings[j] > 1e-6f) {
                float amt = whole_shares(std::min(sell_qty, port.holdings[j]));
                port.holdings[j] -= amt;
                port.cash        += sell_net(amt, nd_open);
                sym_cash[j]        += sell_net(amt, nd_open);
                local_sell       += amt;
            }

            // Gap-up sell_all at open
            if (port.holdings[j] > 1e-6f && nd_open >= sell_all_price) {
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, nd_open);
                sym_cash[j]        += sell_net(amt, nd_open);
                local_sell       += amt;
            }

            // High-first: intraday sell_all fires before buy
            if (!low_first && port.holdings[j] > 1e-6f &&
                nd_low < sell_all_price && sell_all_price < nd_high) {
                float slipped = sell_all_price * (1.f - SLIPPAGE_RATE);
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, slipped);
                sym_cash[j]        += sell_net(amt, slipped);
                local_sell       += amt;
            }

            // Stop loss
            float stop_p = port.stop_prices[j];
            if (stop_p > 0.f && port.holdings[j] > 1e-6f) {
                if (nd_open <= stop_p) {
                    float amt = port.holdings[j];
                    port.holdings[j] = 0.f;
                    port.cash        += sell_net(amt, nd_open);
                    sym_cash[j]        += sell_net(amt, nd_open);
                    local_sell       += amt;
                } else if (nd_low <= stop_p) {
                    float slipped = stop_p * (1.f - SLIPPAGE_RATE);
                    float amt = port.holdings[j];
                    port.holdings[j] = 0.f;
                    port.cash        += sell_net(amt, slipped);
                    sym_cash[j]        += sell_net(amt, slipped);
                    local_sell       += amt;
                }
            }

            // Limit buy
            if (buy_qty > 1e-6f && buy_price > 0.f) {
                float fill_price = 0.f;
                if (nd_open <= buy_price)
                    fill_price = nd_open;
                else if (nd_low < buy_price && buy_price < nd_high)
                    fill_price = buy_price * (1.f + SLIPPAGE_RATE);

                if (fill_price > 0.f) {
                    float affordable = port.cash / fill_price;
                    float buy_amount = std::min(buy_qty, affordable);
                    if (buy_amount > 1e-6f) {
                        // 60% concentration cap
                        float port_value = port.cash;
                        for (int k = 0; k < IND_SYMS; k++) {
                            float cp = fill_sym[k].valid ? fill_sym[k].close :
                                       day_sym[k].valid  ? day_sym[k].close  : 0.f;
                            port_value += port.holdings[k] * cp;
                        }
                        float cur_sym_val = port.holdings[j] * fill_price;
                        float max_spend   = std::max(0.f, MAX_SINGLE_STOCK_PCT * port_value - cur_sym_val);
                        buy_amount = std::min(buy_amount, max_spend / fill_price);
                    }
                    buy_amount = whole_shares(buy_amount);
                    if (buy_amount > 1e-6f) {
                        port.holdings[j]    += buy_amount;
                        port.cash           -= buy_amount * fill_price;
                        sym_cash[j]           -= buy_amount * fill_price;
                        port.stop_prices[j]  = stop_loss;
                        local_buy           += buy_amount;
                    }
                }
            }
        }

        // ── Phase 2: low-first intraday sell_all ──────────────────────────────
        for (int j = 0; j < IND_SYMS; j++) {
            if (!day_sym[j].valid) continue;
            if (!seq_flags[ind_i * IND_SYMS + j]) continue;  // high-first: skip

            float sell_all_price_frac = out48[4*j+2];
            float span_t = std::max(day_sym[j].high - day_sym[j].low, 1e-9f);
            float sell_all_price = day_sym[j].low + sell_all_price_frac * span_t;

            float nd_low  = fill_sym[j].valid ? fill_sym[j].low  : day_sym[j].low;
            float nd_high = fill_sym[j].valid ? fill_sym[j].high : day_sym[j].high;

            if (port.holdings[j] > 1e-6f && nd_low < sell_all_price && sell_all_price < nd_high) {
                float slipped = sell_all_price * (1.f - SLIPPAGE_RATE);
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, slipped);
                sym_cash[j]        += sell_net(amt, slipped);
                local_sell       += amt;
            }
        }

        slot_scores[slot] = compute_value_ind(port, day_sym, fill_sym);
        // book P&L for symbol j = what the position is worth now, minus what it was worth at
        // yesterday's close, plus the cash this symbol moved. Summed over j the cash terms telescope
        // into the slot's total cash change and the identity closes exactly -- see the assert below.
        for (int j = 0; j < IND_SYMS; j++) {
            const float pf = fill_sym[j].valid ? fill_sym[j].close :
                             day_sym[j].valid  ? day_sym[j].close  : 0.f;
            const float pt = day_sym[j].valid  ? day_sym[j].close :
                             fill_sym[j].valid ? fill_sym[j].close : 0.f;
            res_sym_pnl[slot][j] = port.holdings[j] * pf - ref_hold[j] * pt + sym_cash[j];
        }
        // The identity is the only thing standing between MT1S-stock and training on a residual
        // that belongs to no symbol. Checked on slot 0 -- the deployed model, so the one whose
        // numbers get read -- and only reported when it actually breaks, because a per-day log
        // line that always says "fine" is a line nobody reads. The tolerance is relative: these
        // are float sums over twelve terms against a book of ~$25-50k, so ~1e-4 relative is the
        // floor that float can hold.
        if (slot == 0) {
            double sum = 0.0;
            for (int j = 0; j < IND_SYMS; j++) sum += res_sym_pnl[0][j];
            const double want = (double)slot_scores[0] - (double)book_prev;
            const double scale = std::max(1.0, std::fabs((double)book_prev));
            if (std::fabs(sum - want) / scale > 1e-4) {
                char m[192];
                snprintf(m, sizeof(m),
                         "   WARNING: per-symbol P&L does not close: sum %.2f vs book %.2f "
                         "(diff %.2f) -- MT1S-stock targets are unreliable",
                         sum, want, sum - want);
                log_msg(std::string("[") + IND_SHORT[ind_i] + "]" + m);
            }
        }
        trade_count[slot] = (int)((local_buy + local_sell) > 1e-6f);
        buy_exec  += local_buy;
        sell_exec += local_sell;
    }

    // ── Score history models ─────────────────────────────────────────────────
    Portfolio hist_ports[HIST_DAYS * HIST_PER_DAY] = {};
    float hist_scores[HIST_DAYS * HIST_PER_DAY] = {};
    int n_hist = scratch.hist_count * HIST_PER_DAY;
    // Score one arbitrary weight vector against TODAY's reference portfolio and return the
    // resulting portfolio value. Lifted out of the history loop unchanged so the control sampler
    // can reuse it verbatim: this body IS the fill simulation, and a third hand-copy of it is
    // exactly how two sides of a calculation drift apart in this codebase.
    auto score_weights = [&](const float* W, Portfolio& port) -> float {
        port.cash = ref_cash;
        for (int j = 0; j < IND_SYMS; j++) {
            port.holdings[j]    = ref_hold[j];
            port.stop_prices[j] = ref_stop[j];
        }
        stock_forward(W, history_arr, today_arr, out48);
        float local_buy = 0.f, local_sell = 0.f;
        // ── Phase 1: partial sells, gap sell_all, high-first sell_all, stops, buys ─
        for (int j = 0; j < IND_SYMS; j++) {
            if (!day_sym[j].valid) continue;
            float buy_qty             = out48[4*j+0];
            float buy_price_frac      = out48[4*j+1];
            float sell_all_price_frac = out48[4*j+2];
            float sell_qty            = out48[4*j+3];
            float low_t  = day_sym[j].low;
            float high_t = day_sym[j].high;
            float span_t = std::max(high_t - low_t, 1e-9f);
            float sell_all_price = low_t + sell_all_price_frac * span_t;
            float buy_price      = low_t + buy_price_frac * span_t;
            float stop_loss      = buy_price * 0.9f;
            float nd_open = fill_sym[j].valid ? fill_sym[j].open  : day_sym[j].close;
            float nd_low  = fill_sym[j].valid ? fill_sym[j].low   : day_sym[j].low;
            float nd_high = fill_sym[j].valid ? fill_sym[j].high  : day_sym[j].high;
            bool low_first = seq_flags[ind_i * IND_SYMS + j];
            if (sell_qty > 1e-6f && port.holdings[j] > 1e-6f) {
                float amt = whole_shares(std::min(sell_qty, port.holdings[j]));
                port.holdings[j] -= amt;
                port.cash        += sell_net(amt, nd_open);
                local_sell       += amt;
            }
            if (port.holdings[j] > 1e-6f && nd_open >= sell_all_price) {
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, nd_open);
                local_sell       += amt;
            }
            if (!low_first && port.holdings[j] > 1e-6f &&
                nd_low < sell_all_price && sell_all_price < nd_high) {
                float slipped = sell_all_price * (1.f - SLIPPAGE_RATE);
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, slipped);
                local_sell       += amt;
            }
            float stop_p = port.stop_prices[j];
            if (stop_p > 0.f && port.holdings[j] > 1e-6f) {
                if (nd_open <= stop_p) {
                    float amt = port.holdings[j];
                    port.holdings[j] = 0.f;
                    port.cash        += sell_net(amt, nd_open);
                    local_sell       += amt;
                } else if (nd_low <= stop_p) {
                    float slipped = stop_p * (1.f - SLIPPAGE_RATE);
                    float amt = port.holdings[j];
                    port.holdings[j] = 0.f;
                    port.cash        += sell_net(amt, slipped);
                    local_sell       += amt;
                }
            }
            if (buy_qty > 1e-6f && buy_price > 0.f) {
                float fill_price = 0.f;
                if (nd_open <= buy_price)
                    fill_price = nd_open;
                else if (nd_low < buy_price && buy_price < nd_high)
                    fill_price = buy_price * (1.f + SLIPPAGE_RATE);
                if (fill_price > 0.f) {
                    float affordable = port.cash / fill_price;
                    float buy_amount = std::min(buy_qty, affordable);
                    if (buy_amount > 1e-6f) {
                        float port_value = port.cash;
                        for (int k = 0; k < IND_SYMS; k++) {
                            float cp = fill_sym[k].valid ? fill_sym[k].close :
                                       day_sym[k].valid  ? day_sym[k].close  : 0.f;
                            port_value += port.holdings[k] * cp;
                        }
                        float cur_sym_val = port.holdings[j] * fill_price;
                        float max_spend   = std::max(0.f, MAX_SINGLE_STOCK_PCT * port_value - cur_sym_val);
                        buy_amount = std::min(buy_amount, max_spend / fill_price);
                    }
                    buy_amount = whole_shares(buy_amount);
                    if (buy_amount > 1e-6f) {
                        port.holdings[j]    += buy_amount;
                        port.cash           -= buy_amount * fill_price;
                        port.stop_prices[j]  = stop_loss;
                        local_buy           += buy_amount;
                    }
                }
            }
        }
        // ── Phase 2: low-first intraday sell_all ──────────────────────────────
        for (int j = 0; j < IND_SYMS; j++) {
            if (!day_sym[j].valid) continue;
            if (!seq_flags[ind_i * IND_SYMS + j]) continue;
            float sell_all_price_frac = out48[4*j+2];
            float span_t = std::max(day_sym[j].high - day_sym[j].low, 1e-9f);
            float sell_all_price = day_sym[j].low + sell_all_price_frac * span_t;
            float nd_low  = fill_sym[j].valid ? fill_sym[j].low  : day_sym[j].low;
            float nd_high = fill_sym[j].valid ? fill_sym[j].high : day_sym[j].high;
            if (port.holdings[j] > 1e-6f && nd_low < sell_all_price && sell_all_price < nd_high) {
                float slipped = sell_all_price * (1.f - SLIPPAGE_RATE);
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, slipped);
                local_sell       += amt;
            }
        }
        (void)local_buy; (void)local_sell;
        return compute_value_ind(port, day_sym, fill_sym);
    };

    for (int h = 0; h < n_hist; h++)
        hist_scores[h] = score_weights(scratch.hist(h / HIST_PER_DAY, h % HIST_PER_DAY),
                                       hist_ports[h]);

    // ── Score, flags, floor check ────────────────────────────────────────────
    float best_score  = *std::max_element(slot_scores, slot_scores + N_SLOTS);
    float best_delta  = best_score - baseline;
    float worst_delta = *std::min_element(slot_scores, slot_scores + N_SLOTS) - baseline;

    // Mutation success rate: each mutation slot against the parent it was mutated FROM, matching
    // the parent assignment in the forward pass (uniform, MUTATIONS_PER_PARENT children each).
    int mut_wins = 0, mut_total = 0;
    for (int slot = ELITE_POOL; slot < N_SLOTS; slot++) {
        int parent = (slot - ELITE_POOL) / MUTATIONS_PER_PARENT;
        if (parent >= ELITE_POOL) continue;
        mut_total++;
        if (slot_scores[slot] > slot_scores[parent]) mut_wins++;
    }
    float mut_success = mut_total > 0 ? (float)mut_wins / (float)mut_total : 0.f;

    // Elite stats (slots 0..ELITE_COUNT-1 portfolio values)
    float elite_max_val = *std::max_element(slot_scores, slot_scores + ELITE_COUNT);
    float elite_min_val = *std::min_element(slot_scores, slot_scores + ELITE_COUNT);
    float elite_mean_val = 0.f;
    for (int s = 0; s < ELITE_COUNT; s++) elite_mean_val += slot_scores[s];
    elite_mean_val /= ELITE_COUNT;

    log_msg(std::string("[") + IND_SHORT[ind_i] + "] Day " +
            std::to_string(actual_day + 1) + "/" + std::to_string(total_avail) +
            " | best Δ" + (best_delta >= 0 ? "+" : "") + std::to_string(safe_int(best_delta)) +
            " worst Δ" + (worst_delta >= 0 ? "+" : "") + std::to_string(safe_int(worst_delta)) +
            " | buys=" + std::to_string(safe_int(buy_exec)) +
            " sells=" + std::to_string(safe_int(sell_exec)) +
            " | prod=$" + std::to_string(safe_int(baseline)) +
            " | mut_ok=" + std::to_string((int)(mut_success * 100.f + 0.5f)) + "%");

    // Hard floor reset
    float abs_floor = IND_STARTING_CASH * 0.9f;
    if (baseline < abs_floor) {
        log_msg(std::string("[") + IND_SHORT[ind_i] +
                "] Production below floor — resetting portfolios");
        for (int s = 0; s < N_SLOTS; s++) {
            state.portfolios[s].cash = IND_STARTING_CASH;
            for (int j = 0; j < IND_SYMS; j++) {
                state.portfolios[s].holdings[j]    = 0.f;
                state.portfolios[s].stop_prices[j] = 0.f;
            }
        }
        // Designated, not positional: this aggregate silently reassigns every field if a member
        // is added ahead of it, and only got caught here because the shift happened to narrow a
        // float into an int. book_prev == slot0_score on a reset day, so the day contributes a
        // book P&L of exactly 0 rather than a spurious jump to the reset level.
        IndResult r{};          // value-initialised: order intent is all zero on a reset day
        r.book_prev      = baseline;
        r.baseline       = baseline;
        r.slot0_score    = baseline;
        r.new_streak     = 0;
        r.elite_max_val  = IND_STARTING_CASH;
        r.elite_min_val  = IND_STARTING_CASH;
        r.elite_mean_val = IND_STARTING_CASH;
        return r;
    }

    // Zero-trade inaction filter
    int new_streak = 0;
    bool all_inactive = false;
    bool inactive[N_SLOTS] = {};
    if (num_past >= HIST_WINDOW && day_num > 0) {
        int n_inactive = 0;
        for (int s = 0; s < N_SLOTS; s++) {
            if (!trade_count[s]) { inactive[s] = true; n_inactive++; }
        }
        if (n_inactive == N_SLOTS) {
            all_inactive = true;
            new_streak   = state.streak + 1;
        } else if (n_inactive > 0) {
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   Zero-trade filter: " +
                    std::to_string(n_inactive) + " slot(s) excluded");
        }
    }

    // Selection scores with invested_pct multiplier
    float sel_scores[N_SLOTS];
    for (int s = 0; s < N_SLOTS; s++) {
        float raw_delta = slot_scores[s] - baseline;
        if (raw_delta > 0.f && slot_scores[s] > 0.f) {
            float invested_pct = std::max(0.f, 1.f - state.portfolios[s].cash / slot_scores[s]);
            sel_scores[s] = raw_delta * invested_pct;
        } else {
            sel_scores[s] = raw_delta;
        }
    }

    // Preserve slot 0's own portfolio before selection overwrites it
    Portfolio slot0_own = state.portfolios[0];

    // ── Selection + mutation ─────────────────────────────────────────────────
    if (!freeze) {
        float survival_floor = -(baseline * 0.1f);
        float below_floor_thresh = baseline * 0.9f;

        // Build sorted surviving list (N_SLOTS current slots + history candidates)
        std::vector<std::pair<float,int>> surviving;
        for (int s = 0; s < N_SLOTS; s++) {
            if (inactive[s]) continue;
            if (slot_scores[s] < below_floor_thresh) continue;
            if (sel_scores[s] < survival_floor) continue;
            surviving.push_back({sel_scores[s], s});
        }
        // History candidates (no inactive filter; slot index = N_SLOTS+h)
        for (int h = 0; h < n_hist; h++) {
            float raw_delta = hist_scores[h] - baseline;
            float sel_h = (raw_delta > 0.f && hist_scores[h] > 0.f)
                ? raw_delta * std::max(0.f, 1.f - hist_ports[h].cash / hist_scores[h])
                : raw_delta;
            if (hist_scores[h] < below_floor_thresh) continue;
            if (sel_h < survival_floor) continue;
            surviving.push_back({sel_h, N_SLOTS + h});
        }
        if (surviving.empty()) {
            // relax: drop inactive filter (history already included above)
            for (int s = 0; s < N_SLOTS; s++) {
                if (slot_scores[s] < below_floor_thresh) continue;
                if (sel_scores[s] < survival_floor) continue;
                surviving.push_back({sel_scores[s], s});
            }
        }

        if (!surviving.empty()) {
            std::sort(surviving.begin(), surviving.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });
            int n_top = std::min((int)surviving.size(), ELITE_COUNT);

            // Source ranks (indices into current state.elites[]) for top-17
            int   src_rank[ELITE_COUNT] = {};
            float src_val[ELITE_COUNT]  = {};
            for (int k = 0; k < n_top; k++) {
                src_rank[k] = surviving[k].second;
                src_val[k]  = surviving[k].first;
            }

            float w5_weights[5], w10_weights[10], w15_weights[15];
            int n5  = std::min(n_top, 5);
            int n10 = std::min(n_top, 10);
            int n15 = std::min(n_top, 15);
            normalize_weights(src_val, w5_weights, n5);
            normalize_weights(src_val, w10_weights, n10);
            normalize_weights(src_val, w15_weights, n15);

            // Portfolio wavg: normal slots → state.portfolios, history slots → hist_ports
            auto get_port = [&](int sl) -> const Portfolio* {
                return (sl < N_SLOTS) ? &state.portfolios[sl] : &hist_ports[sl - N_SLOTS];
            };
            Portfolio wp5{}, wp10{}, wp15{};
            const Portfolio* p5[5], *p10[10], *p15[15];
            for (int k = 0; k < n5;  k++) p5[k]  = get_port(src_rank[k]);
            for (int k = 0; k < n10; k++) p10[k] = get_port(src_rank[k]);
            for (int k = 0; k < n15; k++) p15[k] = get_port(src_rank[k]);
            wavg_portfolio(p5,  w5_weights,  n5,  wp5);
            wavg_portfolio(p10, w10_weights, n10, wp10);
            wavg_portfolio(p15, w15_weights, n15, wp15);

            // Copy/regenerate top-n_top into new_elites[0..n_top-1].
            // slot < ELITE_POOL: copy elite directly.
            // ELITE_POOL <= slot < N_SLOTS: mutation — regenerate from parent + seed.
            // slot >= N_SLOTS: history candidate — copy from hist_buf.
            Portfolio new_ports[ELITE_POOL];
            for (int k = 0; k < n_top; k++) {
                int slot = src_rank[k];
                if (slot >= N_SLOTS) {
                    int h = slot - N_SLOTS;
                    memcpy(scratch.new_elite(k), scratch.hist(h / HIST_PER_DAY, h % HIST_PER_DAY),
                           STOCKNN_PARAMS * sizeof(float));
                    new_ports[k] = hist_ports[h];
                } else if (slot < ELITE_POOL) {
                    memcpy(scratch.new_elite(k), scratch.elite(slot), STOCKNN_PARAMS * sizeof(float));
                    new_ports[k] = state.portfolios[slot];
                } else {
                    int mut_i  = slot - ELITE_POOL;
                    int parent = mut_i / MUTATIONS_PER_PARENT;
                    memcpy(scratch.new_elite(k), scratch.elite(parent), STOCKNN_PARAMS * sizeof(float));
                    apply_gaussian(scratch.new_elite(k), STOCKNN_PARAMS, sigma, mut_seeds[mut_i]);
                    new_ports[k] = state.portfolios[slot];
                }
            }
            for (int k = n_top; k < ELITE_COUNT; k++) {
                memcpy(scratch.new_elite(k), scratch.new_elite(0), STOCKNN_PARAMS * sizeof(float));
                new_ports[k] = new_ports[0];
            }

            // Weight wavg uses new_elites[0..n-1] (consecutive, no OOB risk)
            int seq[ELITE_COUNT]; for (int k = 0; k < ELITE_COUNT; k++) seq[k] = k;
            wavg_weights_flat(scratch.new_elites, STOCKNN_PARAMS, seq, w5_weights,  n5,  scratch.wavg(0));
            wavg_weights_flat(scratch.new_elites, STOCKNN_PARAMS, seq, w10_weights, n10, scratch.wavg(1));
            wavg_weights_flat(scratch.new_elites, STOCKNN_PARAMS, seq, w15_weights, n15, scratch.wavg(2));

            memcpy(scratch.new_elite(ELITE_COUNT),     scratch.wavg(0), STOCKNN_PARAMS * sizeof(float));
            memcpy(scratch.new_elite(ELITE_COUNT + 1), scratch.wavg(1), STOCKNN_PARAMS * sizeof(float));
            memcpy(scratch.new_elite(ELITE_COUNT + 2), scratch.wavg(2), STOCKNN_PARAMS * sizeof(float));
            new_ports[ELITE_COUNT]     = wp5;
            new_ports[ELITE_COUNT + 1] = wp10;
            new_ports[ELITE_COUNT + 2] = wp15;

            for (int k = 0; k < ELITE_POOL; k++) {
                memcpy(scratch.elite(k), scratch.new_elite(k), STOCKNN_PARAMS * sizeof(float));
                state.portfolios[k] = new_ports[k];
            }

            // Assign mutation children from new parents (ELITE_POOL parents × 9 each = 180)
            for (int mut_i = 0; mut_i < N_SLOTS - ELITE_POOL; mut_i++) {
                int child_slot  = ELITE_POOL + mut_i;
                int parent_rank = mut_i / MUTATIONS_PER_PARENT;
                state.portfolios[child_slot] = state.portfolios[parent_rank];
            }

            std::string elite_display = "";
            for (int k = 0; k < std::min(n_top, 5); k++) {
                if (k > 0) elite_display += ",";
                int sl = src_rank[k];
                if (sl >= N_SLOTS) {
                    int h = sl - N_SLOTS;
                    elite_display += "H" + std::to_string(h / HIST_PER_DAY) +
                                     "." + std::to_string(h % HIST_PER_DAY);
                } else if (sl < ELITE_COUNT) {
                    elite_display += std::to_string(sl) + ".0";
                } else if (sl < ELITE_POOL) {
                    const char* nm[] = {"w5","w10","w15"};
                    elite_display += nm[sl - ELITE_COUNT];
                } else {
                    int p = (sl - ELITE_POOL) / MUTATIONS_PER_PARENT;
                    int m = (sl - ELITE_POOL) % MUTATIONS_PER_PARENT + 1;
                    elite_display += std::to_string(p) + "." + std::to_string(m);
                }
            }
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   Selection done | elite=[" +
                    elite_display + "] | top=$" + std::to_string((int)src_val[0]));
        }

        // Diversity injection for all-zero streak >= 2
        if (all_inactive && new_streak >= 2) {
            int half = ELITE_COUNT / 2;
            PCG32 div_rng; div_rng.seed(mix_seed((uint64_t)actual_day * 99991ULL + ind_i));
            for (int k = half; k < ELITE_COUNT; k++) {
                // blend top half with random: 0.5 * elite + 0.5 * random (reuse mut_buf)
                init_stock_weights(scratch.mut_buf, div_rng);
                for (int p = 0; p < STOCKNN_PARAMS; p++)
                    scratch.elite(k)[p] = 0.5f * scratch.elite(k - half)[p] + 0.5f * scratch.mut_buf[p];
                state.portfolios[k] = state.portfolios[k - half];
            }
            new_streak = 0;
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   Diversity injection complete — streak reset");
        }
    }

    // Restore slot 0's own portfolio
    state.portfolios[0] = slot0_own;

    // ── Push today's top-7 elites + 3 wavg slots to history circular buffer ──
    {
        int hd = scratch.hist_head;
        for (int k = 0; k < HIST_ELITE; k++)
            memcpy(scratch.hist(hd, k), scratch.elite(k), STOCKNN_PARAMS * sizeof(float));
        for (int k = 0; k < HIST_WAVG; k++)
            memcpy(scratch.hist(hd, HIST_ELITE + k), scratch.elite(ELITE_COUNT + k),
                   STOCKNN_PARAMS * sizeof(float));
        scratch.hist_head = (hd + 1) % HIST_DAYS;
        if (scratch.hist_count < HIST_DAYS) scratch.hist_count++;
    }

    // ── Untrained control: N random models, scored on today, then discarded ──────────────
    // Runs AFTER selection and mutation so it cannot perturb them. It reads only the saved
    // reference portfolio (ref_cash/ref_hold/ref_stop), which selection does not touch.
    float ctl_best = 0.f, ctl_mean = 0.f;
    int   ctl_n = 0;
    if (g_control_random && g_cur_pass == 0) {
        // Match the day's candidate count exactly. The trained side reports the best of
        // N_SLOTS + n_hist, and max-of-N grows with N, so a smaller control pool would lose on
        // pool size alone and that would read as a training effect.
        ctl_n = N_SLOTS + n_hist;
        static thread_local std::vector<float> ctl_buf;
        if (ctl_buf.size() != (size_t)STOCKNN_PARAMS) ctl_buf.resize(STOCKNN_PARAMS);
        PCG32 crng;
        crng.seed(mix_seed((uint64_t)actual_day * 1000003ULL +
                           (uint64_t)ind_i * 65537ULL + 7ULL));
        Portfolio cport;
        double sum = 0.0;
        ctl_best = -1e30f;
        for (int c = 0; c < ctl_n; c++) {
            init_stock_weights(ctl_buf.data(), crng);
            float d = score_weights(ctl_buf.data(), cport) - baseline;
            sum += (double)d;
            if (d > ctl_best) ctl_best = d;
        }
        ctl_mean = (float)(sum / (double)ctl_n);
        log_msg(std::string("[") + IND_SHORT[ind_i] + "]   control: " +
                std::to_string(ctl_n) + " random | best Δ" +
                (ctl_best >= 0 ? "+" : "") + std::to_string((int)ctl_best) +
                " mean Δ" + (ctl_mean >= 0 ? "+" : "") + std::to_string((int)ctl_mean) +
                " | trained best Δ" + (best_delta >= 0 ? "+" : "") +
                std::to_string(safe_int(best_delta)) +
                (ctl_best >= best_delta ? "  <-- CONTROL WINS" : ""));
    }

    // Report: top_hold = slot0 holdings value, top_cash = slot0 cash
    float top_hold = 0.f;
    for (int j = 0; j < IND_SYMS; j++) {
        float price = fill_sym[j].valid ? fill_sym[j].close :
                      day_sym[j].valid  ? day_sym[j].close  : 0.f;
        top_hold += slot0_own.holdings[j] * price;
    }

    // Save updated elites and history back to disk. These are reloaded at the top of the next
    // day, so skipping the write is what broke --no-save; under --no-save models_dir is already
    // redirected to a scratch directory instead. The one case that genuinely skips the write is
    // --control-untrained, where tomorrow re-randomises rather than loads, so saving would only
    // burn 3 GB of I/O and leave a directory of weights that look trained.
    if (!g_control_untrained) {
        save_industry_elites(models_dir, ind_i, scratch.elite_buf);
        save_ind_history(models_dir, ind_i, scratch);
    }

    IndResult res;
    res.ref_cash_v = ref_cash;
    for (int j = 0; j < IND_SYMS; j++) {
        res.ref_hold[j] = ref_hold[j];
        res.si_bqty[j]  = si_bq[j];
        res.si_bfrac[j] = si_bf[j];
        res.si_sfrac[j] = si_sf[j];
        res.si_sqty[j]  = si_sq[j];
    }
    res.oi_n_buy    = (float)oi_nb;
    res.oi_n_sell   = (float)oi_ns;
    res.oi_buy_aggr = oi_nb ? (float)(oi_ba / oi_nb) : 0.f;
    res.oi_sell_aggr= oi_ns ? (float)(oi_sa / oi_ns) : 0.f;
    // population sd of buy_price_frac; 0 with fewer than two orders, which is the honest value
    res.oi_buy_disp = (oi_nb > 1)
        ? (float)sqrt(std::max(0.0, oi_bf2 / oi_nb - (oi_bf / oi_nb) * (oi_bf / oi_nb))) : 0.f;
    res.oi_buy_val  = (float)oi_bv;
    res.oi_sell_val = (float)oi_sv;
    for (int s = 0; s < ELITE_POOL; s++) res.elite_score[s] = slot_scores[s];
    for (int s = 0; s < N_SLOTS; s++) {
        res.slot_score[s] = slot_scores[s];
        memcpy(res.slot_bqty [s], res_slot_bq[s], IND_SYMS * sizeof(float));
        memcpy(res.slot_bfrac[s], res_slot_bf[s], IND_SYMS * sizeof(float));
        memcpy(res.slot_sfrac[s], res_slot_sf[s], IND_SYMS * sizeof(float));
        memcpy(res.slot_sqty [s], res_slot_sq[s], IND_SYMS * sizeof(float));
        memcpy(res.slot_sym_pnl[s], res_sym_pnl[s], IND_SYMS * sizeof(float));
    }
    res.book_prev     = book_prev;
    res.baseline      = baseline;
    res.slot0_score   = slot_scores[0];
    res.best_delta    = best_delta;
    res.top_hold      = top_hold;
    res.top_cash      = slot0_own.cash;
    res.new_streak    = new_streak;
    res.mut_success   = mut_success;
    res.elite_max_val = elite_max_val;
    res.elite_min_val = elite_min_val;
    res.elite_mean_val= elite_mean_val;
    res.ctl_best      = ctl_best;
    res.ctl_mean      = ctl_mean;
    for (int j = 0; j < IND_SYMS; j++) res.hold[j] = slot0_own.holdings[j];
    res.hold_cash     = slot0_own.cash;
    res.ctl_n         = ctl_n;
    state.streak      = new_streak;
    return res;
}

// ── Master feature construction helpers ─────────────────────────────────────────

// Solve 3×3 augmented system A[3][4] in-place; result in x[3].
static void gauss_solve3(float A[3][4], float x[3]) {
    for (int col = 0; col < 3; col++) {
        int piv = col;
        for (int r = col+1; r < 3; r++)
            if (fabsf(A[r][col]) > fabsf(A[piv][col])) piv = r;
        if (piv != col) for (int j = 0; j < 4; j++) { float t=A[col][j]; A[col][j]=A[piv][j]; A[piv][j]=t; }
        if (fabsf(A[col][col]) < 1e-12f) continue;
        for (int r = col+1; r < 3; r++) {
            float f = A[r][col] / A[col][col];
            for (int j = col; j < 4; j++) A[r][j] -= f * A[col][j];
        }
    }
    for (int i = 2; i >= 0; i--) {
        x[i] = A[i][3];
        for (int j = i+1; j < 3; j++) x[i] -= A[i][j] * x[j];
        x[i] = fabsf(A[i][i]) > 1e-12f ? x[i] / A[i][i] : 0.f;
    }
}

// Solve 4×4 augmented system A[4][5] in-place; result in x[4].
static void gauss_solve4(float A[4][5], float x[4]) {
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col+1; r < 4; r++)
            if (fabsf(A[r][col]) > fabsf(A[piv][col])) piv = r;
        if (piv != col) for (int j = 0; j < 5; j++) { float t=A[col][j]; A[col][j]=A[piv][j]; A[piv][j]=t; }
        if (fabsf(A[col][col]) < 1e-12f) continue;
        for (int r = col+1; r < 4; r++) {
            float f = A[r][col] / A[col][col];
            for (int j = col; j < 5; j++) A[r][j] -= f * A[col][j];
        }
    }
    for (int i = 3; i >= 0; i--) {
        x[i] = A[i][4];
        for (int j = i+1; j < 4; j++) x[i] -= A[i][j] * x[j];
        x[i] = fabsf(A[i][i]) > 1e-12f ? x[i] / A[i][i] : 0.f;
    }
}

// Degree-2 polyfit via normal equations; coefs[3] = {a2, a1, a0} (highest power first).
// x = linspace(0,1,n); n must be ≥ 1.
static void polyfit2(const float* y, int n, float coefs[3]) {
    float A[3][4] = {};
    for (int i = 0; i < n; i++) {
        float xi = (n > 1) ? (float)i / (float)(n-1) : 0.f;
        float v[3] = {xi*xi, xi, 1.f};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) A[r][c] += v[r] * v[c];
            A[r][3] += v[r] * y[i];
        }
    }
    gauss_solve3(A, coefs);
}

// Degree-3 polyfit; coefs[4] = {a3, a2, a1, a0}.
static void polyfit3(const float* y, int n, float coefs[4]) {
    float A[4][5] = {};
    for (int i = 0; i < n; i++) {
        float xi = (n > 1) ? (float)i / (float)(n-1) : 0.f;
        float v[4] = {xi*xi*xi, xi*xi, xi, 1.f};
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) A[r][c] += v[r] * v[c];
            A[r][4] += v[r] * y[i];
        }
    }
    gauss_solve4(A, coefs);
}

// hist_at: clamp-to-oldest accessor into a rolling oldest-first buffer.
// hist[0..hist_count-1]; k=0 → newest, k=hist_count-1 → oldest.
static inline float hist_at(const float* hist, int hist_count, int k) {
    int idx = hist_count - 1 - k;
    return idx < 0 ? hist[0] : hist[idx];
}

// Fill window[0..win-1] with oldest-first values ending at newest, left-padding with oldest.
static void fill_window(const float* hist, int hist_count, int win, float* window) {
    for (int j = 0; j < win; j++) {
        int src = (hist_count - win) + j;
        window[j] = (src <= 0) ? hist[0] : hist[src];
    }
}

// Build the 37-feature block for ONE industry from a cumulative-value history `h`.
// feat37 layout (must stay bit-identical to training_lib.build_master_features):
//   [0..9]   10 daily returns (Δt=1, days 1..10 back)
//   [10..16] 7 cumulative 10-day bucket returns (Δt=10, coherent decade-resolution series)
//   [17]     realized return-volatility (mean |daily return| over trailing 20 days) — the
//            scale/uncertainty anchor for the magnitude + range/confidence outputs
//   [18..20] poly-2 over 5d, values normalized by current level (dimensionless shape)
//   [21..36] poly-3 over {10,30,60,90}, values normalized by current level
static void build_ind_features37(const float* h, int hist_count, float* feat) {
    static constexpr int POLY3_WINS[4] = {10,30,60,90};
    // Return features are ~0.01 in magnitude; lift them to ~unit scale so the net produces
    // output variance at init. Information-preserving constant, NOT a normalization.
    static constexpr float RETURN_SCALE = 100.f;
    float cur = hist_at(h, hist_count, 0);
    float cur_denom = fabsf(cur) > 1e-9f ? cur : 1e-9f;

    for (int lt = 0; lt < 10; lt++) {                 // [0..9] daily returns
        int t = lt + 1;
        float vt  = hist_at(h, hist_count, t);
        float vt1 = hist_at(h, hist_count, t + 1);
        float denom = fabsf(vt1) > 1e-9f ? vt1 : (vt1 >= 0.f ? 1e-9f : -1e-9f);
        feat[lt] = (vt - vt1) / fabsf(denom);
    }
    for (int k = 1; k <= 7; k++) {                    // [10..16] 10-day bucket returns
        float va = hist_at(h, hist_count, 10 * (k - 1));
        float vb = hist_at(h, hist_count, 10 * k);
        float denom = fabsf(vb) > 1e-9f ? vb : (vb >= 0.f ? 1e-9f : -1e-9f);
        feat[9 + k] = (va - vb) / fabsf(denom);
    }
    {                                                 // [17] realized return-volatility
        float s = 0.f;
        for (int d = 1; d <= 20; d++) {
            float vt  = hist_at(h, hist_count, d);
            float vt1 = hist_at(h, hist_count, d + 1);
            float denom = fabsf(vt1) > 1e-9f ? vt1 : (vt1 >= 0.f ? 1e-9f : -1e-9f);
            s += fabsf((vt - vt1) / fabsf(denom));
        }
        feat[17] = s / 20.f;
    }
    for (int j = 0; j < 18; j++) feat[j] *= RETURN_SCALE;   // lift returns/vol to ~unit scale
    float win5[5];                                    // [18..20] poly-2, level-normalized
    fill_window(h, hist_count, 5, win5);
    for (int j = 0; j < 5; j++) win5[j] /= cur_denom;
    polyfit2(win5, 5, feat + 18);
    float polywin[90];                                // [21..36] poly-3, level-normalized
    for (int wi = 0; wi < 4; wi++) {
        int W = POLY3_WINS[wi];
        fill_window(h, hist_count, W, polywin);
        for (int j = 0; j < W; j++) polywin[j] /= cur_denom;
        polyfit3(polywin, W, feat + 21 + wi * 4);
    }
}

// Build the 888-feature dual vector for MT1 (Part C): per industry, 74 features =
// [37 market-index features ‖ 37 portfolio (slot-0 StockNN) features]. The two blocks share the
// identical 37-feature builder; only the source cumulative-value history differs.
static void build_master_features(const float mkt_val_hist[][IND_HIST_CAP],
                                   const float pf_val_hist[][IND_HIST_CAP],
                                   int hist_count, float* out888) {
    memset(out888, 0, 888 * sizeof(float));
    for (int i = 0; i < N_IND; i++) {
        float* feat = out888 + i * 74;
        build_ind_features37(mkt_val_hist[i], hist_count, feat);        // [0..36]  market
        build_ind_features37(pf_val_hist[i],  hist_count, feat + 37);   // [37..73] portfolio
    }
}

// ── step_master (legacy MasterNN allocator) removed in Part C ─────────────────────
// The 444-wide MasterNN path is fully superseded by MT2 (48 raw MT1 activations). Its
// build_master_features + master_forward wiring was incompatible with the 888-wide dual
// (market ‖ portfolio) feature vector, so the dead function was dropped. master_forward /
// MasterNN class remain (unused, self-contained) for backward-compat of any .bin readers.

// ── MT1 per-industry training step ──────────────────────────────────────────────


// Gaussian mutation of one MT1Net.
static void mt1_mutate(const float* parent, float* dst, int n_params, float sigma, uint64_t seed) {
    memcpy(dst, parent, sizeof(float) * n_params);
    apply_gaussian(dst, n_params, sigma, mix_seed(seed));
}

// ── MT1: one pool per industry, one step per day ─────────────────────────────────
//
// Replaces step_mt1_pool / step_mt1_dir_pool / step_mt1_tail / step_mt1_head and the T1/H/T2
// block alternation. One call per industry per day, in three parts:
//
//   1. SCORE   yesterday's parked predictions against today's realised P&L.
//   2. EVOLVE  cull the worst mature individuals, breed replacements from the elite.
//   3. PREDICT tomorrow's P&L with every individual; park the answers.
//
// Scoring precedes prediction on purpose: a model is only ever graded on a call it made before
// the outcome existed. There is no snapshot, no replay and no in-sample path — which is exactly
// what the V10 out-of-sample instrument had to be built to work around, and why it could report
// 49.54% OOS against 61.09% in-sample on the same pool.

// Kaiming-init one MT1Net. Layer order and offsets mirror models.py MT1NET_LAYER_DEFS; the two
// trunks are identical in shape, so the market offsets serve both with a base of 0 / NT_P_A1_W.
static void mt1_init_weights(float* W, uint64_t seed) {
    PCG32 rng; rng.seed(mix_seed(seed));
    for (int half = 0; half < 2; half++) {
        float* T = W + half * NT_P_A1_W;
        kaiming_init(T + NT_M_A1_W, 20, 20, rng);
        kaiming_init(T + NT_M_A2_W, 20, 20, rng);
        kaiming_init(T + NT_M_B1_W,  6, 10, rng);
        kaiming_init(T + NT_M_B2_W,  4,  6, rng);
        kaiming_init(T + NT_M_C1_W,  5,  7, rng);
        kaiming_init(T + NT_M_C2_W,  4,  5, rng);
    }
    kaiming_init(W + NT_D1_W, 22, 56, rng);
    kaiming_init(W + NT_D2_W, 10, 23, rng);
    kaiming_init(W + NT_D3_W,  1, 10, rng);
}

// Kaiming-init one MT1CNet. Plain taper, so the layer list is short.
static inline MT1Init2 race_init(RaceArch a);

static void mt1s_init_weights(float* W, uint64_t seed) {
    PCG32 rng; rng.seed(mix_seed(seed));
    kaiming_init(W + SS_L1_W, MT1S_H1, MT1S_PER_SYM, rng);
    kaiming_init(W + SS_L2_W, MT1S_H2, MT1S_H1,      rng);
    kaiming_init(W + SS_L3_W, 1,       MT1S_H2,      rng);
}

static void mt1c_init_weights(float* W, uint64_t seed) {
    PCG32 rng; rng.seed(mix_seed(seed));
    kaiming_init(W + CN_L1_W, 64, MT1C_IN, rng);
    kaiming_init(W + CN_L2_W, 24, 64, rng);
    kaiming_init(W + CN_L3_W,  8, 24, rng);
    kaiming_init(W + CN_L4_W,  1,  8, rng);
}

static inline MT1Init2 race_init(RaceArch a) {
    return a == ARCH_MT1 ? mt1_init_weights
         : a == ARCH_MT1C ? mt1c_init_weights : mt1s_init_weights;
}

// Kaiming-init one MT2INet.

// Rank every slot by its rolling register. Immature models sort last regardless of score: with
// MT1_POOL_MIN_AGE = 8 calls behind it, a fresh model's single lucky prediction would otherwise
// top the pool, which is the max-of-N artifact this design exists to remove.
static void mt1_rank(const MT1PoolScratch& sc, int* order) {
    for (int i = 0; i < MT1_POOL_SLOTS; i++) order[i] = i;
    std::stable_sort(order, order + MT1_POOL_SLOTS, [&](int a, int b) {
        const bool ma = mt1_slot_mature(sc.meta[a]), mb = mt1_slot_mature(sc.meta[b]);
        if (ma != mb) return ma;                       // mature first
        return mt1_slot_better(sc.meta[a], sc.meta[b]);
    });
}

// Lineage shares, with hysteresis: a lineage over MT1_POOL_LINEAGE_CAP of the pool stops breeding
// until it falls back under MT1_POOL_LINEAGE_RESUME. This is the direct read on monoculture that
// the old pools left to be inferred from spread.
static void mt1_update_barred(MT1PoolScratch& sc, int* out_max_n, int* out_distinct) {
    uint32_t ids[MT1_POOL_SLOTS]; int cnt[MT1_POOL_SLOTS]; int n_ids = 0;
    for (int s = 0; s < MT1_POOL_SLOTS; s++) {
        const uint32_t id = sc.meta[s].lineage;
        int k = 0; for (; k < n_ids; k++) if (ids[k] == id) break;
        if (k == n_ids) { ids[n_ids] = id; cnt[n_ids] = 0; n_ids++; }
        cnt[k]++;
    }
    const int cap    = (int)(MT1_POOL_LINEAGE_CAP    * MT1_POOL_SLOTS);
    const int resume = (int)(MT1_POOL_LINEAGE_RESUME * MT1_POOL_SLOTS);
    uint32_t keep[16]; int n_keep = 0, max_n = 0;
    for (int k = 0; k < n_ids; k++) {
        if (cnt[k] > max_n) max_n = cnt[k];
        bool was_barred = false;
        for (int b = 0; b < sc.barred_count; b++) if (sc.barred[b] == ids[k]) { was_barred = true; break; }
        const bool bar = was_barred ? (cnt[k] > resume) : (cnt[k] > cap);
        if (bar && n_keep < 16) keep[n_keep++] = ids[k];
    }
    memcpy(sc.barred, keep, sizeof(uint32_t) * n_keep);
    sc.barred_count = n_keep;
    if (out_max_n)    *out_max_n    = max_n;
    if (out_distinct) *out_distinct = n_ids;
}

static bool mt1_is_barred(const MT1PoolScratch& sc, uint32_t lineage) {
    for (int b = 0; b < sc.barred_count; b++) if (sc.barred[b] == lineage) return true;
    return false;
}

// One MT1 day for one industry. `actual` is the P&L this industry's StockNN realised from the
// previous session's close to this one — the thing yesterday's prediction was a prediction OF.
// `have_actual` is false only on the first day of a pass, before any outcome exists.

static MT1DayResult mt1_step_day(int ind_i, MT1PoolScratch& sc, const float* in,
                                 float actual, bool have_actual, int actual_day, float sigma,
                                 MT1Forward fwd = mt1net_forward) {
    MT1DayResult r{};
    r.actual = actual;

    // ── 1. PREDICT ───────────────────────────────────────────────────────────────
    // in74 is built from the value curves BEFORE today's append, so it contains nothing after
    // close(t). `actual` is the move from close(t) to close(t+1). Predicting one from the other is
    // a genuine one-step-ahead forecast and leaks nothing.
    //
    // This used to score the PREVIOUS day's parked predictions first, on the reasoning that "a
    // model should only be graded on a call it made before the outcome existed". That confused
    // program order with INFORMATION order. The parked prediction knew only close(t-1) while
    // `actual` spans close(t)..close(t+1) — two days out, with a full day's hole in between — so
    // MT1 was solving a strictly harder problem than the one production asks. Out-of-sample is a
    // property of what the model can see, not of the order the loop happens to run in.
    for (int s = 0; s < MT1_POOL_SLOTS; s++)
        sc.pending[s] = mt1_pred(fwd(sc.slot(s), in, 0.f));
    sc.has_pending = true;
    sc.pending_day = actual_day;
    // The deployed model's call, captured BEFORE selection can reorder the pool — this is the
    // number that would actually have been handed to MT2 for today's allocation.
    r.pred0 = sc.pending[sc.best_slot];

    // ── 2. SCORE ─────────────────────────────────────────────────────────────────
    if (have_actual) {
        const float base = sc.baseline();
        const float fl   = sc.floor_v();
        // Baseline and floor are built from sessions strictly before the one being scored:
        // `actual` is pushed below, not above.
        assert(mt1_windows_are_causal(actual_day, sc.last_pushed_day, sc.last_pushed_day));
        r.baseline = base; r.floor_v = fl; r.scored = true;
        double sum = 0.0; float best = -1.f, worst = 2.f;
        for (int s = 0; s < MT1_POOL_SLOTS; s++) {
            const float v = mt1_score(actual, sc.pending[s], base, fl);
            mt1_slot_record(sc.meta[s], v);
            sum += v;
            if (v > best)  best  = v;
            if (v < worst) worst = v;
            if (s == sc.best_slot) r.score0 = v;
        }
        r.score_mean = (float)(sum / MT1_POOL_SLOTS);
        r.score_best = best;
        r.score_min  = worst;
        sc.has_pending = false;
        sc.push_actual(actual, actual_day);
    }

    // ── 3. EVOLVE ────────────────────────────────────────────────────────────────
    int order[MT1_POOL_SLOTS];
    mt1_rank(sc, order);
    int n_mature = 0;
    for (int s = 0; s < MT1_POOL_SLOTS; s++) if (mt1_slot_mature(sc.meta[s])) n_mature++;
    r.mature = n_mature;

    if (r.scored && n_mature > 0) {
        mt1_update_barred(sc, &r.lineage_max, &r.lineage_n);

        const int n_cull  = (int)(MT1_POOL_CULL_PCT  * n_mature + 0.5f);
        const int n_elite = std::max(1, (int)(MT1_POOL_ELITE_PCT * n_mature));

        int parents[MT1_POOL_SLOTS]; int n_par = 0;
        for (int k = 0; k < n_mature && n_par < n_elite; k++)
            if (!mt1_is_barred(sc, sc.meta[order[k]].lineage)) parents[n_par++] = order[k];
        if (n_par == 0) parents[n_par++] = order[0];

        for (int c = 0; c < n_cull; c++) {
            const int victim = order[n_mature - 1 - c];
            if (victim == sc.best_slot) continue;
            const int par = parents[c % n_par];
            const uint64_t seed = (uint64_t)actual_day * 0x9E3779B97F4A7C15ULL
                                ^ ((uint64_t)ind_i << 32) ^ ((uint64_t)victim * 2654435761ULL);
            const uint32_t age = sc.meta[victim].n_pred;
            mt1_mutate(sc.slot(par), sc.slot(victim), sc.n_params, sigma, seed);
            mt1_slot_init(sc.meta[victim], sc.meta[par].lineage);
            sc.retire_n++;
            sc.retire_age_sum += age;
            sc.retire_hist[mt1_life_bucket(age)]++;
            r.culled++; r.births++;
        }
        sc.best_slot = order[0];
    }
    sc.culled_today = r.culled;
    sc.births_today = r.births;
    return r;
}




// ── MT1 composite blend-pool step ────────────────────────────────────────────────

// Regenerate blend b into dst (deterministic per actual_day, ind_i, b)



// ── MT1 orchestrator ─────────────────────────────────────────────────────────────


// ── MT2 training step (replaces step_master) ────────────────────────────────────

static MasterResult step_mt2(MasterState& state, MT2Scratch& scratch,
                              const float in12[N_IND], const float actual_perf[N_IND],
                              int actual_day, int total_avail,
                              float sigma, bool* injected_out) {
    if (actual_day < MASTER_START_DAY) {
        if (injected_out) *injected_out = false;
        // Designated, not positional: MasterResult has NINE members and this supplies five.
        // A member inserted anywhere above shifts every later value silently, because they are
        // all floats and neither narrowing nor -Wmissing-field-initializers would stop it. The
        // IndResult twin of this bug shipped and was caught only because one float happened to
        // land in an int field.
        return MasterResult{};
    }

    {
        PCG32 seed_rng; seed_rng.seed(mix_seed((uint64_t)actual_day * 777017ULL + 99999ULL));
        for (int i = 0; i < N_SLOTS - ELITE_POOL; i++)
            scratch.mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    // Compute optimal tiers retroactively
    int opt_tier[N_IND] = {};
    {
        int pos_idx[N_IND]; int n_pos = 0;
        for (int i = 0; i < N_IND; i++)
            if (actual_perf[i] >= 0.f) pos_idx[n_pos++] = i;
        sort_index_prefix(pos_idx, n_pos,
                          [&actual_perf](int a, int b){ return actual_perf[a] < actual_perf[b]; });
        if (n_pos == 1) { opt_tier[pos_idx[0]] = 3; }
        else if (n_pos == 2) { opt_tier[pos_idx[0]] = 2; opt_tier[pos_idx[1]] = 3; }
        else {
            int base = n_pos/3, rem = n_pos%3;
            int n1 = base + (rem>=1?1:0), n2 = base + (rem>=2?1:0);
            for (int rank = 0; rank < n_pos; rank++) {
                int ind = pos_idx[rank];
                if      (rank < n1)      opt_tier[ind] = 1;
                else if (rank < n1 + n2) opt_tier[ind] = 2;
                else                     opt_tier[ind] = 3;
            }
        }
    }
    float ideal_pts = 0.f;
    for (int i = 0; i < N_IND; i++) ideal_pts += (float)opt_tier[i];

    float ref_cash = state.portfolios[0].cash;
    float ref_hold[N_IND];
    for (int i = 0; i < N_IND; i++) ref_hold[i] = state.portfolios[0].holdings[i];
    float baseline = ref_cash;
    for (int i = 0; i < N_IND; i++) baseline += ref_hold[i] * IND_UNIT_PRICE;

    for (int s = 0; s < N_SLOTS; s++) {
        state.portfolios[s].cash = ref_cash;
        for (int i = 0; i < N_IND; i++) state.portfolios[s].holdings[i] = ref_hold[i];
    }
    float pred_scores[N_SLOTS] = {}, port_vals[N_SLOTS] = {};
    int   slot_tiers[N_SLOTS][N_IND] = {};
    float out48[48];

    for (int slot = 0; slot < N_SLOTS; slot++) {
        const float* W;
        if (slot < ELITE_POOL) {
            W = scratch.elite(slot);
        } else {
            int mut_i = slot - ELITE_POOL, parent = mut_i / MUTATIONS_PER_PARENT;
            memcpy(scratch.mut_buf, scratch.elite(parent), MT2NN_PARAMS * sizeof(float));
            apply_gaussian(scratch.mut_buf, MT2NN_PARAMS, sigma, scratch.mut_seeds[mut_i]);
            W = scratch.mut_buf;
        }
        mt2_forward(W, in12, out48);

        int tier[N_IND];
        for (int i = 0; i < N_IND; i++) {
            const float* lg = out48 + i * 4; int best = 0;
            for (int k = 1; k < 4; k++) if (lg[k] > lg[best]) best = k;
            tier[i] = best; slot_tiers[slot][i] = best;
        }

        MasterPortfolio& port = state.portfolios[slot];
        int positives[N_IND]; int n_pos = 0;
        for (int i = 0; i < N_IND; i++) if (tier[i] > 0) positives[n_pos++] = i;
        float alloc[N_IND] = {};
        if (n_pos > 0) {
            sort_index_prefix(positives, n_pos,
                              [&tier](int a, int b){ return tier[a] < tier[b]; });
            float pool = port.cash;
            for (int i = 0; i < N_IND; i++) pool += port.holdings[i] * IND_UNIT_PRICE;
            if (n_pos == 1) {
                alloc[positives[0]] = TIER_WEIGHTS[3] / NULL_DENOM * pool;
            } else if (n_pos == 2) {
                alloc[positives[0]] = TIER_WEIGHTS[2] / NULL_DENOM * pool;
                alloc[positives[1]] = TIER_WEIGHTS[3] / NULL_DENOM * pool;
            } else {
                int base = n_pos/3, rem = n_pos%3;
                int n1 = base+(rem>=1?1:0), n2 = base+(rem>=2?1:0);
                int assigned[N_IND] = {};
                for (int rank = 0; rank < n_pos; rank++) {
                    int ind = positives[rank];
                    if      (rank < n1)      assigned[ind] = 1;
                    else if (rank < n1 + n2) assigned[ind] = 2;
                    else                     assigned[ind] = 3;
                }
                float total_w = 0.f;
                for (int k = 0; k < n_pos; k++) total_w += TIER_WEIGHTS[assigned[positives[k]]];
                if (total_w > 0.f)
                    for (int k = 0; k < n_pos; k++) {
                        int ind = positives[k];
                        alloc[ind] = TIER_WEIGHTS[assigned[ind]] / total_w * pool;
                    }
            }
        }
        for (int i = 0; i < N_IND; i++) {
            float cur_v = port.holdings[i] * IND_UNIT_PRICE, tgt_v = alloc[i];
            if (cur_v > tgt_v + 1e-6f) {
                float units = (cur_v - tgt_v) / IND_UNIT_PRICE;
                port.holdings[i] -= units; port.cash += sell_net(units, IND_UNIT_PRICE);
            }
        }
        for (int i = 0; i < N_IND; i++) {
            float cur_v = port.holdings[i] * IND_UNIT_PRICE, tgt_v = alloc[i];
            if (tgt_v > cur_v + 1e-6f) {
                float diff = std::min(tgt_v - cur_v, port.cash), units = diff / IND_UNIT_PRICE;
                port.holdings[i] += units; port.cash -= units * IND_UNIT_PRICE;
            }
        }
        for (int i = 0; i < N_IND; i++) port.holdings[i] *= (1.f + actual_perf[i]);

        float port_val = port.cash;
        for (int i = 0; i < N_IND; i++) port_val += port.holdings[i] * IND_UNIT_PRICE;
        port_vals[slot] = port_val;
        float pts = 0.f;
        for (int i = 0; i < N_IND; i++) {
            int pred = tier[i], opt = opt_tier[i];
            if (opt == 0) { if (pred > 0) pts += -2.f - 0.25f * pred; }
            else if (pred == 0) { pts -= (float)opt; }
            else if (pred <= opt) { pts += (float)pred; }
            else { pts += (float)opt - 0.25f * (float)(pred - opt); }
        }
        pred_scores[slot] = pts * 1e9f + port_val;
    }

    // ── Pool-consensus allocation diagnostic (read-only; never fed into training) ──
    // Wisdom-of-crowds: rank industries by the pool's mean tier vote, re-tier by the pool's
    // OWN observed mean tier-0 count (NOT opt_tier — that would leak the realized outcome),
    // and grade that consensus allocation with the normal pts formula. Two variants:
    //   flat   = every slot votes equally (all N_SLOTS)
    //   wtd    = elites weighted by look-behind reliability over the last MT2_LB_DAYS days
    float consensus_flat_pts = 0.f, consensus_wtd_pts = 0.f;
    {
        auto opt_from_perf = [](const float* perf, int* out_tier) {
            for (int i = 0; i < N_IND; i++) out_tier[i] = 0;
            int pos_idx[N_IND]; int n_pos = 0;
            for (int i = 0; i < N_IND; i++) if (perf[i] >= 0.f) pos_idx[n_pos++] = i;
            sort_index_prefix(pos_idx, n_pos, [&perf](int a, int b){ return perf[a] < perf[b]; });
            if (n_pos == 1) { out_tier[pos_idx[0]] = 3; }
            else if (n_pos == 2) { out_tier[pos_idx[0]] = 2; out_tier[pos_idx[1]] = 3; }
            else if (n_pos > 0) {
                int base = n_pos/3, rem = n_pos%3, n1 = base+(rem>=1?1:0), n2 = base+(rem>=2?1:0);
                for (int rank = 0; rank < n_pos; rank++) { int ind = pos_idx[rank];
                    if (rank < n1) out_tier[ind] = 1; else if (rank < n1+n2) out_tier[ind] = 2; else out_tier[ind] = 3; }
            }
        };
        auto score_ct = [&](const int* ct) -> float {
            float pts = 0.f;
            for (int i = 0; i < N_IND; i++) { int pred = ct[i], opt = opt_tier[i];
                if (opt == 0) { if (pred > 0) pts += -2.f - 0.25f*pred; }
                else if (pred == 0) { pts -= (float)opt; }
                else if (pred <= opt) { pts += (float)pred; }
                else { pts += (float)opt - 0.25f*(float)(pred-opt); } }
            return pts;
        };
        float sum_t0 = 0.f;
        for (int s = 0; s < N_SLOTS; s++) { int c = 0; for (int i = 0; i < N_IND; i++) if (slot_tiers[s][i] == 0) c++; sum_t0 += c; }
        float mean_t0 = sum_t0 / N_SLOTS;
        auto build_ct = [&](const float* ord, int* ct) {
            for (int i = 0; i < N_IND; i++) ct[i] = 0;
            int idx[N_IND]; for (int i = 0; i < N_IND; i++) idx[i] = i;
            sort_index_prefix(idx, N_IND, [&ord](int a, int b){ return ord[a] > ord[b]; });
            int n0 = (int)lroundf(mean_t0); if (n0 < 0) n0 = 0; if (n0 > N_IND) n0 = N_IND;
            int n_pos = N_IND - n0;
            if (n_pos > 0) { int base = n_pos/3, rem = n_pos%3, n1 = base+(rem>=1?1:0), n2 = base+(rem>=2?1:0), n3 = n_pos-n1-n2;
                for (int rank = 0; rank < n_pos; rank++) { int ind = idx[rank];
                    if (rank < n3) ct[ind] = 3; else if (rank < n3+n2) ct[ind] = 2; else ct[ind] = 1; } }
        };
        float ord_flat[N_IND] = {};
        for (int i = 0; i < N_IND; i++) { float s = 0.f; for (int sl = 0; sl < N_SLOTS; sl++) s += slot_tiers[sl][i]; ord_flat[i] = s / N_SLOTS; }
        int ct_flat[N_IND]; build_ct(ord_flat, ct_flat); consensus_flat_pts = score_ct(ct_flat);

        float ord_wtd[N_IND] = {};
        if (scratch.lb_count > 0) {
            float lbpts[ELITE_POOL] = {}, wmin = 1e30f, wsum = 0.f, w[ELITE_POOL] = {};
            float out48b[48]; int tt[N_IND], ot[N_IND];
            for (int e = 0; e < ELITE_POOL; e++) {
                float p = 0.f;
                for (int d = 0; d < scratch.lb_count; d++) {
                    mt2_forward(scratch.elite(e), scratch.lb_in12[d], out48b);
                    for (int i = 0; i < N_IND; i++) { const float* lg = out48b+i*4; int b = 0; for (int k=1;k<4;k++) if (lg[k]>lg[b]) b = k; tt[i] = b; }
                    opt_from_perf(scratch.lb_perf[d], ot);
                    for (int i = 0; i < N_IND; i++) { int pred = tt[i], opt = ot[i];
                        if (opt == 0) { if (pred > 0) p += -2.f-0.25f*pred; }
                        else if (pred == 0) { p -= (float)opt; }
                        else if (pred <= opt) { p += (float)pred; }
                        else { p += (float)opt - 0.25f*(float)(pred-opt); } }
                }
                lbpts[e] = p; if (p < wmin) wmin = p;
            }
            for (int e = 0; e < ELITE_POOL; e++) { w[e] = lbpts[e] - wmin + 1e-3f; wsum += w[e]; }
            for (int i = 0; i < N_IND; i++) { float s = 0.f; for (int e = 0; e < ELITE_POOL; e++) s += w[e]*slot_tiers[e][i]; ord_wtd[i] = wsum>0 ? s/wsum : 0.f; }
        } else {
            for (int i = 0; i < N_IND; i++) { float s = 0.f; for (int e = 0; e < ELITE_POOL; e++) s += slot_tiers[e][i]; ord_wtd[i] = s / ELITE_POOL; }
        }
        int ct_wtd[N_IND]; build_ct(ord_wtd, ct_wtd); consensus_wtd_pts = score_ct(ct_wtd);

        // Push today into the look-behind buffer AFTER using it (keeps weights leak-free).
        for (int i = 0; i < N_IND; i++) scratch.lb_in12[scratch.lb_head][i] = in12[i];
        for (int i = 0; i < N_IND; i++) scratch.lb_perf[scratch.lb_head][i] = actual_perf[i];
        scratch.lb_head = (scratch.lb_head + 1) % MT2_LB_DAYS;
        if (scratch.lb_count < MT2_LB_DAYS) scratch.lb_count++;
    }

    // ── Score history candidates ─────────────────────────────────────────────────
    int n_hist_mt2 = scratch.hist_count * HIST_PER_DAY;
    std::vector<float> hist_sc_mt2(n_hist_mt2, 0.f);
    {
        int total_h = HIST_DAYS * HIST_PER_DAY;
        for (int h = 0; h < n_hist_mt2; h++) {
            int oldest  = (scratch.hist_head * HIST_PER_DAY - n_hist_mt2 + total_h) % total_h;
            int abs_pos = (oldest + h) % total_h;
            const float* W = scratch.hist_buf + (size_t)abs_pos * MT2NN_PARAMS;
            mt2_forward(W, in12, out48);
            float pts_h = 0.f;
            for (int i = 0; i < N_IND; i++) {
                const float* lg = out48 + i * 4; int best = 0;
                for (int k = 1; k < 4; k++) if (lg[k] > lg[best]) best = k;
                int pred = best, opt = opt_tier[i];
                if (opt == 0) { if (pred > 0) pts_h += -2.f - 0.25f * pred; }
                else if (pred == 0) { pts_h -= (float)opt; }
                else if (pred <= opt) { pts_h += (float)pred; }
                else { pts_h += (float)opt - 0.25f * (float)(pred - opt); }
            }
            hist_sc_mt2[h] = pts_h * 1e9f;  // no port_val tiebreak for history
        }
    }

    int tier_counts[4] = {};
    for (int i = 0; i < N_IND; i++) tier_counts[slot_tiers[0][i]]++;
    int best_slot_idx = (int)(std::max_element(pred_scores, pred_scores + N_SLOTS) - pred_scores);
    float best_sc   = pred_scores[best_slot_idx];
    float best_pts_v  = (best_sc - port_vals[best_slot_idx]) / 1e9f;
    float slot0_pts_v = (pred_scores[0] - port_vals[0]) / 1e9f;

    auto fmt_pts = [](float v) -> std::string {
        char buf[32]; snprintf(buf, sizeof(buf), "%+.2f", v); return buf;
    };
    log_msg(std::string("[mt2     ] Day ") + std::to_string(actual_day+1) +
            "/" + std::to_string(total_avail) +
            " | best_pts=" + fmt_pts(best_pts_v) +
            " slot0_pts=" + fmt_pts(slot0_pts_v) +
            " | t0=" + std::to_string(tier_counts[0]) +
            " t1=" + std::to_string(tier_counts[1]) +
            " t2=" + std::to_string(tier_counts[2]) +
            " t3=" + std::to_string(tier_counts[3]));

    if (baseline < MST_STARTING_CASH * 0.9f) {
        // Portfolio simulation drifted — reset for next day but still select on pts
        log_msg("[mt2     ] Portfolio reset (baseline dropped below 90%)");
        for (int s = 0; s < N_SLOTS; s++) {
            state.portfolios[s].cash = MST_STARTING_CASH;
            for (int i = 0; i < N_IND; i++) state.portfolios[s].holdings[i] = 0.f;
        }
    }

    MasterPortfolio slot0_own = state.portfolios[0];

    // Enforce 10-day post-injection hold to prevent consecutive diversity washes
    if (state.mt2_injection_hold > 0) --state.mt2_injection_hold;
    bool injection_suppressed = (state.mt2_injection_hold > 0);

    // Count slots below injection threshold (75%-of-pool criterion)
    int below_thresh = 0;
    for (int s = 0; s < N_SLOTS; s++) {
        if ((pred_scores[s] - port_vals[s]) / 1e9f < MT2_INJ_THRESHOLD) ++below_thresh;
    }
    bool inject_triggered = (below_thresh >= MT2_INJ_MIN_BELOW) && !injection_suppressed;

    bool injected = false;
    if (!inject_triggered) {
        float mean_ps = 0.f;
        for (int s = 0; s < N_SLOTS; s++) mean_ps += pred_scores[s];
        mean_ps /= N_SLOTS;
        float var_ps = 0.f;
        for (int s = 0; s < N_SLOTS; s++) { float d = pred_scores[s]-mean_ps; var_ps += d*d; }
        float pool_floor = mean_ps - sqrtf(var_ps / N_SLOTS);

        std::vector<std::pair<float,int>> surviving;
        for (int s = 0; s < N_SLOTS; s++)
            if (pred_scores[s] >= pool_floor) surviving.push_back({pred_scores[s], s});
        if (surviving.empty()) surviving.push_back({pred_scores[0], 0});
        // Add history candidates unconditionally (no pool_floor filter)
        for (int h = 0; h < n_hist_mt2; h++)
            surviving.push_back({hist_sc_mt2[h], N_SLOTS + h});  // idx >= N_SLOTS → history
        std::sort(surviving.begin(), surviving.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        int n_top = std::min((int)surviving.size(), ELITE_COUNT);

        int   src_rank[ELITE_COUNT] = {};
        float src_val [ELITE_COUNT] = {};
        for (int k = 0;    k < n_top;       k++) { src_rank[k] = surviving[k].second; src_val[k] = surviving[k].first; }
        for (int k = n_top; k < ELITE_COUNT; k++) { src_rank[k] = src_rank[0]; src_val[k] = src_val[0]; }

        float w5[5], w10[10], w15[15];
        int n5 = std::min(n_top,5), n10 = std::min(n_top,10), n15 = std::min(n_top,15);
        normalize_weights(src_val, w5,  n5);
        normalize_weights(src_val, w10, n10);
        normalize_weights(src_val, w15, n15);

        MasterPortfolio new_mports[ELITE_POOL];
        // For history slots we have no live portfolio — use slot0_own as starting point
        auto get_mport = [&](int sl) -> const MasterPortfolio& {
            return (sl < N_SLOTS) ? state.portfolios[sl] : slot0_own;
        };
        const MasterPortfolio* mp5[5], *mp10[10], *mp15[15];
        for (int k=0;k<n5;k++)  mp5[k]  = &get_mport(src_rank[k]);
        for (int k=0;k<n10;k++) mp10[k] = &get_mport(src_rank[k]);
        for (int k=0;k<n15;k++) mp15[k] = &get_mport(src_rank[k]);
        MasterPortfolio wp5={}, wp10={}, wp15={};
        wavg_mst_portfolio(mp5,  w5,  n5,  wp5);
        wavg_mst_portfolio(mp10, w10, n10, wp10);
        wavg_mst_portfolio(mp15, w15, n15, wp15);

        for (int k = 0; k < n_top; k++) {
            int slot = src_rank[k];
            if (slot >= N_SLOTS) {
                // History model — copy weights from hist_buf
                int h = slot - N_SLOTS;
                int total_h = HIST_DAYS * HIST_PER_DAY;
                int oldest  = (scratch.hist_head * HIST_PER_DAY - n_hist_mt2 + total_h) % total_h;
                int abs_pos = (oldest + h) % total_h;
                memcpy(scratch.new_elite(k), scratch.hist_buf + (size_t)abs_pos * MT2NN_PARAMS,
                       MT2NN_PARAMS * sizeof(float));
                new_mports[k] = slot0_own;  // history models start from slot0's portfolio
            } else if (slot < ELITE_POOL) {
                memcpy(scratch.new_elite(k), scratch.elite(slot), MT2NN_PARAMS * sizeof(float));
                new_mports[k] = state.portfolios[slot];
            } else {
                int mut_i = slot - ELITE_POOL, parent = mut_i / MUTATIONS_PER_PARENT;
                memcpy(scratch.new_elite(k), scratch.elite(parent), MT2NN_PARAMS * sizeof(float));
                apply_gaussian(scratch.new_elite(k), MT2NN_PARAMS, sigma, scratch.mut_seeds[mut_i]);
                new_mports[k] = state.portfolios[slot];
            }
        }
        for (int k = n_top; k < ELITE_COUNT; k++) {
            memcpy(scratch.new_elite(k), scratch.new_elite(0), MT2NN_PARAMS * sizeof(float));
            new_mports[k] = new_mports[0];
        }
        int seq[ELITE_COUNT]; for (int k = 0; k < ELITE_COUNT; k++) seq[k] = k;
        wavg_weights_flat(scratch.new_elites, MT2NN_PARAMS, seq, w5,  n5,  scratch.wavg(0));
        wavg_weights_flat(scratch.new_elites, MT2NN_PARAMS, seq, w10, n10, scratch.wavg(1));
        wavg_weights_flat(scratch.new_elites, MT2NN_PARAMS, seq, w15, n15, scratch.wavg(2));
        memcpy(scratch.new_elite(ELITE_COUNT),     scratch.wavg(0), MT2NN_PARAMS * sizeof(float));
        memcpy(scratch.new_elite(ELITE_COUNT + 1), scratch.wavg(1), MT2NN_PARAMS * sizeof(float));
        memcpy(scratch.new_elite(ELITE_COUNT + 2), scratch.wavg(2), MT2NN_PARAMS * sizeof(float));
        new_mports[ELITE_COUNT] = wp5; new_mports[ELITE_COUNT+1] = wp10; new_mports[ELITE_COUNT+2] = wp15;

        for (int k = 0; k < ELITE_POOL; k++) {
            memcpy(scratch.elite(k), scratch.new_elite(k), MT2NN_PARAMS * sizeof(float));
            state.portfolios[k] = new_mports[k];
        }
        for (int mut_i = 0; mut_i < N_SLOTS - ELITE_POOL; mut_i++)
            state.portfolios[ELITE_POOL + mut_i] = state.portfolios[mut_i / MUTATIONS_PER_PARENT];
    } else {
        injected = true;
        state.mt2_injection_hold = 10;  // suppress re-injection for 10 days
        log_msg("[mt2     ] " + std::to_string(below_thresh) + "/" + std::to_string(N_SLOTS) +
                " slots < " + fmt_pts(MT2_INJ_THRESHOLD) + " — injecting diversity");
        PCG32 div_rng; div_rng.seed(mix_seed((uint64_t)actual_day * 55555ULL + 77777ULL));
        int half = ELITE_COUNT / 2;
        for (int k = half; k < ELITE_COUNT; k++) {
            init_mt2_weights(scratch.mut_buf, div_rng);
            for (int p = 0; p < MT2NN_PARAMS; p++)
                scratch.elite(k)[p] = 0.5f * scratch.elite(k-half)[p] + 0.5f * scratch.mut_buf[p];
            state.portfolios[k] = state.portfolios[k - half];
        }
    }

    state.portfolios[0] = slot0_own;

    // ── Save top HIST_ELITE direct elites + HIST_WAVG wavg blends to MT2 history ─
    {
        int& head  = scratch.hist_head;
        int& count = scratch.hist_count;
        for (int k = 0; k < HIST_ELITE; k++)
            memcpy(scratch.hist(head, k), scratch.elite(k), MT2NN_PARAMS * sizeof(float));
        for (int k = 0; k < HIST_WAVG; k++)
            memcpy(scratch.hist(head, HIST_ELITE + k), scratch.elite(ELITE_COUNT + k),
                   MT2NN_PARAMS * sizeof(float));
        head = (head + 1) % HIST_DAYS;
        if (count < HIST_DAYS) count++;
    }

    float elite_max_pts = -1e9f, elite_min_pts = 1e9f, elite_mean_pts = 0.f;
    for (int s = 0; s < ELITE_COUNT; s++) {
        float p = (pred_scores[s] - port_vals[s]) / 1e9f;
        if (p > elite_max_pts) elite_max_pts = p;
        if (p < elite_min_pts) elite_min_pts = p;
        elite_mean_pts += p;
    }
    elite_mean_pts /= ELITE_COUNT;

    if (injected_out) *injected_out = injected;
    // Every member named, including the two that are 0 here on purpose: slot0_pts_pf and
    // slot0_pts_mkt are filled in by process_block. Writing the zeros out rather than leaning on
    // the default keeps -Wextra silent for a real reason instead of a suppressed one, and states
    // the intent where the next reader will look.
    return MasterResult{
        .best_pts           = best_pts_v,
        .elite_max_pts      = elite_max_pts,
        .elite_min_pts      = elite_min_pts,
        .elite_mean_pts     = elite_mean_pts,
        .ideal_pts          = ideal_pts,
        .consensus_flat_pts = consensus_flat_pts,
        .consensus_wtd_pts  = consensus_wtd_pts,
        .slot0_pts_pf       = 0.f,          // set later in process_block
        .slot0_pts_mkt      = 0.f,          // set later in process_block
    };
}

// ── History update (main thread after workers finish) ────────────────────────────

static void update_hist_sym(SymHist& h, const OHLCV& d) {
    if (!d.valid) return;
    HistEntry e;
    e.data[0] = d.open; e.data[1] = d.close; e.data[2] = d.high;
    e.data[3] = d.low;  e.data[4] = d.volume;
    const HistEntry* prev = h.newest();
    if (prev) {
        e.data[5] = d.open   - prev->data[0];
        e.data[6] = d.close  - prev->data[1];
        e.data[7] = d.high   - prev->data[2];
        e.data[8] = d.low    - prev->data[3];
        e.data[9] = d.volume - prev->data[4];
    } else {
        for (int k = 5; k < 10; k++) e.data[k] = 0.f;
    }
    h.push(e);
}

// ── Model persistence ──────────────────────────────────────────────────────────

static void load_ind_history(const std::string& dir, int ind_i, WorkerScratch& scratch) {
    std::string path = dir + "/" + g_ind_names[ind_i] + "_hist.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { scratch.hist_head = 0; scratch.hist_count = 0; return; }
    int meta[2] = {};
    if (fread(meta, sizeof(int), 2, f) == 2) {
        scratch.hist_head  = std::max(0, std::min(meta[0], HIST_DAYS - 1));
        scratch.hist_count = std::max(0, std::min(meta[1], HIST_DAYS));
    } else {
        scratch.hist_head = 0; scratch.hist_count = 0;
    }
    size_t n = (size_t)HIST_DAYS * HIST_PER_DAY * STOCKNN_PARAMS;
    fread(scratch.hist_buf, sizeof(float), n, f);
    fclose(f);
}

static void save_ind_history(const std::string& dir, int ind_i, const WorkerScratch& scratch) {
    std::string path = dir + "/" + g_ind_names[ind_i] + "_hist.bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { log_msg("WARNING: could not save history for " + g_ind_names[ind_i]); return; }
    int meta[2] = {scratch.hist_head, scratch.hist_count};
    fwrite(meta, sizeof(int), 2, f);
    size_t n = (size_t)HIST_DAYS * HIST_PER_DAY * STOCKNN_PARAMS;
    fwrite(scratch.hist_buf, sizeof(float), n, f);
    fclose(f);
}

static void save_industry_elites(const std::string& dir, int ind_i,
                                  const float* elite_buf) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        std::string path = elite_path(dir, g_ind_names[ind_i].c_str(), slot);
        if (!save_bin(path, elite_buf + (size_t)slot * STOCKNN_PARAMS, STOCKNN_PARAMS))
            log_msg("WARNING: could not save " + path);
    }
}

// ── Pass-boundary seeding (v0.6.3.0) ──────────────────────────────────────────
// At each pass boundary, judge the finishing pass against the standing champion on slot-0's
// percent portfolio change over the last PASS_JUDGE_DAYS, PER INDUSTRY, and seed the next pass
// with a rank-preserving proportional interleave of the two 20-elite sets.
//
// ORDERING HAZARD: load_or_init_industry reads, and save_industry_elites writes, the SAME output
// directory — so at the end of pass N that directory IS pass N's elites. Writing the blend into it
// destroys exactly the models the crowning step then needs. Pass N's elites are therefore staged
// to a scratch directory first, on disk rather than in RAM: two full sets is 142 MB per industry
// on a box already mlocking ~720 MB.
//
// The champion is a PER-INDUSTRY COMPOSITE — energy's may come from pass 2 while financials' comes
// from pass 1 — so no pass directory or log section describes it. That is what pass_reference.csv
// and champion/ exist for. See PASS_SEEDING.md.

struct PassRefRow { int champ_pass = 0; float champ_pct = 0.f; bool found = false; };

// Last recorded row for this industry. The carried-forward champion is new_champ_pass/new_champ_pct.
static PassRefRow read_pass_ref(const std::string& path, const std::string& ind) {
    PassRefRow r;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return r;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char ind_buf[64], regime[32];
        int p, ds, de, cp, chp, slots, ncp;
        float cpct, chpct, share, npct;
        if (sscanf(line, "%d,%63[^,],%d,%d,%d,%f,%d,%f,%31[^,],%f,%d,%d,%f",
                   &p, ind_buf, &ds, &de, &cp, &cpct, &chp, &chpct,
                   regime, &share, &slots, &ncp, &npct) == 13 && ind == ind_buf) {
            r.champ_pass = ncp; r.champ_pct = npct; r.found = true;
        }
    }
    fclose(f);
    return r;
}

static bool copy_elites(const std::string& from, const std::string& to, const char* ind) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        std::error_code ec;
        fs::copy_file(elite_path(from, ind, slot), elite_path(to, ind, slot),
                      fs::copy_options::overwrite_existing, ec);
        if (ec) return false;
    }
    // The gradient-trained MT1 travels WITH the elites it was raised against. Missing is not an
    // error: a pass that ran before MT1_BP_START_DAY, or with the feature off, has none to copy,
    // and the elites are still a valid champion.
    {
        std::error_code ec2;
        const std::string f = from + "/mt1bp_" + ind + ".bin";
        if (fs::exists(f, ec2))
            fs::copy_file(f, to + "/mt1bp_" + ind + ".bin",
                          fs::copy_options::overwrite_existing, ec2);
    }
    return true;
}

// seed_next=false on the final pass: crown and record, but skip the blend. Otherwise the run
// would end with out_dir holding a seed built for a pass that never runs — neither the champion
// nor the final pass's models. The deliverable is champion/ either way.
static void pass_boundary(const std::string& out_dir, int pass_num,
                          int day_start, int day_end,
                          const float* judge_start, const float* judge_end,
                          bool seed_next) {
    const std::string champ_dir = out_dir + "/champion";
    const std::string stage_dir = out_dir + "/stage";
    const std::string ref_path  = out_dir + "/pass_reference.csv";
    std::error_code ec;
    fs::create_directories(champ_dir, ec);
    fs::create_directories(stage_dir, ec);

    // Read every industry's standing champion BEFORE opening the file for append, so buffered
    // writes from this same boundary can never be read back as prior state.
    PassRefRow prev[N_IND];
    for (int i = 0; i < N_IND; i++) prev[i] = read_pass_ref(ref_path, g_ind_names[i]);

    const bool fresh = !fs::exists(ref_path);
    FILE* csv = fopen(ref_path.c_str(), "a");
    if (!csv) {
        log_msg("WARNING: cannot write " + ref_path + " — pass seeding skipped at this boundary");
        return;
    }
    if (fresh)
        fprintf(csv, "# pass_reference v%d — see PASS_SEEDING.md\n"
                     "pass,industry,day_start,day_end,champ_pass,champ_pct,chal_pass,chal_pct,"
                     "regime,share_new,slots_new,new_champ_pass,new_champ_pct\n", PASS_REF_VERSION);

    for (int i = 0; i < N_IND; i++) {
        const char* ind  = g_ind_names[i].c_str();
        const bool  have = judge_start[i] > 1e-6f;
        const float chal_pct = have ? (judge_end[i] / judge_start[i] - 1.f) : 0.f;
        char msg[320];

        if (!copy_elites(out_dir, stage_dir, ind)) {
            log_msg(std::string("[") + IND_SHORT[i] +
                    "]   pass-seed: cannot stage elites — seed left unchanged");
            continue;
        }

        bool champ_ok = prev[i].found;
        for (int slot = 0; slot < ELITE_POOL && champ_ok; slot++)
            if (!fs::exists(elite_path(champ_dir, ind, slot))) champ_ok = false;
        if (prev[i].found && !champ_ok)
            log_msg(std::string("[") + IND_SHORT[i] +
                    "]   pass-seed: champion weights MISSING — falling back to 100% from this pass");

        if (!champ_ok || !have) {
            // First boundary, missing champion, or no usable metric: this pass becomes the
            // champion outright and the seed is left as-is — the pre-v0.6.3.0 behaviour.
            copy_elites(stage_dir, champ_dir, ind);
            fprintf(csv, "%d,%s,%d,%d,%d,%+.6f,%d,%+.6f,%s,%.4f,%d,%d,%+.6f\n",
                    pass_num, ind, day_start + 1, day_end, pass_num, chal_pct, 0, 0.f,
                    "seed", 1.0, ELITE_POOL, pass_num, chal_pct);
            snprintf(msg, sizeof(msg), "   pass-seed: champion = pass %d (%+.2f%%), seed unchanged",
                     pass_num, chal_pct * 100.f);
            log_msg(std::string("[") + IND_SHORT[i] + "]" + msg);
            continue;
        }

        const double share = pass_share_new(prev[i].champ_pct, chal_pct);
        int src[ELITE_POOL], idx[ELITE_POOL];
        pass_interleave(share, ELITE_POOL, src, idx);
        int slots_new = 0;
        for (int k = 0; k < ELITE_POOL; k++) slots_new += (src[k] == 1);

        // Blend champion + staged challenger into the live directory — next pass's seed.
        if (seed_next) {
            for (int k = 0; k < ELITE_POOL; k++) {
                const std::string from = (src[k] == 0) ? elite_path(champ_dir, ind, idx[k])
                                                       : elite_path(stage_dir, ind, idx[k]);
                fs::copy_file(from, elite_path(out_dir, ind, k),
                              fs::copy_options::overwrite_existing, ec);
            }
        }

        const bool  chal_wins = chal_pct > prev[i].champ_pct;
        const int   nc_pass   = chal_wins ? pass_num : prev[i].champ_pass;
        const float nc_pct    = chal_wins ? chal_pct : prev[i].champ_pct;
        if (chal_wins) copy_elites(stage_dir, champ_dir, ind);

        fprintf(csv, "%d,%s,%d,%d,%d,%+.6f,%d,%+.6f,%s,%.4f,%d,%d,%+.6f\n",
                pass_num, ind, day_start + 1, day_end,
                prev[i].champ_pass, prev[i].champ_pct, pass_num, chal_pct,
                pass_regime(prev[i].champ_pct, chal_pct), share, slots_new, nc_pass, nc_pct);

        snprintf(msg, sizeof(msg),
                 "   pass-seed: p%d %+.2f%% vs p%d %+.2f%% [%s] -> %d/%d slots from p%d%s, champion = p%d",
                 prev[i].champ_pass, prev[i].champ_pct * 100.f, pass_num, chal_pct * 100.f,
                 pass_regime(prev[i].champ_pct, chal_pct), slots_new, ELITE_POOL, pass_num,
                 seed_next ? "" : " (final pass — not seeded)", nc_pass);
        log_msg(std::string("[") + IND_SHORT[i] + "]" + msg);
    }
    fclose(csv);
    fs::remove_all(stage_dir, ec);
}

// save_master_elites() removed in v0.8.1.10: unreferenced (-Wunused-function).

static void load_or_init_industry(const std::string& dir, const std::string& load_dir,
                                   int ind_i, float* elite_buf, int actual_day) {
    PCG32 rng; rng.seed(mix_seed((uint64_t)ind_i * 987654321ULL + 123456789ULL));

    // Control mode: fresh random weights every day, never loaded. Seeded by DAY as well as
    // industry so the parents genuinely differ day to day — the accidental --no-save version
    // re-drew the same models each day, which left a fixed pool that selection could still
    // exploit. This is the cleaner null.
    if (g_control_untrained) {
        PCG32 crng;
        crng.seed(mix_seed((uint64_t)actual_day * 7919ULL + (uint64_t)ind_i * 104729ULL + 31ULL));
        for (int slot = 0; slot < ELITE_POOL; slot++)
            init_stock_weights(elite_buf + (size_t)slot * STOCKNN_PARAMS, crng);
        return;
    }
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        float* e = elite_buf + (size_t)slot * STOCKNN_PARAMS;
        // WORKING STORE FIRST, seed second. The order used to be reversed, and because this runs
        // at the top of EVERY day a populated --load-dir was re-read daily — so a run seeded from
        // a previous one reloaded that seed every day and never made progress. --load-dir is a
        // seed: it should only be reached while the working store has nothing for this slot.
        bool loaded = false;
        {
            std::string p = elite_path(dir, g_ind_names[ind_i].c_str(), slot);
            loaded = load_bin(p, e, STOCKNN_PARAMS);
        }
        if (!loaded && !load_dir.empty()) {
            std::string p = elite_path(load_dir, g_ind_names[ind_i].c_str(), slot);
            loaded = load_bin(p, e, STOCKNN_PARAMS);
        }
        if (!loaded) {
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   Slot " +
                    std::to_string(slot) + ": random init");
            init_stock_weights(e, rng);
        }
    }
}

// load_or_init_master() removed in v0.8.1.10: unreferenced (-Wunused-function).

// ── MT1/MT2 persistence ──────────────────────────────────────────────────────────

// All 200 individuals, plus a metadata sidecar carrying each slot's rolling score register,
// age and lineage. The sidecar is a SEPARATE file on purpose: save_bin/load_bin are raw
// headerless float arrays validated by exact element count, so appending metadata to a weight
// file makes the loader reject it and fall back to random init SILENTLY — the failure mode that

// `load_dir` is a SEED, consulted only when `dir` has nothing — the same contract as
// load_or_init_industry. Checking it every day is what made --load-dir runs stand still before
// v0.6.6.0.
using MT1Init = void (*)(float*, uint64_t);

// Spearman rank correlation. Used to ask the only question the per-model rows can answer:
// given each model's own orders, can MT1 RANK the models by what they actually earned? Every slot
// resets to slot 0's portfolio, so the market move is common and the ranking IS the trading
// outcome. Ordinal ranks; exact ties are vanishingly rare in float P&L and cost at most a hair of
// precision when they happen.
static void bp_avg_ranks(const float* v, int n, std::vector<int>& idx, std::vector<float>& out)
{
    idx.resize(n); out.resize(n);
    for (int k = 0; k < n; k++) idx[k] = k;
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return v[x] < v[y]; });
    // AVERAGE ranks over ties. Ordinal ranks were a defect, not a simplification: with every
    // prediction equal -- which is exactly what the CONTROL variant produces, since it hands all
    // 200 models slot 0's input -- the sort leaves them in original order and the "ranks" come
    // back as 0,1,2,...,199, i.e. the SLOT INDEX. That is not arbitrary: slots 0-16 are the
    // selected elites, 17-19 the blends, 20-199 this day's mutation children. So the control was
    // measuring "do elite slots beat mutation children", a very real +0.039, instead of reading
    // the 0.0000 that proves the harness does not leak. Averaging ties makes a constant vector
    // produce zero rank variance, which the caller then skips.
    int k = 0;
    while (k < n) {
        int j = k + 1;
        while (j < n && !(v[idx[k]] < v[idx[j]]) && !(v[idx[j]] < v[idx[k]])) j++;
        const float avg = 0.5f * (float)(k + j - 1);
        for (int t = k; t < j; t++) out[idx[t]] = avg;
        k = j;
    }
}

static float bp_rank_corr(const float* a, const float* b, int n)
{
    if (n < 3) return NAN;
    static thread_local std::vector<int> ia, ib;
    static thread_local std::vector<float> ra, rb;
    bp_avg_ranks(a, n, ia, ra);
    bp_avg_ranks(b, n, ib, rb);
    double ma = 0.0, mb = 0.0;
    for (int k = 0; k < n; k++) { ma += ra[k]; mb += rb[k]; }
    ma /= n; mb /= n;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int k = 0; k < n; k++) {
        const double xa = ra[k] - ma, xb = rb[k] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    if (da <= 0.0 || db <= 0.0) return NAN;
    return (float)(num / sqrt(da * db));
}

// ── memory pressure, logged so the fallback decision is data-driven ──────────────
//
// The box has under 2 GB and the trainer already mlocks ~720 MB (elite_buf + hist_buf are
// PER-WORKER, ~315 MB each). Adding the MT1 race pools costs a measured 127 MB, which fits but
// not generously. The lever if it stops fitting is `--workers 1`: it halves the mlocked StockNN
// buffers, freeing ~315 MB -- more than the pools take -- at roughly 2x wall-clock.
//
// This logs the numbers that decide it, once per block, so churn shows up in the training log
// instead of needing someone to be watching `free -m` at the right moment. It does NOT switch
// workers on its own: changing thread count mid-run would silently change the work distribution
// and make the pass incomparable to the one before it.
struct MemStat { long rss_mb = 0, avail_mb = 0, swap_used_mb = 0; };

static MemStat read_mem_stat()
{
    MemStat m;
    char line[256];
    if (FILE* f = fopen("/proc/self/status", "r")) {
        while (fgets(line, sizeof(line), f)) {
            long kb;
            if (sscanf(line, "VmRSS: %ld kB", &kb) == 1) { m.rss_mb = kb / 1024; break; }
        }
        fclose(f);
    }
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        long total_swap = 0, free_swap = 0, kb;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) m.avail_mb = kb / 1024;
            else if (sscanf(line, "SwapTotal: %ld kB", &kb) == 1) total_swap = kb;
            else if (sscanf(line, "SwapFree: %ld kB", &kb) == 1) free_swap = kb;
        }
        fclose(f);
        m.swap_used_mb = (total_swap - free_swap) / 1024;
    }
    return m;
}

// Warns when available memory is low enough that the run is at risk of thrashing, and names the
// remedy. The threshold is deliberately generous: by the time a 2-core box with 1.9 GB is
// swapping hard, a training day takes minutes instead of seconds.
static void log_mem_stat(int workers)
{
    static long prev_swap = -1;
    const MemStat m = read_mem_stat();
    const long dswap = (prev_swap < 0) ? 0 : m.swap_used_mb - prev_swap;
    prev_swap = m.swap_used_mb;
    char buf[256];
    snprintf(buf, sizeof(buf), "   mem: rss %ld MB | avail %ld MB | swap %ld MB (%+ld)",
             m.rss_mb, m.avail_mb, m.swap_used_mb, dswap);
    log_msg(buf);
    if (m.avail_mb < 150 || dswap > 100) {
        snprintf(buf, sizeof(buf),
                 "   WARNING: memory is tight (avail %ld MB, swap %+ld MB this block).%s",
                 m.avail_mb, dswap,
                 workers > 1 ? "  Re-run with --workers 1 to free ~315 MB of mlocked buffers."
                             : "  Already at 1 worker.");
        log_msg(buf);
    }
}

// Per-PASS snapshot of every race entry's final model, kept rather than overwritten so the
// weights can be compared afterwards -- evolutionary against gradient for the same architecture,
// and pass against pass for the same entry. Both sides of a pair share a layout (see mt1_grad.h),
// so W[k] denotes the same parameter in both and an element-wise diff is meaningful.
// Name carries the entry so a directory listing is self-describing.
static std::string race_weight_path(const std::string& dir, int v, int ind_i, int pass_num) {
    std::string nm = RACE[v].name;
    for (char& c : nm) if (c == ' ') c = '_';
    while (!nm.empty() && nm.back() == '_') nm.pop_back();
    return dir + "/race_p" + std::to_string(pass_num) + "_" + nm + "_" +
           g_ind_names[ind_i] + ".bin";
}




// already cost this project a full run (see the v0.6.0.0 STOCKNN_PARAMS note in CLAUDE.md).
static constexpr uint32_t MT1_META_MAGIC   = 0x4D543150u;   // "MT1P"
static constexpr uint32_t MT1_META_VERSION = 1u;

static std::string mt1_pool_path(const std::string& dir, int ind_i, int slot,
                                 const char* tag = "mt1") {
    return dir + "/" + tag + "_" + g_ind_names[ind_i] + "_slot_" + std::to_string(slot) + ".bin";
}
static std::string mt1_meta_path(const std::string& dir, int ind_i, const char* tag = "mt1") {
    return dir + "/" + tag + "_" + g_ind_names[ind_i] + "_meta.bin";
}

// Written every block, which is what makes evolutionary carryover work: load_or_init_mt1_pool
// reads these back at the next pass start so the pool continues rather than restarting.
static void save_mt1_pool(const std::string& dir, int ind_i, const MT1PoolScratch& sc,
                          const char* tag = "mt1") {
    for (int s = 0; s < MT1_POOL_SLOTS; s++)
        save_bin(mt1_pool_path(dir, ind_i, s, tag), sc.slot(s), sc.n_params);
    FILE* f = fopen(mt1_meta_path(dir, ind_i, tag).c_str(), "wb");
    if (!f) return;
    uint32_t hdr[4] = {MT1_META_MAGIC, MT1_META_VERSION, (uint32_t)MT1_POOL_SLOTS,
                       (uint32_t)MT1_SCORE_HIST};
    fwrite(hdr, sizeof(uint32_t), 4, f);
    fwrite(sc.meta, sizeof(MT1SlotMeta), MT1_POOL_SLOTS, f);
    int32_t ints[5] = {sc.best_slot, sc.actual_head, sc.actual_count, sc.last_pushed_day,
                       (int32_t)sc.next_lineage};
    fwrite(ints, sizeof(int32_t), 5, f);
    fwrite(sc.actual_buf, sizeof(float), MT1_BASELINE_DAYS, f);
    fclose(f);
}

static void load_or_init_mt1_pool(const std::string& dir, const std::string& load_dir,
                                  int ind_i, MT1PoolScratch& sc,
                                  const char* tag = "mt1", MT1Init init = mt1_init_weights) {
    int loaded = 0;
    for (int s = 0; s < MT1_POOL_SLOTS; s++) {
        if (load_bin(mt1_pool_path(dir, ind_i, s, tag), sc.slot(s), sc.n_params) ||
            (!load_dir.empty() &&
             load_bin(mt1_pool_path(load_dir, ind_i, s, tag), sc.slot(s), sc.n_params))) {
            loaded++;
        } else {
            init(sc.slot(s), 0xB1A5E0000000ULL ^ ((uint64_t)ind_i << 20) ^ (uint64_t)s
                            ^ ((uint64_t)tag[0] << 40));
            mt1_slot_init(sc.meta[s], sc.next_lineage++);
        }
    }
    if (loaded == 0) {
        log_msg(std::string(tag) + " " + g_ind_names[ind_i] + ": random init (200 slots, "
                + std::to_string(sc.n_params) + " params)");
        return;
    }
    // Metadata is optional: weights without it are usable, they just start the registers empty,
    // which costs MT1_POOL_MIN_AGE days of maturity and nothing else.
    FILE* f = fopen(mt1_meta_path(dir, ind_i, tag).c_str(), "rb");
    if (!f && !load_dir.empty()) f = fopen(mt1_meta_path(load_dir, ind_i, tag).c_str(), "rb");
    if (!f) { for (int s = 0; s < MT1_POOL_SLOTS; s++) mt1_slot_init(sc.meta[s], sc.next_lineage++); return; }
    uint32_t hdr[4] = {0,0,0,0};
    bool ok = fread(hdr, sizeof(uint32_t), 4, f) == 4
              && hdr[0] == MT1_META_MAGIC && hdr[1] == MT1_META_VERSION
              && hdr[2] == (uint32_t)MT1_POOL_SLOTS && hdr[3] == (uint32_t)MT1_SCORE_HIST;
    if (ok) ok = fread(sc.meta, sizeof(MT1SlotMeta), MT1_POOL_SLOTS, f) == (size_t)MT1_POOL_SLOTS;
    if (ok) {
        int32_t ints[5];
        if (fread(ints, sizeof(int32_t), 5, f) == 5) {
            sc.best_slot = ints[0]; sc.actual_head = ints[1]; sc.actual_count = ints[2];
            sc.last_pushed_day = ints[3]; sc.next_lineage = (uint32_t)ints[4];
        }
        fread(sc.actual_buf, sizeof(float), MT1_BASELINE_DAYS, f);
    } else {
        log_msg(std::string(tag) + " " + g_ind_names[ind_i] + ": metadata sidecar rejected, registers reset");
        for (int s = 0; s < MT1_POOL_SLOTS; s++) mt1_slot_init(sc.meta[s], sc.next_lineage++);
    }
    fclose(f);
    if (sc.best_slot < 0 || sc.best_slot >= MT1_POOL_SLOTS) sc.best_slot = 0;
    // A prediction parked before the process died refers to an outcome we can no longer align.
    sc.has_pending = false;
}

static void save_mt2_elites(const std::string& dir, MT2Scratch& scratch) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        char p[512]; snprintf(p, sizeof(p), "%s/mt2_elite_%d.bin", dir.c_str(), slot);
        if (!save_bin(p, scratch.elite_buf + (size_t)slot * MT2NN_PARAMS, MT2NN_PARAMS))
            log_msg(std::string("WARNING: could not save ") + p);
    }
    char hp[512]; snprintf(hp, sizeof(hp), "%s/mt2_hist.bin", dir.c_str());
    FILE* hf = fopen(hp, "wb");
    if (hf) {
        int meta[2] = {scratch.hist_head, scratch.hist_count};
        fwrite(meta, sizeof(int), 2, hf);
        fwrite(scratch.hist_buf, sizeof(float),
               (size_t)HIST_DAYS * HIST_PER_DAY * MT2NN_PARAMS, hf);
        fclose(hf);
    }
}

static void load_or_init_mt2(const std::string& dir, const std::string& load_dir,
                               MT2Scratch& scratch) {
    PCG32 rng; rng.seed(mix_seed(0xCAFED00DBEEF1234ULL));
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        float* e = scratch.elite_buf + (size_t)slot * MT2NN_PARAMS;
        bool loaded = false;
        if (!load_dir.empty()) {
            char p[512]; snprintf(p, sizeof(p), "%s/mt2_elite_%d.bin", load_dir.c_str(), slot);
            loaded = load_bin(p, e, MT2NN_PARAMS);
        }
        if (!loaded) {
            char p[512]; snprintf(p, sizeof(p), "%s/mt2_elite_%d.bin", dir.c_str(), slot);
            loaded = load_bin(p, e, MT2NN_PARAMS);
        }
        if (!loaded) init_mt2_weights(e, rng);
    }
    // Load MT2 history
    {
        char hp[512];
        const std::string& hdir = load_dir.empty() ? dir : load_dir;
        snprintf(hp, sizeof(hp), "%s/mt2_hist.bin", hdir.c_str());
        FILE* hf = fopen(hp, "rb");
        if (!hf && !load_dir.empty()) {
            snprintf(hp, sizeof(hp), "%s/mt2_hist.bin", dir.c_str());
            hf = fopen(hp, "rb");
        }
        if (hf) {
            int meta[2] = {};
            if (fread(meta, sizeof(int), 2, hf) == 2) {
                scratch.hist_head  = std::max(0, std::min(meta[0], HIST_DAYS - 1));
                scratch.hist_count = std::max(0, std::min(meta[1], HIST_DAYS));
                fread(scratch.hist_buf, sizeof(float),
                      (size_t)HIST_DAYS * HIST_PER_DAY * MT2NN_PARAMS, hf);
            }
            fclose(hf);
        }
    }
}


// ── MT binary log ────────────────────────────────────────────────────────────────

static constexpr uint32_t MT_LOG_MAGIC   = 0x4D543132u;  // 'MT12'
// v8 (record V9): records are written PER BLOCK-DAY, not once per 25-day block, and carry the
// per-industry target mt1_actual_d[] so any run can be re-graded offline under any scheme
// (deployed hit rate, calibration, prediction-vs-target correlation). None of that was possible
// from a v7 log: 50 points per pass and the target recorded nowhere at all.
static constexpr uint32_t MT_LOG_VERSION = 12u;   // V12: single-output MT1

static bool write_mt_log_header(FILE* f) {
    uint32_t hdr[4] = {MT_LOG_MAGIC, MT_LOG_VERSION, (uint32_t)N_IND, 0u};
    return fwrite(hdr, sizeof(uint32_t), 4, f) == 4;
}

struct MTLogRecord {
    uint32_t pass_num, actual_day;

    // ── MT1 (V12): one pool, one output, one record per DAY ──────────────────────
    // The five-pool era's 20 score columns and the four-channel activation/OOS/skill blocks are
    // gone with the pools that produced them. What replaces them is smaller and says more: the
    // prediction, the outcome, the baseline it had to beat, and the pool's score distribution.
    float    mt1_pred[N_IND];        // deployed model's call for the NEXT session, in dollars
    float    mt1_actual_d[N_IND];    // realised P&L scored on this day (yesterday's call)
    float    mt1_baseline[N_IND];    // trailing mean of actual — the predictor MT1 must beat
    float    mt1_floor[N_IND];       // score-denominator floor in force for this day
    // Score distribution over the 200 individuals. Read the MEAN and slot-0; mt1_score_best is
    // max-of-200 and rises with pool size under a null, which is the artifact that made the old
    // in-sample numbers look like skill.
    float    mt1_score0[N_IND], mt1_score_mean[N_IND];
    float    mt1_score_best[N_IND], mt1_score_min[N_IND];
    // Lifecycle: {mature, culled_today, max_lineage_size, distinct_lineages, mean_retirement_age}
    float    mt1_pool_stats[N_IND][5];
    // Cumulative retirement-age histogram, buckets 8-15 / 16-31 / 32-63 / 64-127 / 128+.
    float    mt1_life[N_IND][5];

    // ── MT2 ───────────────────────────────────────────────────────────────────────
    float    mt2_best_pts, mt2_slot0_pts, mt2_ideal_pts;
    uint8_t  mt2_injected;
    uint8_t  pad[3];
    float    mt2_consensus_flat_pts;    // pool-consensus allocation score (flat vote)
    float    mt2_consensus_wtd_pts;     // pool-consensus allocation score (look-behind weighted)
    float    mt2_slot0_pts_pf;          // deployed slot0 graded on slot-0 portfolio delta
    float    mt2_slot0_pts_mkt;         // deployed slot0 graded on market forward return
};
static_assert(sizeof(MTLogRecord) == 904,
              "MTLogRecord must be 904 bytes — read_mt_log.py and plot_training.py parse by size");

// Build MT1CNet's 146-wide input for one industry: everything available at TODAY's close and
// nothing before it. Mirrors the CN_* field offsets in mt1_pool.h — a drift here shifts every
// symbol's block and still produces a number, so the offsets are named, not literal.
//
// Volume is deliberately absent: with no history there is no norm to read a raw share count
// against. Holdings and cash are the PRE-trade reference book, because today's orders do not fill
// until tomorrow — what is held at today's close is what we had this morning.
// Sanitise and bound. StockNN's raw head is ReLU on the quantities and saturating on the price
// fractions, so out48 carries values up to ~1e10 and is NON-FINITE for a large share of
// symbol-days. The fill path hides both: min(buy_qty, affordable) clamps the size, and
// `buy_qty > 1e-6f && buy_price > 0.f` silently drops NaN because every comparison against it is
// false. As a FEATURE neither is acceptable — one non-finite input poisons the whole forward pass,
// and a 1e10 input swamps every other column.
static inline float cn_ok(float v, float fallback = 0.f) {
    return std::isfinite(v) ? v : fallback;
}
static inline float cn_clamp(float v, float lo, float hi) {
    v = cn_ok(v);
    return v < lo ? lo : (v > hi ? hi : v);
}

// Which model's order intent goes into the feature vector. The default (nullptr) is slot 0 --
// the deployed model -- which is what the dataset log and the race competitors want. The gradient
// MT1 passes a specific slot's arrays so that each training row describes the orders that actually
// produced its target.
static void build_mt1c_input_for(const OHLCV* day_sym, const IndResult& ir,
                                 const float* in_bqty, const float* in_bfrac,
                                 const float* in_sfrac, const float* in_sqty,
                                 float* out /*MT1C_IN*/) {
    const float* si_bqty  = in_bqty  ? in_bqty  : ir.si_bqty;
    const float* si_bfrac = in_bfrac ? in_bfrac : ir.si_bfrac;
    const float* si_sfrac = in_sfrac ? in_sfrac : ir.si_sfrac;
    const float* si_sqty  = in_sqty  ? in_sqty  : ir.si_sqty;
    // Everything here is SCALE-FREE, and that is not cosmetic. MT1Net's own 74 inputs are already
    // normalised (returns lifted by RETURN_SCALE = 100, every polynomial level-normalised), so raw
    // dollars here would make the comparison measure normalisation rather than feature set.
    //
    // It also has to survive gradient-free search. The first layer computes w.x, so one mutation
    // of sigma moves the pre-activation by sigma * x: a $304 price and a +-0.4 close_pos differ by
    // ~750x in how hard the same mutation hits them, and selection would see only the price
    // columns. StockNN can feed raw dollars because it has 928,825 parameters and a dedicated
    // 422->300 layer to learn the rescaling; this has 11,177 in total.
    //
    // The OUTPUT is unaffected — both nets emit tanh(out) * MT1_PRED_SCALE and are scored against
    // dollar P&L. Input scale and output scale are independent.
    float mean_close = 0.f;
    int   n_close = 0;
    for (int j = 0; j < MT1C_SYMS; j++) {
        if (day_sym[j].valid && day_sym[j].close > 1e-6f) { mean_close += day_sym[j].close; n_close++; }
    }
    mean_close = n_close ? mean_close / (float)n_close : 1.f;
    const float book = (ir.book_prev > 1.f) ? ir.book_prev : 1.f;

    for (int j = 0; j < MT1C_SYMS; j++) {
        float* f = out + j * MT1C_PER_SYM;
        const OHLCV& b = day_sym[j];
        const bool ok = b.valid && b.high > b.low && b.close > 1e-6f;
        // Bar shape as ratios to today's close, and the close itself relative to the industry's
        // own level today — cross-sectional, so still present-day only.
        f[CN_OPEN]  = ok ? b.open / b.close : 1.f;
        f[CN_HIGH]  = ok ? b.high / b.close : 1.f;
        f[CN_LOW]   = ok ? b.low  / b.close : 1.f;
        f[CN_CLOSE] = ok ? b.close / mean_close : 1.f;
        f[CN_CPOS]  = ok ? stock_close_pos(b.open, b.high, b.low, b.close) : 0.f;
        f[CN_CWAP]  = ok ? stock_close_vs_wap(b.open, b.high, b.low, b.close) : 0.f;
        // (H-L)/A — the range as a fraction of the weighted average price. close_pos and
        // close_vs_wap share a numerator and differ only in denominator, so their RATIO is
        // exactly this; it is supplied directly rather than left for the net to divide.
        const float A = ok ? (2.f * b.open + 3.f * b.close + b.high + b.low) / 7.f : 0.f;
        f[CN_RANGE_A] = (ok && A > 1e-6f) ? (b.high - b.low) / A : 0.f;
        // Position WEIGHT, not a share count: holdings x close is the exposure that actually
        // determines the P&L being predicted, and 875 shares means nothing without the price.
        f[CN_HOLD]  = ok ? cn_clamp(cn_ok(ir.ref_hold[j]) * b.close / book, 0.f, 1.f) : 0.f;
        // whole_shares() on both, because that is what would actually execute — Alpaca stop
        // orders forbid fractional quantities, so the fill path floors every amount and a
        // fractional intent would describe a trade that cannot be placed. It matters most where
        // the floor bites: a small cash balance against a high price rounds to zero shares, which
        // is real information about whether the model can act at all.
        const float affordable = ok ? whole_shares(cn_ok(ir.ref_cash_v) / b.close) : 0.f;
        f[CN_BQTY]  = (affordable > 1e-9f)
                      ? cn_clamp(whole_shares(fminf(cn_ok(si_bqty[j]), affordable)) / affordable,
                                 0.f, 1.f)
                      : 0.f;
        const float pos = whole_shares(cn_ok(ir.ref_hold[j]));
        f[CN_SQTY]  = (pos > 1e-9f)
                      ? cn_clamp(whole_shares(fminf(cn_ok(si_sqty[j]), pos)) / pos, 0.f, 1.f)
                      : 0.f;
        // Price fractions are meant to index into today's range; clamp to it. 0.5 is the neutral
        // fallback (mid-range) so a non-finite output reads as "no opinion" rather than "at the low".
        f[CN_BFRAC] = cn_clamp(cn_ok(si_bfrac[j], 0.5f), 0.f, 1.f);
        f[CN_SFRAC] = cn_clamp(cn_ok(si_sfrac[j], 0.5f), 0.f, 1.f);
    }
    out[CN_CASH] = cn_clamp(cn_ok(ir.ref_cash_v) / book, 0.f, 1.f);
    out[CN_BOOK] = cn_ok(ir.book_prev) / (float)IND_STARTING_CASH;
}

// Slot 0's intent -- the deployed model. Every existing caller wants this.
static void build_mt1c_input(const OHLCV* day_sym, const IndResult& ir, float* out /*MT1C_IN*/) {
    build_mt1c_input_for(day_sym, ir, nullptr, nullptr, nullptr, nullptr, out);
}

// ── MT1 dataset log ──────────────────────────────────────────────────────────────
//
// Everything an offline fit needs, and nothing that depends on a choice we have not made yet:
// the 74 features MT1 sees per industry, and the day's outcome DECOMPOSED. Forward-h book P&L
// for ANY h is a sum over consecutive records, so the horizon becomes an offline sweep rather
// than one training run per candidate.
//
// The components are logged separately on purpose. book P&L = market move + trade delta, and the
// two behave nothing alike -- the trade delta is ~8% of the book's gain, zero-mean and heavy
// tailed, while the market move carries the level. Logging only the total would average them and
// we would rediscover that months later, which is the same mistake as averaging over the learning
// curve. The identity is also a free self-check: book_now - book_prev must equal mkt_move +
// trade_delta to floating-point tolerance on every row.
static constexpr uint32_t DS_LOG_MAGIC   = 0x4D543144u;   // "MT1D"
static constexpr uint32_t DS_LOG_VERSION = 4u;   // v4: + MT2INet, three pools paired
static constexpr int      DS_FEAT        = 74;            // per industry, = MT1Net's input width

struct DSLogRecord {
    uint32_t pass_num, actual_day;
    float    feat[N_IND][DS_FEAT];     // exactly the slice handed to mt1_step_day
    float    book_prev[N_IND];         // book at TODAY's close, before the day's trades
    float    book_now[N_IND];          // book at the NEXT close, after them
    float    mkt_move[N_IND];          // baseline - book_prev : prices moved, holdings fixed
    float    trade_delta[N_IND];       // slot0_score - baseline : holdings moved, prices fixed
    // ── v2: slot-0 order INTENT — see the IndResult comment for why this is causal and
    //    buy_exec/sell_exec are not. These carry per-symbol information that the 74 features,
    //    being built from two scalar value curves, provably cannot.
    float    oi_n_buy[N_IND];          // symbols carrying a buy order (0..IND_SYMS)
    float    oi_n_sell[N_IND];
    float    oi_buy_aggr[N_IND];       // mean (buy limit - close_t)/span_t; >0 = wants to fill now
    float    oi_sell_aggr[N_IND];
    float    oi_buy_disp[N_IND];       // sd of buy_price_frac across ordering symbols
    float    oi_buy_val[N_IND];        // intended buy, dollars at today's close
    float    oi_sell_val[N_IND];
    // ── v3: the competitor's whole input, so the two FEATURE SETS can be compared offline under
    //    one walk-forward protocol with the same controls, without another run.
    float    cfeat[N_IND][MT1C_IN];
    // Both pools' deployed call and its score, on the same day against the same target, so the
    // comparison is PAIRED in one place rather than joined across two files.
    float    m_pred[N_IND],  m_score[N_IND];    // MT1Net   (74  own industry, trailing)
    float    c_pred[N_IND],  c_score[N_IND];    // MT1CNet  (146 own industry, today + intent)
    float    i_pred[N_IND],  i_score[N_IND];    // MT2INet  (888 all industries, trailing)
};
static_assert(sizeof(DSLogRecord) == 8 + 4 * (DS_FEAT * N_IND + 11 * N_IND + MT1C_IN * N_IND
                                              + 6 * N_IND),
              "DSLogRecord packing drifted — the offline reader parses this by size");

static bool write_ds_log_header(FILE* f) {
    uint32_t h[4] = {DS_LOG_MAGIC, DS_LOG_VERSION, (uint32_t)N_IND, (uint32_t)DS_FEAT};
    return fwrite(h, sizeof(uint32_t), 4, f) == 4;
}

static void write_ds_log_record(FILE* f, const DSLogRecord& r) {
    fwrite(&r, sizeof(DSLogRecord), 1, f);
    fflush(f);
}

static void write_mt_log_record(FILE* f, const MTLogRecord& r) {
    fwrite(&r, sizeof(MTLogRecord), 1, f);
    fflush(f);
}

// ── Stock data loading ──────────────────────────────────────────────────────────

static std::vector<DayData> load_all_stock_data(const std::string& data_dir,
                                                  std::vector<std::string>& out_dates) {
    // Collect per-symbol maps
    std::map<std::string, OHLCV> sym_maps[N_IND][IND_SYMS];
    int loaded = 0;
    for (int i = 0; i < N_IND; i++) {
        for (int j = 0; j < IND_SYMS; j++) {
            std::string path = data_dir + "/" + g_syms[i][j] + ".json";
            if (load_sym_data(path, sym_maps[i][j])) loaded++;
        }
    }
    log_msg("Loaded local data for " + std::to_string(loaded) + "/" +
            std::to_string(N_SYMS) + " symbols");
    if (g_bad_bars > 0)
        log_msg("  WARNING: rejected " + std::to_string(g_bad_bars) +
                " non-finite/non-positive OHLC bar(s) — those sessions are skipped, not valued at 0");

    // Merge all dates
    std::map<std::string, int> date_index;
    for (int i = 0; i < N_IND; i++)
        for (int j = 0; j < IND_SYMS; j++)
            for (auto& [date, _] : sym_maps[i][j])
                date_index[date] = 0;

    int idx = 0;
    for (auto& [date, _] : date_index) { out_dates.push_back(date); date_index[date] = idx++; }

    std::vector<DayData> all_days(out_dates.size());
    for (int i = 0; i < N_IND; i++) {
        for (int j = 0; j < IND_SYMS; j++) {
            for (auto& [date, ohlcv] : sym_maps[i][j]) {
                all_days[date_index[date]].sym[i][j] = ohlcv;
            }
        }
    }
    return all_days;
}

// ── Pre-load history warmup ─────────────────────────────────────────────────────

static void warmup_history(IndustryState* ind_states,
                            const std::vector<DayData>& all_days, int day_start) {
    if (day_start <= 0) return;
    int start = std::max(0, day_start - HIST_WINDOW);
    for (int d = start; d < day_start; d++)
        for (int i = 0; i < N_IND; i++)
            for (int j = 0; j < IND_SYMS; j++)
                update_hist_sym(ind_states[i].hist[j], all_days[d].sym[i][j]);
    log_msg("Warmup: pre-loaded " + std::to_string(day_start - start) + " days of history");
}

// ── CSV logging ────────────────────────────────────────────────────────────────

static void write_csv_row(FILE* csv, int pass_num, int actual_day,
                           const IndResult* res, const MasterResult& mst,
                           const float mkt_ret[N_IND], const float mkt_val[N_IND]) {
    fprintf(csv, "%d,%d", pass_num + 1, actual_day + 1);
    for (int i = 0; i < N_IND; i++)
        fprintf(csv, ",%+10.2f,%+10.2f,%+10.2f,%.4f",
                res[i].elite_max_val, res[i].elite_min_val, res[i].elite_mean_val,
                res[i].mut_success);
    if (g_hold_csv) {
        for (int i = 0; i < N_IND; i++) {
            fprintf(g_hold_csv, "%d,%d,%s,%.2f", pass_num + 1, actual_day + 1,
                    g_ind_names[i].c_str(), res[i].hold_cash);
            for (int j = 0; j < IND_SYMS; j++) fprintf(g_hold_csv, ",%.0f", res[i].hold[j]);
            fprintf(g_hold_csv, "\n");
        }
        fflush(g_hold_csv);   // the other CSVs sit at 0 bytes for the whole run; this one must not
    }
    if (g_ctl_csv)
        for (int i = 0; i < N_IND; i++)
            if (res[i].ctl_n > 0)
                fprintf(g_ctl_csv, "%d,%d,%s,%d,%.2f,%.2f,%.2f,%d\n",
                        pass_num + 1, actual_day + 1, g_ind_names[i].c_str(),
                        res[i].ctl_n, res[i].best_delta, res[i].ctl_best, res[i].ctl_mean,
                        res[i].ctl_best >= res[i].best_delta ? 1 : 0);
    fprintf(csv, ",%+.2f,%+.2f,%+.2f,%+.2f",
            mst.elite_max_pts, mst.elite_min_pts, mst.elite_mean_pts, mst.ideal_pts);
    fprintf(csv, ",%+.2f,%+.2f", mst.consensus_flat_pts, mst.consensus_wtd_pts);
    fprintf(csv, ",%+.2f,%+.2f", mst.slot0_pts_pf, mst.slot0_pts_mkt);
    for (int i = 0; i < N_IND; i++)
        fprintf(csv, ",%.6f,%.2f", mkt_ret[i], mkt_val[i]);
    fprintf(csv, "\n");
    fflush(csv);
}

// ── Threading ──────────────────────────────────────────────────────────────────

struct WorkerCtx {
    IndustryState*      ind_states;
    const DayData*      day_ptr;
    const DayData*      fill_ptr;
    IndResult*          results;
    const bool*         seq_flags;
    std::string         models_dir;
    std::string         load_dir;   // only used on day_num == 0
    int                 actual_day, total_avail, day_num, num_days;
    float               sigma;
    bool                freeze, master_only;
    std::atomic<int>    next_ind{N_IND};
    std::counting_semaphore<N_IND> work_ready{0};
    std::counting_semaphore<N_IND> work_done{0};
    std::atomic<bool>   shutdown{false};
};

static void worker_fn(WorkerCtx* ctx) {
    WorkerScratch scratch;  // ~162 MB heap, allocated once per worker thread
    while (true) {
        ctx->work_ready.acquire();
        if (ctx->shutdown.load(std::memory_order_relaxed)) {
            ctx->work_done.release();
            return;
        }
        // Only pass load_dir on the first day of each pass to avoid re-seeding from stale checkpoint
        std::string use_load = (ctx->day_num == 0) ? ctx->load_dir : std::string();
        while (true) {
            int i = ctx->next_ind.fetch_add(1, std::memory_order_relaxed);
            if (i >= N_IND) break;
            // In master_only mode, run industry with freeze=true so master gets real
            // performance data for ind_val_hist without mutating industry models.
            bool freeze = ctx->freeze || ctx->master_only;
            ctx->results[i] = step_industry(i, ctx->ind_states[i], scratch,
                                             ctx->models_dir, use_load,
                                             *ctx->day_ptr, ctx->fill_ptr,
                                             ctx->actual_day, ctx->total_avail,
                                             ctx->day_num, ctx->num_days,
                                             ctx->sigma, freeze, ctx->seq_flags);
        }
        ctx->work_done.release();
    }
}

// ── main ───────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --account ACCT [--start-day N] [--stop-day N]\n"
        "          [--passes N] [--sigma F] [--master-sigma F] [--sigma-decay F]\n"
        "          [--dir-sigma F] [--rng-sigma F] [--acc-sigma F] [--cfd-sigma F] [--mt2-sigma F]\n"
        "          [--workers N] [--master-only] [--no-save] [--no-nn-race]\n"
        "          [--seed N]   (default: clock, RE-SEEDED EVERY PASS; N derives passes from N)\n"
        "          [--trade-lock PATH | --no-trade-lock]  pause while production holds the lock\n"
        "          [--control-untrained]  NO-LEARNING CONTROL: re-randomise StockNN daily\n"
        "          [--control-random]  pass-1 paired control: score N random models/day\n"
        "       %s --output DIR [--load-dir DIR] ...  (diagnostic/override)\n"
        "       %s --drift-study --load-dir SEED --drift-scratch DIR  (phase-3 calibration)\n",
        prog, prog, prog);
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase-3 lag/daily drift study   (--drift-study)   — see PHASE3_DRIFT_STUDY.md
// ────────────────────────────────────────────────────────────────────────────────
// Offline calibration tool that fixes the look-ahead retrain cadence. For each weekly
// freeze point it captures an immutable 8-model BAND (composite MT1 + MT2 slots 0-7)
// as a yardstick, then rolls a daily-upkept TRACK forward over a 6-week belt and scores,
// each gradable day, how far the track has pulled away from the frozen band (±1 tanh
// metric, §4). Output = four rolling buckets (week 2/3/4/5 lag), read off drift_log.bin.
//
// NON-DESTRUCTIVE: seeds (read-only) from the fully-trained production models via
// --load-dir and writes ONLY to --drift-scratch; every base/track it generates is
// throwaway, the seed models are never modified. Sequential-per-base: exactly one base
// resident at a time (STOCKNN pools make co-residence impossible — §5). Run only on
// mature models, ~once per codebase increment.
//
// STATUS: compile-checked; NOT yet runtime-validated (study is not run until phase 3).
// ════════════════════════════════════════════════════════════════════════════════

static constexpr int      DRIFT_WEEK_LEN    = 5;                    // business week (sessions)
static constexpr int      DRIFT_BELT_WEEKS  = 6;                    // weeks 0..5
static constexpr int      DRIFT_LAG_WEEKS   = 2;                    // weeks 0-1 forward-lag, generate-only
static constexpr int      DRIFT_BAND_N      = 8;                    // frozen elites per band (slots 0-7)
static constexpr int      DRIFT_WARMUP_DAYS = 2 * DRIFT_WEEK_LEN;   // 10-session rolling-buffer fill before freeze
static constexpr int      DRIFT_BELT_DAYS   = DRIFT_BELT_WEEKS * DRIFT_WEEK_LEN; // 30
static constexpr uint32_t DRIFT_LOG_MAGIC   = 0x44524654u;         // "DRFT"
static constexpr uint32_t DRIFT_LOG_VERSION = 1u;

// ±1 asymptotic relative metric (§4): 0 = track ties band mean; tanh(1)≈0.76 at band
// best/worst; →±1 only for decisive (multi-spread) results. floor guards the tight-band
// (fresh-retrain) case where mx≈mean would blow the ratio up.
static inline float drift_rel_metric(float S, float mean, float mx, float mn, float floor) {
    float spread = (S >= mean) ? (mx - mean) : (mean - mn);
    spread = fmaxf(spread, floor);
    return tanhf((S - mean) / spread);
}

// Adaptive MT1 floors for a given day — mirrors the block in step_mt1 so band and track
// are scored under identical floors (rel stays internally consistent).
// One frozen MT1 model's score on one day, for the drift band. Same scoring function the live
// pool uses, so band and track are directly comparable.
static float drift_score_mt1(const float* W, const float* in74,
                             float actual, float baseline, float floor_v) {
    return mt1_score(actual, mt1_pred(mt1net_forward(W, in74, 0.f)), baseline, floor_v);
}

// Tier-classification pts for one MT2 model (mirrors the step_mt2 grading at lines ~2843).
static float drift_score_mt2_pts(const float* W, const float in12[N_IND], const float actual_perf[N_IND]) {
    int opt_tier[N_IND] = {0};
    {
        int pos_idx[N_IND]; int n_pos = 0;
        for (int i = 0; i < N_IND; i++) if (actual_perf[i] >= 0.f) pos_idx[n_pos++] = i;
        sort_index_prefix(pos_idx, n_pos, [&](int a, int b){ return actual_perf[a] < actual_perf[b]; });
        if (n_pos == 1) { opt_tier[pos_idx[0]] = 3; }
        else if (n_pos == 2) { opt_tier[pos_idx[0]] = 2; opt_tier[pos_idx[1]] = 3; }
        else if (n_pos > 0) {
            int base = n_pos/3, rem = n_pos%3, n1 = base+(rem>=1?1:0), n2 = base+(rem>=2?1:0);
            for (int r = 0; r < n_pos; r++) { int ind = pos_idx[r];
                opt_tier[ind] = (r < n1) ? 1 : (r < n1+n2) ? 2 : 3; }
        }
    }
    float out48[48]; mt2_forward(W, in12, out48);
    int tier[N_IND];
    for (int i = 0; i < N_IND; i++) {
        const float* lg = out48 + i*4;
        int b = 0;
        for (int k = 1; k < 4; k++) { if (lg[k] > lg[b]) b = k; }
        tier[i] = b;                       // once per industry, AFTER the argmax loop
    }
    float pts = 0.f;
    for (int i = 0; i < N_IND; i++) {
        int pred = tier[i], opt = opt_tier[i];
        if (opt == 0)            { if (pred > 0) pts += -2.f - 0.25f * pred; }
        else if (pred == 0)      { pts -= (float)opt; }
        else if (pred <= opt)    { pts += (float)pred; }
        else                     { pts += (float)opt - 0.25f * (float)(pred - opt); }
    }
    return pts;
}

struct DriftLogRecord {
    uint32_t freeze_day;            // base freeze actual_day (provenance)
    uint32_t calendar_day;          // actual_day of this scored day
    uint32_t bucket;                // 2..5 = age_week = N-week lag
    uint32_t pad0;
    float    mt1_rel[N_IND];        // per-industry composite ±1 rel (track vs 8-band)
    float    mt2_rel;               // MT2 ±1 rel (track vs 8-band)
    float    mt2_band_mean, mt2_track;  // diagnostics
    float    mt1_band_mean[N_IND];
    float    mt1_track[N_IND];
};
static_assert(sizeof(DriftLogRecord) == 16 + 4*N_IND + 12 + 8*N_IND, "DriftLogRecord packing");

static bool write_drift_log_header(FILE* f) {
    uint32_t h[4] = {DRIFT_LOG_MAGIC, DRIFT_LOG_VERSION, (uint32_t)N_IND, 0u};
    return fwrite(h, sizeof(uint32_t), 4, f) == 4;
}
static void write_drift_log_record(FILE* f, const DriftLogRecord& r) {
    fwrite(&r, sizeof(DriftLogRecord), 1, f); fflush(f);
}

// Advance the live track one pipeline day (full upkeep: StockNN×12 + MT1×12 + MT2),
// mirroring main()'s per-day body. Fills in12/actual_perf for band scoring by the caller.
// Returns false if today has no forward target (last MT1_FWD_DAYS days — not gradable).
static bool drift_advance_day(int run_day_num, int actual_day, int total_days,
                              std::vector<DayData>& all_days,
                              IndustryState* ind_states, MT1PoolScratch* mt1_scr,
                              MT2Scratch& mt2_scr, MasterState& mst, WorkerScratch& wscr,
                              const std::string& scratch_dir, const std::string& seed_dir,
                              float sigma, float mt2_sigma,
                              float out_in12[N_IND], float out_actual_perf[N_IND],
                              float out_in74[N_IND][74], float out_mt1_actual[N_IND]) {
    const DayData* day_ptr  = &all_days[actual_day];
    const DayData* fill_ptr = (actual_day + 1 < total_days) ? &all_days[actual_day + 1] : nullptr;
    const DayData* fwd_ptr  = (actual_day + MT1_FWD_DAYS < total_days) ? &all_days[actual_day + MT1_FWD_DAYS] : nullptr;
    const bool fwd_valid = (fwd_ptr != nullptr);

    bool seq_flags[N_SYMS];
    PCG32 seq_rng; seq_rng.seed(mix_seed((uint64_t)actual_day * 0xABCDEF01234567ULL));
    for (int si = 0; si < N_SYMS; si++) seq_flags[si] = (seq_rng.next() & 1);

    // StockNN ×12 (serial — study trades CPU for RAM, §5)
    for (int i = 0; i < N_IND; i++)
        step_industry(i, ind_states[i], wscr, scratch_dir, seed_dir, *day_ptr, fill_ptr,
                      actual_day, total_days, run_day_num, 1 << 30, sigma, false, seq_flags);

    for (int i = 0; i < N_IND; i++)
        for (int j = 0; j < IND_SYMS; j++)
            update_hist_sym(ind_states[i].hist[j], day_ptr->sym[i][j]);

    float mkt_ret[N_IND] = {};
    if (fill_ptr) for (int i = 0; i < N_IND; i++) {
        float sum = 0.f; int v = 0;
        for (int j = 0; j < IND_SYMS; j++) { const OHLCV& t = day_ptr->sym[i][j], &n = fill_ptr->sym[i][j];
            if (t.valid && n.valid && t.close > 1e-3f) { sum += n.close / t.close - 1.f; v++; } }
        mkt_ret[i] = v > 0 ? sum / v : 0.f;
    }
    float fwd_ret[N_IND] = {};
    if (fwd_valid) for (int i = 0; i < N_IND; i++) {
        float sum = 0.f; int v = 0;
        for (int j = 0; j < IND_SYMS; j++) { const OHLCV& t = day_ptr->sym[i][j], &f = fwd_ptr->sym[i][j];
            if (t.valid && f.valid && t.close > 1e-3f) { sum += f.close / t.close - 1.f; v++; } }
        fwd_ret[i] = v > 0 ? sum / v : 0.f;
    }
    for (int i = 0; i < N_IND; i++) out_actual_perf[i] = fwd_ret[i];

    float today888[888];
    build_master_features(mst.mkt_val_hist, mst.ind_val_hist, mst.ind_hist_count, today888);

    // MT1 target stand-in. The study walks day-by-day with no StockNN P&L plumbed through, so it
    // grades against the industry's next-session market return on a fully-invested book. The
    // drift track measures how fast a FROZEN model's score decays relative to a live one; any
    // consistent target serves that, and this one shares the live target's scale and sign.
    if (fill_ptr) for (int i = 0; i < N_IND; i++) {
        const float* in74 = today888 + i * 74;
        const float actual = mkt_ret[i] * (float)IND_STARTING_CASH;
        MT1DayResult dr = mt1_step_day(i, mt1_scr[i], in74, actual, true, actual_day, sigma);
        out_in12[i] = dr.pred0;
        memcpy(out_in74[i], in74, 74 * sizeof(float));
        out_mt1_actual[i] = actual;
    }

    if (fwd_valid && actual_day >= MASTER_START_DAY) {
        bool inj = false;
        step_mt2(mst, mt2_scr, out_in12, out_actual_perf, actual_day, total_days, mt2_sigma, &inj);
    }

    // Append rolling market index (mirrors main lines ~3973)
    for (int i = 0; i < N_IND; i++) {
        float prev = (mst.ind_hist_count > 0)
            ? (mst.ind_hist_count < IND_HIST_CAP ? mst.mkt_val_hist[i][mst.ind_hist_count-1]
                                                 : mst.mkt_val_hist[i][IND_HIST_CAP-1])
            : (float)IND_STARTING_CASH;
        float today_v = prev * (1.f + mkt_ret[i]);
        if (mst.ind_hist_count < IND_HIST_CAP) mst.mkt_val_hist[i][mst.ind_hist_count] = today_v;
        else { memmove(mst.mkt_val_hist[i], mst.mkt_val_hist[i]+1, (IND_HIST_CAP-1)*sizeof(float));
               mst.mkt_val_hist[i][IND_HIST_CAP-1] = today_v; }
    }
    if (mst.ind_hist_count < IND_HIST_CAP) mst.ind_hist_count++;
    return fwd_valid;
}

// Reset all per-base state to the seed and wipe the scratch dir so step_* re-seed from
// the read-only production models (load_dir) rather than the previous base's track.
static void drift_reseed(const std::string& scratch_dir, const std::string& seed_dir,
                         IndustryState* ind_states, MT1PoolScratch* mt1_scr, MT2Scratch& mt2_scr,
                         MasterState& mst, std::vector<DayData>& all_days, int run_start) {
    std::error_code ec;
    fs::remove_all(scratch_dir, ec);          // throw away previous base's evolved track
    fs::create_directories(scratch_dir);
    for (int i = 0; i < N_IND; i++) {
        ind_states[i].portfolios[0].cash = IND_STARTING_CASH;
        for (int j = 0; j < IND_SYMS; j++) { ind_states[i].portfolios[0].holdings[j] = 0.f;
                                             ind_states[i].portfolios[0].stop_prices[j] = 0.f; }
        ind_states[i].streak = 0;
        for (int s = 1; s < N_SLOTS; s++) ind_states[i].portfolios[s] = ind_states[i].portfolios[0];
        for (int j = 0; j < IND_SYMS; j++) ind_states[i].hist[j] = SymHist{};
        // Reset MT1 window state IN PLACE (MT1PoolScratch owns heap buffers — never reassign
        // it: copy-assign is deleted precisely because a shallow copy would double-free).
        MT1PoolScratch& sc = mt1_scr[i];
        memset(sc.actual_buf, 0, sizeof(sc.actual_buf));
        sc.actual_head = sc.actual_count = 0;
        sc.last_pushed_day = -1;
        sc.has_pending = false;
        sc.barred_count = 0;
        load_or_init_mt1_pool(scratch_dir, seed_dir, i, sc);
    }
    load_or_init_mt2(scratch_dir, seed_dir, mt2_scr);
    mt2_scr.hist_head = mt2_scr.hist_count = 0;
    mt2_scr.lb_head   = mt2_scr.lb_count   = 0;
    mst.portfolios[0].cash = MST_STARTING_CASH;
    for (int i = 0; i < N_IND; i++) mst.portfolios[0].holdings[i] = 0.f;
    for (int s = 1; s < N_SLOTS; s++) mst.portfolios[s] = mst.portfolios[0];
    memset(mst.ind_val_hist, 0, sizeof(mst.ind_val_hist));
    memset(mst.mkt_val_hist, 0, sizeof(mst.mkt_val_hist));
    mst.ind_hist_count = 0;
    memset(mst.zero_counts, 0, sizeof(mst.zero_counts));
    warmup_history(ind_states, all_days, run_start);
}

static int run_drift_study(const std::string& seed_dir, const std::string& scratch_dir,
                           const std::string& log_dir, std::vector<DayData>& all_days,
                           int total_days, int win_start, int win_end,
                           float sigma, float mt2_sigma) {
    if (seed_dir.empty() || scratch_dir.empty()) {
        log_msg("ERROR: --drift-study needs --load-dir SEED and --drift-scratch DIR."); return 1;
    }
    // Earliest freeze: needs DRIFT_WARMUP_DAYS of rolling fill before freeze AND the freeze
    // itself must sit past MT1_START_DAY so the seeded pools are active.
    int first_freeze = std::max(win_start, std::max(DRIFT_WARMUP_DAYS, MT1_START_DAY) + DRIFT_WARMUP_DAYS);
    // Latest freeze: the last gradable belt day (freeze + DRIFT_BELT_DAYS) still needs a forward target.
    int last_freeze  = win_end - DRIFT_BELT_DAYS - MT1_FWD_DAYS;
    if (first_freeze > last_freeze) {
        log_msg("ERROR: window too short for the drift belt — need >= "
                + std::to_string(DRIFT_WARMUP_DAYS + DRIFT_BELT_DAYS + MT1_FWD_DAYS) + " sessions of runway.");
        return 1;
    }

    fs::create_directories(log_dir);
    std::string drift_path = log_dir + "/drift_log.bin";
    FILE* dlog = fopen(drift_path.c_str(), "wb");
    if (!dlog) { log_msg("ERROR: cannot open " + drift_path); return 1; }
    write_drift_log_header(dlog);

    log_msg("===== DRIFT STUDY ===== seed=" + seed_dir + "  scratch=" + scratch_dir +
            "  freeze " + std::to_string(first_freeze) + ".." + std::to_string(last_freeze) +
            " step " + std::to_string(DRIFT_WEEK_LEN));

    // One base resident (§5)
    auto ind_states = std::make_unique<IndustryState[]>(N_IND);
    auto mt1_scr    = std::make_unique<MT1PoolScratch[]>(N_IND);
    auto mt2_scr    = std::make_unique<MT2Scratch>();
    auto mst        = std::make_unique<MasterState>();
    auto wscr       = std::make_unique<WorkerScratch>();
    // Band = the pool's top 8 individuals per industry, frozen at the freeze day.
    // Track = whichever individual the live pool has ranked first today.
    std::vector<float> mt1_band((size_t)N_IND * DRIFT_BAND_N * MT1NET_PARAMS);
    std::vector<float> mt2_band((size_t)DRIFT_BAND_N * MT2NN_PARAMS);

    for (int freeze_day = first_freeze; freeze_day <= last_freeze; freeze_day += DRIFT_WEEK_LEN) {
        int run_start = freeze_day - DRIFT_WARMUP_DAYS;
        drift_reseed(scratch_dir, seed_dir, ind_states.get(), mt1_scr.get(), *mt2_scr,
                     *mst, all_days, run_start);

        int run_day_num = 0;
        float in12[N_IND], actual_perf[N_IND], mt1_in74[N_IND][74], mt1_actual[N_IND];
        // Warm-up: fill rolling buffers up to (and including) the freeze day.
        for (int d = run_start; d <= freeze_day; d++)
            drift_advance_day(run_day_num++, d, total_days, all_days, ind_states.get(),
                              mt1_scr.get(), *mt2_scr, *mst, *wscr, scratch_dir, seed_dir,
                              sigma, mt2_sigma, in12, actual_perf, mt1_in74, mt1_actual);

        // Freeze the band: MT1 = the pool's top DRIFT_BAND_N individuals; MT2 = elite slots 0-7.
        for (int i = 0; i < N_IND; i++) {
            MT1PoolScratch& sc = mt1_scr[i];
            int order[MT1_POOL_SLOTS]; mt1_rank(sc, order);
            for (int b = 0; b < DRIFT_BAND_N; b++)
                memcpy(&mt1_band[((size_t)i * DRIFT_BAND_N + b) * MT1NET_PARAMS],
                       sc.slot(order[b]), MT1NET_PARAMS * sizeof(float));
        }
        for (int b = 0; b < DRIFT_BAND_N; b++)
            memcpy(&mt2_band[(size_t)b * MT2NN_PARAMS], mt2_scr->elite(b), MT2NN_PARAMS * sizeof(float));

        // Belt: roll the track forward DRIFT_BELT_DAYS, scoring weeks 2-5.
        for (int age_day = 0; age_day < DRIFT_BELT_DAYS; age_day++) {
            int actual_day = freeze_day + 1 + age_day;
            bool gradable = drift_advance_day(run_day_num++, actual_day, total_days, all_days,
                                              ind_states.get(), mt1_scr.get(), *mt2_scr, *mst, *wscr,
                                              scratch_dir, seed_dir, sigma, mt2_sigma, in12, actual_perf, mt1_in74, mt1_actual);
            int age_week = age_day / DRIFT_WEEK_LEN;          // 0..5
            if (age_week < DRIFT_LAG_WEEKS || !gradable) continue;

            DriftLogRecord rec{};
            rec.freeze_day   = (uint32_t)freeze_day;
            rec.calendar_day = (uint32_t)actual_day;
            rec.bucket       = (uint32_t)age_week;

            // MT1: today's live best vs the 8 frozen individuals, on the same day, same target
            // and the same baseline/floor. The metric is (track - band_mean)/spread, so the
            // shared baseline cancels and only the relative decay is read.
            for (int i = 0; i < N_IND; i++) {
                MT1PoolScratch& sc = mt1_scr[i];
                const float base = sc.baseline(), fl = sc.floor_v();
                const float act  = mt1_actual[i];
                float bmean = 0.f, bmax = -1e30f, bmin = 1e30f;
                for (int b = 0; b < DRIFT_BAND_N; b++) {
                    float v = drift_score_mt1(&mt1_band[((size_t)i*DRIFT_BAND_N + b)*MT1NET_PARAMS],
                                              mt1_in74[i], act, base, fl);
                    bmean += v; bmax = fmaxf(bmax, v); bmin = fminf(bmin, v);
                }
                bmean /= DRIFT_BAND_N;
                float track = drift_score_mt1(sc.slot(sc.best_slot), mt1_in74[i], act, base, fl);
                float floor = fmaxf(1e-3f, 0.05f * fabsf(bmean));   // band-width adaptive floor (§4)
                rec.mt1_rel[i]       = drift_rel_metric(track, bmean, bmax, bmin, floor);
                rec.mt1_band_mean[i] = bmean;
                rec.mt1_track[i]     = track;
            }

            // MT2: track (slot0) vs 8-model band, same in12/actual_perf.
            {
                float bmean = 0.f, bmax = -1e30f, bmin = 1e30f;
                for (int b = 0; b < DRIFT_BAND_N; b++) {
                    float s = drift_score_mt2_pts(&mt2_band[(size_t)b*MT2NN_PARAMS], in12, actual_perf);
                    bmean += s; bmax = fmaxf(bmax, s); bmin = fminf(bmin, s);
                }
                bmean /= DRIFT_BAND_N;
                float track = drift_score_mt2_pts(mt2_scr->elite(0), in12, actual_perf);
                float floor = fmaxf(1e-3f, 0.05f * fabsf(bmean));
                rec.mt2_rel       = drift_rel_metric(track, bmean, bmax, bmin, floor);
                rec.mt2_band_mean = bmean;
                rec.mt2_track     = track;
            }
            write_drift_log_record(dlog, rec);
        }
        log_msg("[drift] base freeze_day=" + std::to_string(freeze_day) + " done");
    }

    fclose(dlog);
    std::error_code ec; fs::remove_all(scratch_dir, ec);   // leave no throwaway models behind
    log_msg("Drift study complete → " + drift_path);
    return 0;
}

int main(int argc, char* argv[]) {
    // Parse args
    std::string output_dir, load_dir, account;
    int  start_day = -1, stop_day = -1, passes = 1, num_workers = 2;
    float sigma = 0.01f, master_sigma = -1.f, sigma_decay = 0.5f;
    float mt1_sigma = -1.f, mt2_sigma_arg = -1.f;
    bool master_only = false;
    bool drift_study = false; std::string drift_scratch;

    for (int a = 1; a < argc; a++) {
        std::string arg(argv[a]);
        if      (arg == "--account"  && a+1<argc) { account    = argv[++a]; }
        else if (arg == "--output"   && a+1<argc) { output_dir = argv[++a]; }
        else if (arg == "--load-dir" && a+1<argc) { load_dir   = argv[++a]; }
        else if (arg == "--start-day"&& a+1<argc) { start_day= atoi(argv[++a]); }
        else if (arg == "--stop-day" && a+1<argc) { stop_day = atoi(argv[++a]); }
        else if (arg == "--passes"   && a+1<argc) { passes   = atoi(argv[++a]); }
        else if (arg == "--sigma"    && a+1<argc) { sigma    = atof(argv[++a]); }
        else if (arg == "--master-sigma"&&a+1<argc){master_sigma=atof(argv[++a]);}
        else if (arg == "--sigma-decay"&&a+1<argc){sigma_decay=atof(argv[++a]);}
        else if (arg == "--mt1-sigma" && a+1<argc) { mt1_sigma    = atof(argv[++a]); }
        else if (arg == "--mt2-sigma" && a+1<argc) { mt2_sigma_arg= atof(argv[++a]); }
        else if (arg == "--workers"  && a+1<argc) { num_workers=atoi(argv[++a]);}
        else if (arg == "--master-only") master_only = true;
        else if (arg == "--no-nn-race") g_no_nn_race = true;
        else if (arg == "--no-save") g_no_save = true;
        else if (arg == "--control-untrained") g_control_untrained = true;
        else if (arg == "--control-random") g_control_random = true;
        else if (arg == "--seed"     && a+1<argc) { g_seed_arg = strtoull(argv[++a], nullptr, 10); }
        else if (arg == "--trade-lock" && a+1<argc) { g_trade_lock = argv[++a]; }
        else if (arg == "--no-trade-lock") { g_trade_lock.clear(); }
        else if (arg == "--dir-reps" && a+1<argc) { g_dir_reps = std::max(1, atoi(argv[++a])); }
        else if (arg == "--drift-study") drift_study = true;
        else if (arg == "--drift-scratch" && a+1<argc) { drift_scratch = argv[++a]; }
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }
    // --account derives model and log dirs; --output/--load-dir are diagnostic overrides
    std::string log_dir;
    if (!account.empty()) {
        std::string acct_dir = "models/" + account + "/training";
        if (output_dir.empty()) output_dir = acct_dir;
        if (load_dir.empty())   load_dir   = acct_dir;
        log_dir = "logs/" + account + "/training";
    } else {
        log_dir = output_dir;  // diagnostic: co-locate logs with models
    }
    if (output_dir.empty()) { print_usage(argv[0]); return 1; }
    if (master_sigma  < 0.f) master_sigma  = sigma;
    // Per-component defaults: dir/acc keep master_sigma; rng=2/3, cfd=1/2, mt2=1/3
    if (mt1_sigma     < 0.f) mt1_sigma     = master_sigma;
    if (mt2_sigma_arg < 0.f) mt2_sigma_arg = master_sigma / 6.f;  // v0.2.6.0: 0.002→0.001 (tighten MT2 pool)

    log_msg(std::string("training_v4_cpp v") + TRAINER_VERSION +
            "  account=" + (account.empty() ? "(diagnostic)" : account));
    log_msg(g_seed_arg ? "RNG: --seed " + std::to_string(g_seed_arg) + " (per-pass, derived)"
                       : std::string("RNG: clock-seeded per pass — runs will not repeat"));

    if (g_control_untrained) {
        log_msg("");
        log_msg("################################################################");
        log_msg("#  --control-untrained : THIS IS NOT A TRAINING RUN            #");
        log_msg("#  StockNN elites are re-randomised every day and never saved. #");
        log_msg("#  Any performance below is selection-on-noise, not learning.  #");
        log_msg("#  CSV goes to training_log_CONTROL.csv; do not compare it to  #");
        log_msg("#  a real run without saying which is which.                   #");
        log_msg("################################################################");
        log_msg("");
    }

    if (!load_universe_json("universe.json")) return 1;

    // --no-save: train into a scratch directory and drop it at exit. Writing is REQUIRED for
    // training to work at all — elite_buf and hist_buf are reloaded from disk every day — so the
    // flag redirects the writes instead of suppressing them. Real disk, never /tmp: a run
    // directory is ~3 GB and /tmp here is a 978 MB tmpfs.
    std::string nosave_scratch;
    if (g_no_save) {
        std::error_code ec;
        nosave_scratch = output_dir + ".nosave";
        fs::remove_all(nosave_scratch, ec);
        fs::create_directories(nosave_scratch, ec);
        if (ec) {
            fprintf(stderr, "FATAL: --no-save could not create scratch %s\n", nosave_scratch.c_str());
            return 1;
        }
        output_dir = nosave_scratch;
        log_msg("--no-save: training into scratch " + nosave_scratch +
                " (~3 GB, removed at exit); canonical models untouched");
    }

    // Disable OpenBLAS internal threading: N workers × M BLAS threads = N×M threads on N CPUs
    openblas_set_num_threads(1);

    fs::create_directories(output_dir);
    fs::create_directories(log_dir);
    fs::create_directories("stock_data");

    // Load stock data
    std::vector<std::string> dates;
    std::vector<DayData> all_days = load_all_stock_data("stock_data", dates);
    if (all_days.empty()) {
        log_msg("ERROR: No stock data found in stock_data/. Run download_5y_data.py first.");
        return 1;
    }
    int total_days = (int)all_days.size();
    int day_start  = (start_day >= 0) ? start_day : 0;
    int day_end    = (stop_day  >= 0) ? stop_day  : total_days;
    day_end = std::min(day_end, total_days);
    log_msg("Total trading days: " + std::to_string(total_days) +
            "  Training: days " + std::to_string(day_start) +
            "–" + std::to_string(day_end));

    // Phase-3 drift study: separate non-destructive driver (seeds from --load-dir, writes
    // only to --drift-scratch). Routes here before any training state is allocated.
    if (drift_study)
        return run_drift_study(load_dir, drift_scratch, log_dir, all_days, total_days,
                               day_start, day_end, sigma, mt2_sigma_arg);

    // Allocate state on heap
    auto ind_states   = std::make_unique<IndustryState[]>(N_IND);
    auto mst          = std::make_unique<MasterState>();   // portfolio state reused by MT2
    // Online conditional models: 5 memory kernels x depths {1,2}, pooled across industries,
    // trained during the pass from the scores already in hand. Kilobytes, not megabytes.
    auto bo_family = bo_make_family();
    // Gradient-trained MT1, one per industry. Re-initialised at the START OF EVERY PASS: the next
    // pass seeds StockNN from a fresh champion/challenger blend, which is a different network, and
    // an MT1 carried across would be predicting a model that no longer exists.
    std::vector<float> bp_pred(N_IND, 0.f);
    // One state per race entry. Evolutionary entries own a 200-slot pool per industry and reuse
    // the existing lifecycle (score registers, maturity, culling, lineage) via mt1_step_day;
    // gradient entries own one net per industry plus its Adam state. Measured on the droplet:
    // 127 MB for all eight across twelve industries, 0.39 s/day of work.
    struct RaceState {
        std::unique_ptr<MT1PoolScratch[]> pool;   // evolutionary
        std::vector<std::vector<float>>   w, m, v, g;   // gradient: weights + Adam moments + grad
        double loss_sum = 0.0; long loss_n = 0;
        double rank_sum = 0.0; long rank_n = 0;
        // Distinct PREDICTIONS per industry-day, not distinct inputs. Two different order sets can
        // still map to the same number, and it is prediction ties the rank function has to break.
        // MT1Net ties on all 200 by construction (no order intent in its inputs), which is how it
        // serves as the control; MT1S collapses 146 features into twelve per-symbol slices, so it
        // may tie more than MT1C even where the inputs differ. The residual bias from mishandled
        // ties scales as (k/n)^2 in the tie width, so this is the number that bounds it.
        double distinct_sum = 0.0; long distinct_n = 0;
        // ── allocation backtest ───────────────────────────────────────────────────────
        // The metric that decides paper readiness, which neither rank nor corr answers. A model
        // can correlate +0.05 with realised P&L and still allocate worse than 1/12, because
        // allocation depends on getting the TOP industries right rather than the average rank.
        //
        // Each day the 12 deployed predictions are ranked and capital placed equally across the
        // top k, against an equal 1/12 flat book. Long-only, because that is what production does.
        // The long-short spread is carried too: it is the cleaner read on whether the ranking
        // holds information, since it cancels the market move common to all 12.
        double alloc_flat = 0.0, alloc_top4 = 0.0, alloc_top6 = 0.0, alloc_bot4 = 0.0;
        long   alloc_n = 0;
        std::vector<double> bk_sum = std::vector<double>(16, 0.0);
        std::vector<long>   bk_n   = std::vector<long>(16, 0);
        // across-day: the deployed model's prediction against the realised industry P&L
        std::vector<double> xy, xx, yy, sx, sy;
        std::vector<long>   dn;
        long adam_step = 0;
    };
    // ── how many GENUINELY distinct training rows each architecture is handed ──────
    //
    // A row is (input, next-day bars) -> outcome. Within an industry-day the bars are common to
    // all 200 models, so the count is driven by the inputs; across days the bars differ, so no two
    // days duplicate each other even where the order sets repeat.
    //
    // Counted on INPUTS, not predictions. Two models can agree on a number while being different
    // rows, and the data available is a property of the inputs, not of what a particular net makes
    // of them. This also turns the MT1 control claim into a measurement: feat74 carries no order
    // intent, so it must come back at exactly 1.0 distinct rows per industry-day. If it does not,
    // the claim that MT1 cannot rank order sets is false and its row is not a control.
    //
    // Per-architecture rather than per-entry: six of the eight entries share one input set.
    struct RowCount { double sum = 0.0; long n = 0; };
    RowCount rows_feat74, rows_cfeat146, rows_persym;

    std::vector<RaceState> race(RACE_N);
    for (int v = 0; v < RACE_N; v++) {
        RaceState& r = race[v];
        const int np = race_params(RACE[v].arch);
        if (RACE[v].search == SRCH_EVO) {
            r.pool = std::make_unique<MT1PoolScratch[]>(N_IND);
            for (int i = 0; i < N_IND; i++) r.pool[i].alloc(np);
        } else {
            r.w.assign(N_IND, std::vector<float>(np, 0.f));
            r.m.assign(N_IND, std::vector<float>(np, 0.f));
            r.v.assign(N_IND, std::vector<float>(np, 0.f));
            r.g.assign(N_IND, std::vector<float>(np, 0.f));
        }
        r.xy.assign(N_IND, 0.0); r.xx.assign(N_IND, 0.0); r.yy.assign(N_IND, 0.0);
        r.sx.assign(N_IND, 0.0); r.sy.assign(N_IND, 0.0); r.dn.assign(N_IND, 0);
    }

    // Adam on a flat array, shared by every gradient entry so the optimiser is not a confound.
    auto race_adam = [](RaceState& r, int ind, long step) {
        const float lr = 3e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, decay = 1e-3f;
        const float bc1 = 1.f - powf(b1, (float)step), bc2 = 1.f - powf(b2, (float)step);
        std::vector<float>& W = r.w[ind]; std::vector<float>& M = r.m[ind];
        std::vector<float>& V = r.v[ind]; std::vector<float>& G = r.g[ind];
        for (size_t k = 0; k < W.size(); k++) {
            M[k] = b1 * M[k] + (1.f - b1) * G[k];
            V[k] = b2 * V[k] + (1.f - b2) * G[k] * G[k];
            W[k] -= lr * ((M[k] / bc1) / (sqrtf(V[k] / bc2) + eps) + decay * W[k]);
            G[k] = 0.f;
        }
    };

    // How well MT1 predicts the TRADING OUTCOME, measured two ways, both out-of-sample by
    // construction because every prediction is made before that day is trained on.
    //   within-day : Spearman across the MT1_BP_MODELS order sets, one value per industry-day.
    //                This is what per-model intent buys -- the market move is common to all of
    //                them, so ranking them IS ranking the trading.
    //   across-day : Pearson of slot 0's prediction against slot 0's realised P&L, per industry.
    //                This is what an allocator would consume.
    // ...and the same split into 100-day buckets. MT1 starts from random weights at
    // MT1_BP_START_DAY, so a single pass-average blends "has not learned yet" with "has", and a
    // flat number could not tell a net that never learns from one that learns slowly. The
    // trajectory across buckets is the part that says which.
    static constexpr int BP_BUCKET = 100;
    std::vector<double> bp_bk_sum(16, 0.0);
    std::vector<long>   bp_bk_n(16, 0);
    // Yesterday's conditioning comes from ALL of yesterday's scored elites, not only the
    // deployed slot. The chain is coherent because every slot resets to slot 0's portfolio each
    // morning: slot0(t-2) -> elite_e(t-1) -> slot0(t) is a genuine two-step trajectory for elite
    // e, even though slot INDICES do not persist across days (selection reorders them).
    float bo_prev2[N_IND] = {};                            // slot-0 P&L at t-2
    float bo_prev1_el[N_IND][ELITE_POOL] = {};             // every elite's P&L at t-1
    bool  bo_ok2[N_IND] = {}, bo_ok1[N_IND] = {};
    float bo_pnl_today[N_IND] = {};
    float bo_el_today[N_IND][ELITE_POOL] = {};
    bool  bo_have[N_IND] = {};
    FILE* bo_csv = nullptr;

    // The competitor: same pool, same lifecycle, same score, same target — only the network and
    // its inputs differ, so a difference in outcome is a difference in FEATURE SET.
    // The independent allocator: all 888 features, so it sees the cross-section the other two
    // cannot. ~278 MB of pool across 12 industries — the largest of the three by far, because
    // 888 inputs cannot be read by a small first layer.
    auto mt2_scratch  = std::make_unique<MT2Scratch>();            // ~5.5 MB

    // Open CSV log (goes to log_dir, not output_dir)
    // Deliberately NOT training_log.csv. Every plotting and analysis script in the repo reads
    // that exact name, and a control run's numbers look entirely ordinary — the only thing that
    // makes them safe is that nothing can load them by accident.
    std::string csv_path = log_dir +
        (g_control_untrained ? "/training_log_CONTROL.csv" : "/training_log.csv");
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (csv) {
        fprintf(csv, "pass,day");
        for (int i = 0; i < N_IND; i++)
            fprintf(csv, ",%s_elite_max,%s_elite_min,%s_elite_mean,%s_mut_success",
                    g_ind_names[i].c_str(), g_ind_names[i].c_str(),
                    g_ind_names[i].c_str(), g_ind_names[i].c_str());
        fprintf(csv, ",mt2_elite_max_pts,mt2_elite_min_pts,mt2_elite_mean_pts,mt2_ideal_pts");
        fprintf(csv, ",mt2_consensus_flat_pts,mt2_consensus_wtd_pts");
        fprintf(csv, ",mt2_slot0_pts_pf,mt2_slot0_pts_mkt");
        for (int i = 0; i < N_IND; i++)
            fprintf(csv, ",%s_mkt_ret,%s_mkt_val",
                    g_ind_names[i].c_str(), g_ind_names[i].c_str());
        fprintf(csv, "\n");
    }

    if (g_control_untrained) {
        FILE* mk = fopen((output_dir + "/CONTROL_UNTRAINED").c_str(), "w");
        if (mk) {
            fprintf(mk, "This directory is the output of a --control-untrained run (v%s).\n"
                        "StockNN weights here are NOT trained: the elites were re-randomised\n"
                        "every day. Do not seed a run from this directory and do not run\n"
                        "convert_weights.py against it. MT1/MT2 here were trained, but on top\n"
                        "of a random StockNN portfolio, so they are not usable either.\n",
                    TRAINER_VERSION);
            fclose(mk);
        }
    }

    {
        std::string hp = log_dir + "/holdings_log.csv";
        g_hold_csv = fopen(hp.c_str(), "w");
        if (g_hold_csv) {
            fprintf(g_hold_csv, "pass,day,industry,cash");
            for (int j = 0; j < IND_SYMS; j++) fprintf(g_hold_csv, ",q%d", j);
            fprintf(g_hold_csv, "\n");
            log_msg("slot-0 holdings log -> " + hp);
        } else {
            log_msg("WARNING: could not open " + hp);
        }
    }

    if (g_control_random) {
        std::string ctl_path = log_dir + "/control_log.csv";
        g_ctl_csv = fopen(ctl_path.c_str(), "w");
        if (g_ctl_csv) {
            fprintf(g_ctl_csv, "pass,day,industry,n_control,trained_best_delta,"
                               "control_best_delta,control_mean_delta,control_wins\n");
            log_msg("--control-random: pass-1 paired control active -> " + ctl_path);
        } else {
            log_msg("WARNING: could not open " + ctl_path + " — control stats will only be in the log");
        }
    }

    // Open binary MT log (goes to log_dir, not output_dir)
    std::string mt_log_path = log_dir + "/mt_training_log.bin";
    FILE* mt_log = fopen(mt_log_path.c_str(), "wb");
    if (mt_log) { write_mt_log_header(mt_log); fflush(mt_log); }
    // ~3.7 KB/day -> ~4.6 MB per pass. Always on: it is the only artefact that lets the horizon
    // and the "is there any signal in these 74 features" question be answered without a rerun.
    std::string ds_log_path = log_dir + "/mt1_dataset.bin";
    FILE* ds_log = fopen(ds_log_path.c_str(), "wb");
    if (ds_log) {
        write_ds_log_header(ds_log); fflush(ds_log);
        log_msg("MT1 dataset -> " + ds_log_path + " (" +
                std::to_string(sizeof(DSLogRecord)) + " B/day)");
    } else {
        log_msg("WARNING: could not open " + ds_log_path + " — offline horizon sweep unavailable");
    }

    // Threading setup
    num_workers = std::max(1, std::min(num_workers, N_IND));
    WorkerCtx wctx;
    wctx.ind_states  = ind_states.get();
    wctx.models_dir  = output_dir;
    wctx.load_dir    = load_dir;

    std::vector<std::thread> workers;
    for (int w = 0; w < num_workers; w++)
        workers.emplace_back(worker_fn, &wctx);

    log_msg("Started " + std::to_string(num_workers) + " worker threads");

    // ── Multi-pass loop ─────────────────────────────────────────────────────
    for (int pass = 0; pass < passes; pass++) {
        float decay         = powf(sigma_decay, (float)pass);
        float cur_sigma     = sigma        * decay;
        float cur_mst_sigma = master_sigma * decay;  // composite blends
        float cur_mt1_sigma = mt1_sigma    * decay;
        float cur_mt2_sigma = mt2_sigma_arg* decay;
        // Fresh entropy per pass. Without this the per-day seeds repeat every pass and each pass
        // replays the same perturbation vectors over different parents.
        g_cur_pass = pass;   // written before workers start; read-only inside them
        g_run_seed = g_seed_arg
            ? splitmix64(g_seed_arg + (uint64_t)(pass + 1) * 0x9E3779B97F4A7C15ULL)
            : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        log_msg("  pass seed: " + std::to_string(g_run_seed));

        log_msg("===== PASS " + std::to_string(pass+1) + "/" + std::to_string(passes) +
                " | sigma=" + std::to_string(cur_sigma).substr(0,6) +
                " | mst=" + std::to_string(cur_mst_sigma).substr(0,6) +
                " | mt1=" + std::to_string(cur_mt1_sigma).substr(0,6) +
                " | mt2=" + std::to_string(cur_mt2_sigma).substr(0,6) + " =====");

        // Slot-0 portfolio value at the two ends of the pass-boundary judging window. Captured
        // from results[i].baseline, which is what the log prints as prod=$ — the same quantity the
        // offline study measured. Zeroed each pass so a short pass cannot reuse stale values.
        float judge_start[N_IND] = {}, judge_end[N_IND] = {};

        // Fresh MT1 for this pass -- see the declaration for why it is never carried over.
        for (int v = 0; v < RACE_N; v++) {
            RaceState& r = race[v];
            const int np = race_params(RACE[v].arch);
            // Seed mixes the entry index, so MT1S-stk-evo really is a different draw from
            // MT1S-sum-evo and the gap between them measures run-to-run variance rather than
            // a difference in method.
            for (int i = 0; i < N_IND; i++) {
                const uint64_t sd = 0xB901ULL ^ ((uint64_t)(pass + 1) << 32)
                                             ^ ((uint64_t)v << 16) ^ (uint64_t)i;
                if (RACE[v].search == SRCH_EVO) {
                    // Carries by design: load_or_init consults output_dir first, where the pool
                    // was written each block, so the search that produced this pass's champion is
                    // the search the next pass continues. Falls back to a fresh kaiming init on
                    // the first pass, or whenever no file exists.
                    load_or_init_mt1_pool(output_dir, load_dir, i, race[v].pool[i],
                                          RACE[v].tag, race_init(RACE[v].arch));
                    continue;
                }
                if (RACE[v].carry && pass > 0) continue;   // keep weights AND Adam moments
                PCG32 rng; rng.seed(mix_seed(sd));
                std::vector<float>& W = r.w[i];
                // Same scale the evolutionary pools start from, so neither search method begins
                // at an advantage. Box-Muller from the shared PCG32 keeps the stream identical in
                // kind to the mutation RNG.
                for (int k = 0; k < np; k++) {
                    const float u1 = fmaxf(rng.next_float(), 1e-7f), u2 = rng.next_float();
                    W[k] = sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2) * 0.05f;
                }
                std::fill(r.m[i].begin(), r.m[i].end(), 0.f);
                std::fill(r.v[i].begin(), r.v[i].end(), 0.f);
                std::fill(r.g[i].begin(), r.g[i].end(), 0.f);
            }
            r.loss_sum = 0.0; r.loss_n = 0; r.rank_sum = 0.0; r.rank_n = 0; r.adam_step = 0;
            r.distinct_sum = 0.0; r.distinct_n = 0;
            r.alloc_flat = r.alloc_top4 = r.alloc_top6 = r.alloc_bot4 = 0.0; r.alloc_n = 0;
            std::fill(r.bk_sum.begin(), r.bk_sum.end(), 0.0);
            std::fill(r.bk_n.begin(),   r.bk_n.end(),   0);
            std::fill(r.xy.begin(), r.xy.end(), 0.0); std::fill(r.xx.begin(), r.xx.end(), 0.0);
            std::fill(r.yy.begin(), r.yy.end(), 0.0); std::fill(r.sx.begin(), r.sx.end(), 0.0);
            std::fill(r.sy.begin(), r.sy.end(), 0.0); std::fill(r.dn.begin(), r.dn.end(), 0);
        }
        rows_feat74 = RowCount{}; rows_cfeat146 = RowCount{}; rows_persym = RowCount{};
        std::fill(bp_bk_sum.begin(), bp_bk_sum.end(), 0.0);
        std::fill(bp_bk_n.begin(), bp_bk_n.end(), 0);

        // Init portfolios; industry elites are loaded per-day inside step_industry
        for (int i = 0; i < N_IND; i++) {
            ind_states[i].portfolios[0].cash = IND_STARTING_CASH;
            for (int j = 0; j < IND_SYMS; j++) {
                ind_states[i].portfolios[0].holdings[j]    = 0.f;
                ind_states[i].portfolios[0].stop_prices[j] = 0.f;
            }
            ind_states[i].streak = 0;
            for (int s = 1; s < N_SLOTS; s++) ind_states[i].portfolios[s] = ind_states[i].portfolios[0];
        }
        // Load MT1 head+tail pools (heads/tails redesign) and MT2 once at pass start
        for (int i = 0; i < N_IND; i++) {
            if (!g_no_nn_race) {
            }
        }
        load_or_init_mt2(output_dir, load_dir, *mt2_scratch);
        // Init MT2 portfolio state
        mst->portfolios[0].cash = MST_STARTING_CASH;
        for (int i = 0; i < N_IND; i++) mst->portfolios[0].holdings[i] = 0.f;
        for (int s = 1; s < N_SLOTS; s++) mst->portfolios[s] = mst->portfolios[0];
        memset(mst->ind_val_hist, 0, sizeof(mst->ind_val_hist));
        memset(mst->mkt_val_hist, 0, sizeof(mst->mkt_val_hist));
        mst->ind_hist_count = 0;
        memset(mst->zero_counts, 0, sizeof(mst->zero_counts));

        // Clear industry OHLCV histories
        for (int i = 0; i < N_IND; i++)
            for (int j = 0; j < IND_SYMS; j++)
                ind_states[i].hist[j] = SymHist{};

        // Warmup history
        warmup_history(ind_states.get(), all_days, day_start);

        int num_days = day_end - day_start;
        IndResult results[N_IND] = {};

        // Stable seq_flags per day (random 50/50 per symbol)
        bool seq_flags[N_SYMS];
        PCG32 seq_rng;

        // ── Heads/tails block-alternating accumulation (Increment 3C) ──
        // Each fwd-valid day's market-derived features + targets are cached; every MT1_DAYS
        // cached days a block is processed (MT1 T1/H/T2 phases, then MT2's M phase, then flush CSV
        // + MT log + save). StockNN, OHLCV/market histories and non-fwd-day CSV rows stay per-day.
        // blk_888  = dual [market‖portfolio] MT1 features per day
        // blk_fwd  = MT1 target: PORTFOLIO forward return over MT1_FWD_DAYS (Part C; matured via the
        //            pending ring below, since the portfolio forward value isn't known until t+FWD)
        // blk_perf = market forward return (MT2 read-only diagnostic; MT2 trains on same-day slot-0)
        static float blk_888[MT1_DAYS][888];
        static float blk_perf[MT1_DAYS][N_IND];        // market forward return — MT2 diagnostic only
        static float blk_mkt_ret_c[MT1_DAYS][N_IND];
        static float blk_mkt_val_c[MT1_DAYS][N_IND];
        static IndResult blk_results[MT1_DAYS][N_IND];
        // Today's bars, for MT1CNet's input. Cached rather than rebuilt because process_block
        // runs after the day loop has moved on, and the competitor's whole premise is that it
        // sees TODAY and nothing else.
        const DayData* blk_day[MT1_DAYS] = {};
        int blk_actual_day[MT1_DAYS];

        auto process_block = [&](int blk_len) {
            if (blk_len <= 0) return;
            // Per-day staging for the allocation backtest: each entry's deployed prediction for
            // every industry, plus that industry's realised return, gathered across the industry
            // loop and resolved once all 12 are present.
            static float race_day_pred[RACE_N][N_IND];
            static float race_day_ret[N_IND];
            static bool  race_day_ok[N_IND];
            // 1. MT1 now steps once per industry per DAY, inside the day loop below — there are
            //    no T1/H/T2 phases to run ahead of it and no block-start snapshot to take. The
            //    prediction each model parked yesterday IS its out-of-sample record, so the
            //    snapshot twin, the skill ring and the OOS/in-sample gap they measured are all
            //    gone with the design that needed them.
            static MT1DayResult mt1_day_res[N_IND][MT1_DAYS];
            static MT1DayResult mt1c_day_res[N_IND][MT1_DAYS];
            memset(mt1_day_res, 0, sizeof(mt1_day_res));
            memset(mt1c_day_res, 0, sizeof(mt1c_day_res));
            // 2. MT2 M phase: replay block days with the post-block composed MT1 (head0+tail0).
            //    MT2 is now GRADED/TRAINED on the deployed slot-0 StockNN portfolio delta
            //    (perf_pf = slot0_score/baseline − 1) — the quantity that actually earns — instead
            //    of the coincident market forward return (blk_perf, kept as a read-only diagnostic).
            MasterResult blk_master_res[MT1_DAYS];
            for (int d = 0; d < blk_len; d++) {
                memset(race_day_ok, 0, sizeof(race_day_ok));
                MasterResult mr{}; bool inj = false;
                float s0_pf = 0.f, s0_mkt = 0.f;
                // MT2's input is now ONE number per industry: MT1's dollar prediction of that
                // industry's next-session P&L. The old 48 carried a confidence, a signless
                // magnitude, a band width and a constant — four channels of which one was
                // ungraded, one was the magnitude with its sign discarded, and one graded itself
                // against the other's residual. The prediction's own sign and size say all of it.
                float in12[N_IND];
                for (int i = 0; i < N_IND; i++) {
                    // Realised P&L for THIS session: the deployed book's value now against its
                    // value at the previous close. Includes the market move on what is held --
                    // this was `slot0_score - baseline`, which marks both sides at the SAME
                    // next-day prices and so cancels beta exactly, leaving only the value added by
                    // the day's trades. That component is ~8% of the book's gain, zero-mean and
                    // 2.9x heavy-tailed, and it is not what "how profitable will this industry be"
                    // means. See CHANGELOG / mt1_target notes.
                    const IndResult& ir = blk_results[d][i];
                    const float actual = ir.slot0_score - ir.book_prev;
                    in12[i] = 0.f;

                    // ── gradient-trained MT1 ────────────────────────────────────────────
                    // PREDICT first, then LEARN, so nothing is ever scored on an outcome it has
                    // already seen. The input is this industry's cfeat -- today only, post close --
                    // and the target is the realised book P&L of the orders just placed.
                    //
                    // ONE TRAINING ROW PER SCORED MODEL, each carrying that model's OWN orders.
                    //
                    // This is the whole point of the per-slot intent capture in IndResult. Every
                    // slot resets to slot 0's portfolio each morning, so all MT1_BP_MODELS of them
                    // see an identical market move and differ only in what they chose to trade.
                    // Feeding a shared input would therefore make the rows near-duplicates: measured
                    // post-day-400 over 855 days x 12 industries, the across-model outcome sd is
                    // $166 against a between-day sd of $729, so with one input the 20 elites were
                    // worth 1.05 independent samples and 200 would have been worth no more.
                    //
                    // Varying the input where the outcome varies is what makes them real rows, and
                    // it changes the question MT1 answers: not "what will this industry earn" but
                    // "what will THIS order set earn" -- which is the form that can rank candidate
                    // order sets before any of them is placed.
                    //
                    // The PREDICTION still comes from slot 0's intent, because slot 0 is what
                    // production actually places. Predict before learning, so no row is ever scored
                    // on an outcome the net has already been shown.
                    if (blk_day[d] && blk_actual_day[d] >= MT1_BP_WARM_DAY) {
                        float bin[MT1C_IN];
                        build_mt1c_input(blk_day[d]->sym[i], ir, bin);
                        bool finite_in = true;
                        for (int q = 0; q < MT1C_IN; q++)
                            if (!std::isfinite(bin[q])) { finite_in = false; break; }
                        // Published in DOLLARS -- the predicted next-session book P&L for this

                        // The whole change is a no-op if every model builds the same input, and
                        // nothing else in the run would say so -- the loss would simply be a 200x
                        // learning-rate multiplier again. Report the spread once, on the first day
                        // that trains, so a broken intent capture is visible in the log.
                        static bool bp_spread_logged = false;
                        if (!bp_spread_logged && finite_in) {
                            // Only 4 of the 12 per-symbol features are order intent; the other 8
                            // are bar shape, identical across slots because every slot resets to
                            // slot 0's portfolio. So the CEILING here is 4 * MT1C_SYMS, and the
                            // gap between that and what we measure is signal that whole_shares()
                            // flooring and the [0,1] clamps quantise away before MT1 sees it.
                            static float mrow[MT1_BP_MODELS][MT1C_IN];
                            int built = 0;
                            for (int m = 0; m < MT1_BP_MODELS; m++) {
                                build_mt1c_input_for(blk_day[d]->sym[i], ir,
                                                     ir.slot_bqty[m], ir.slot_bfrac[m],
                                                     ir.slot_sfrac[m], ir.slot_sqty[m], mrow[m]);
                                bool okm = true;
                                for (int q = 0; q < MT1C_IN; q++)
                                    if (!std::isfinite(mrow[m][q])) { okm = false; break; }
                                if (okm) built++;
                            }
                            int varying = 0, ch[4] = {0, 0, 0, 0};
                            const int CH_IDX[4] = {CN_BQTY, CN_SQTY, CN_BFRAC, CN_SFRAC};
                            for (int q = 0; q < MT1C_IN; q++) {
                                float lo = mrow[0][q], hi = mrow[0][q];
                                for (int m = 1; m < MT1_BP_MODELS; m++) {
                                    lo = fminf(lo, mrow[m][q]); hi = fmaxf(hi, mrow[m][q]);
                                }
                                if (hi - lo > 1e-9f) {
                                    varying++;
                                    if (q < MT1C_SYMS * MT1C_PER_SYM)
                                        for (int c = 0; c < 4; c++)
                                            if (q % MT1C_PER_SYM == CH_IDX[c]) ch[c]++;
                                }
                            }
                            // Distinct rows: how many of the 200 are genuinely different vectors.
                            int distinct = 0;
                            for (int m = 0; m < MT1_BP_MODELS; m++) {
                                bool dup = false;
                                for (int k = 0; k < m && !dup; k++) {
                                    dup = true;
                                    for (int q = 0; q < MT1C_IN && dup; q++)
                                        if (std::fabs(mrow[m][q] - mrow[k][q]) > 1e-9f) dup = false;
                                }
                                if (!dup) distinct++;
                            }
                            log_msg(std::string("[") + IND_SHORT[i] + "]   mt1-bp: " +
                                    std::to_string(varying) + "/" + std::to_string(MT1C_IN) +
                                    " features vary across " + std::to_string(MT1_BP_MODELS) +
                                    " models (ceiling " + std::to_string(4 * MT1C_SYMS) + ")" +
                                    "; bqty " + std::to_string(ch[0]) +
                                    " sqty " + std::to_string(ch[1]) +
                                    " bfrac " + std::to_string(ch[2]) +
                                    " sfrac " + std::to_string(ch[3]) +
                                    "; distinct rows " + std::to_string(distinct) + "/" +
                                    std::to_string(built) +
                                    (varying == 0 ? "  <-- BROKEN: intent capture is not per-model"
                                                  : ""));
                            bp_spread_logged = true;
                        }
                        // Build every model's row ONCE, then hand the same bytes to all eight
                        // race entries. Two passes per entry, and the order is the measurement:
                        // score all MT1_BP_MODELS order sets with the model FROZEN, then learn.
                        // Scoring inside the learning loop would grade model m+1 with a net that
                        // had already seen model m's outcome -- same day, same market move -- and
                        // manufacture a ranking out of nothing.
                        static thread_local float mrows[MT1_BP_MODELS][MT1C_IN];
                        static thread_local float mpred[MT1_BP_MODELS], mreal[MT1_BP_MODELS];
                        static thread_local float msym[MT1_BP_MODELS][IND_SYMS];
                        int nm = 0;
                        for (int m = 0; m < MT1_BP_MODELS; m++) {
                            // RAW DOLLARS in units of the CONSTANT MT1_PRED_SCALE. Dividing by the
                            // industry's own book would be a per-industry rescaling, and whatever
                            // consumes these has to read +$500 the same on a $25k book and a $45k one.
                            const float y = (ir.slot_score[m] - ir.book_prev) / MT1_PRED_SCALE;
                            if (!std::isfinite(y)) continue;
                            build_mt1c_input_for(blk_day[d]->sym[i], ir,
                                                 ir.slot_bqty[m], ir.slot_bfrac[m],
                                                 ir.slot_sfrac[m], ir.slot_sqty[m], mrows[nm]);
                            bool ok_in = true;
                            for (int q = 0; q < MT1C_IN; q++)
                                if (!std::isfinite(mrows[nm][q])) { ok_in = false; break; }
                            if (!ok_in) continue;
                            mreal[nm] = y;
                            for (int j = 0; j < IND_SYMS; j++)
                                msym[nm][j] = ir.slot_sym_pnl[m][j] / MT1_PRED_SCALE;
                            nm++;
                        }

                        // Distinct-row census, once per industry-day for the three input sets.
                        if (nm >= 3 && blk_actual_day[d] >= MT1_BP_START_DAY) {
                            auto row_hash = [](const float* v, int n) {
                                uint64_t h = 1469598103934665603ULL;
                                for (int q = 0; q < n; q++) {
                                    uint32_t b; memcpy(&b, &v[q], sizeof(b));
                                    h ^= b; h *= 1099511628211ULL;
                                }
                                return h;
                            };
                            static thread_local std::vector<uint64_t> hs;
                            auto n_distinct = [&]() {
                                std::sort(hs.begin(), hs.end());
                                return (double)(std::unique(hs.begin(), hs.end()) - hs.begin());
                            };
                            hs.clear();
                            for (int m = 0; m < nm; m++)
                                hs.push_back(row_hash(&blk_888[d][i * 74], 74));
                            rows_feat74.sum += n_distinct(); rows_feat74.n++;

                            hs.clear();
                            for (int m = 0; m < nm; m++) hs.push_back(row_hash(mrows[m], MT1C_IN));
                            rows_cfeat146.sum += n_distinct(); rows_cfeat146.n++;

                            // MT1S-stock's rows are (model, symbol) pairs, so its census runs over
                            // the twelve per-symbol slices of each model rather than whole vectors.
                            hs.clear();
                            for (int m = 0; m < nm; m++)
                                for (int j = 0; j < MT1C_SYMS; j++)
                                    hs.push_back(row_hash(mrows[m] + (size_t)j * MT1C_PER_SYM,
                                                          MT1C_PER_SYM));
                            rows_persym.sum += n_distinct(); rows_persym.n++;
                        }

                        for (int v = 0; v < RACE_N; v++) {
                            const RaceEntry& e = RACE[v];
                            RaceState& r = race[v];
                            // MT1Net reads the industry's 74 curves, which carry no order intent,
                            // so every model presents it the same vector -- see the RACE comment.
                            const float* ind_in = (e.arch == ARCH_MT1)
                                                  ? &blk_888[d][i * 74] : bin;
                            const float* W = (e.search == SRCH_EVO)
                                             ? r.pool[i].slot(r.pool[i].best_slot)
                                             : r.w[i].data();

                            // Measurement window is the SAME for every entry, whatever its
                            // training start: the warm-up days buy Adam maturity, not a longer
                            // scoring window.
                            const bool measure = blk_actual_day[d] >= MT1_BP_START_DAY;
                            const bool train   = blk_actual_day[d] >= race_train_start(e, pass);
                            if (!measure && !train) continue;
                            if (measure)
                            for (int m = 0; m < nm; m++) {
                                const float* row = (e.arch == ARCH_MT1) ? ind_in : mrows[m];
                                mpred[m] = (e.arch == ARCH_MT1)  ? mt1net_forward(W, row, 0.f)
                                         : (e.arch == ARCH_MT1C) ? mt1cnet_forward(W, row, 0.f)
                                                                 : mt1s_forward(W, row);
                            }
                            if (measure && nm >= 3) {
                                static thread_local float srt[MT1_BP_MODELS];
                                memcpy(srt, mpred, (size_t)nm * sizeof(float));
                                std::sort(srt, srt + nm);
                                int dist = 1;
                                for (int m = 1; m < nm; m++)
                                    if (srt[m] != srt[m - 1]) dist++;
                                r.distinct_sum += dist; r.distinct_n++;

                                const float rc = bp_rank_corr(mpred, mreal, nm);
                                if (std::isfinite(rc)) {
                                    r.rank_sum += rc; r.rank_n++;
                                    const int bk = (blk_actual_day[d] - MT1_BP_START_DAY) / BP_BUCKET;
                                    if (bk >= 0 && bk < (int)r.bk_n.size()) {
                                        r.bk_sum[bk] += rc; r.bk_n[bk]++;
                                    }
                                }
                            }
                            if (measure && nm > 0) {   // across-day: slot 0 is production's
                                // staged for the allocation backtest once all 12 are in
                                race_day_pred[v][i] = mpred[0];
                                race_day_ret[i]     = (ir.book_prev > 1.f)
                                                      ? (ir.slot0_score - ir.book_prev) / ir.book_prev
                                                      : 0.f;
                                race_day_ok[i]      = true;
                                const double px = mpred[0], py = mreal[0];
                                r.sx[i] += px; r.sy[i] += py;
                                r.xy[i] += px * py; r.xx[i] += px * px; r.yy[i] += py * py;
                                r.dn[i]++;
                            }

                            if (!train) continue;
                            if (e.search == SRCH_EVO) {
                                // One step per industry-day through the existing pool lifecycle.
                                // Selection sees one scalar per model, so there is no per-row
                                // notion here and nm plays no part.
                                mt1_step_day(i, r.pool[i], ind_in, actual,
                                             blk_actual_day[d] >= race_train_start(e, pass),
                                             blk_actual_day[d], cur_mt1_sigma,
                                             race_forward(e.arch));
                            } else {
                                for (int m = 0; m < nm; m++) {
                                    const float* row = (e.arch == ARCH_MT1) ? ind_in : mrows[m];
                                    float* gg = r.g[i].data();
                                    float* ww = r.w[i].data();
                                    double loss = 0.0;
                                    if (e.arch == ARCH_MT1) {
                                        MT1NetCache c;
                                        mt1net_forward_cached(ww, row, 0.f, c);
                                        const float err = c.out - mreal[m];
                                        mt1net_backward(ww, c, 2.f * err, gg);
                                        loss = (double)err * err;
                                    } else if (e.arch == ARCH_MT1C) {
                                        MT1CNetCache c;
                                        mt1cnet_forward_cached(ww, row, c);
                                        const float err = c.out - mreal[m];
                                        mt1cnet_backward(ww, c, 2.f * err, gg);
                                        loss = (double)err * err;
                                    } else {
                                        MT1SCache c;
                                        mt1s_forward_cached(ww, row, c);
                                        float ds[MT1C_SYMS];
                                        if (e.per_symbol) {
                                            // TWELVE rows from one industry-day: each symbol is
                                            // graded on its own realised P&L, which is what the
                                            // decomposition in step_industry exists for.
                                            for (int j = 0; j < MT1C_SYMS; j++) {
                                                const float ej = c.per_sym[j] - msym[m][j];
                                                ds[j] = 2.f * ej;
                                                loss += (double)ej * ej;
                                            }
                                            loss /= MT1C_SYMS;
                                        } else {
                                            const float err = c.out - mreal[m];
                                            for (int j = 0; j < MT1C_SYMS; j++) ds[j] = 2.f * err;
                                            loss = (double)err * err;
                                        }
                                        mt1s_backward(ww, c, ds, gg);
                                    }
                                    r.adam_step++;
                                    race_adam(r, i, r.adam_step);
                                    r.loss_sum += loss; r.loss_n++;
                                }
                            }
                        }
                        // the deployed prediction stays MT1C-grad, published in dollars
                        if (finite_in && !race[RACE_DEPLOYED].w[i].empty())
                            bp_pred[i] = mt1cnet_forward(race[RACE_DEPLOYED].w[i].data(), bin, 0.f)
                                         * MT1_PRED_SCALE;
                    }

                    // ── online conditional models ───────────────────────────────────────
                    // Trained here, during the pass, from the scores already in hand. Every
                    // scored elite contributes an outcome, not just the deployed slot.
                    bo_pnl_today[i] = actual;
                    for (int e = 0; e < ELITE_POOL; e++)
                        bo_el_today[i][e] = ir.elite_score[e] - ir.book_prev;
                    bo_have[i] = true;
                }

                // ── allocation backtest: rank the 12, place capital, compare to flat ──
                //
                // Runs once per day with all twelve industries in hand. Equal weight across the
                // top k by prediction, against an equal 1/12 book -- long-only, matching what
                // production does. bot4 is kept so the long-short spread can be read: it cancels
                // the market move common to all twelve, which the long-only numbers do not.
                {
                    int ok_n = 0;
                    for (int i = 0; i < N_IND; i++) if (race_day_ok[i]) ok_n++;
                    if (ok_n >= 8) {
                        double flat = 0.0;
                        for (int i = 0; i < N_IND; i++)
                            if (race_day_ok[i]) flat += race_day_ret[i];
                        flat /= ok_n;
                        int idx[N_IND];
                        for (int v = 0; v < RACE_N; v++) {
                            RaceState& r = race[v];
                            int n = 0;
                            for (int i = 0; i < N_IND; i++) if (race_day_ok[i]) idx[n++] = i;
                            // sort_index_prefix, not std::sort: at -O3 the latter emits an
                            // -Warray-bounds false positive on a small int[] (same reason it was
                            // replaced at five other sites -- see sort_util.h).
                            sort_index_prefix(idx, n, [&](int a, int b) {
                                return race_day_pred[v][a] > race_day_pred[v][b];
                            });
                            auto mean_of = [&](int from, int cnt) {
                                double t = 0.0;
                                for (int k = 0; k < cnt; k++) t += race_day_ret[idx[from + k]];
                                return cnt ? t / cnt : 0.0;
                            };
                            r.alloc_flat += flat;
                            r.alloc_top4 += mean_of(0, std::min(4, n));
                            r.alloc_top6 += mean_of(0, std::min(6, n));
                            r.alloc_bot4 += mean_of(std::max(0, n - 4), std::min(4, n));
                            r.alloc_n++;
                        }
                    }
                }

                // ── online conditional models: predict, score, then learn ────────────────
                // Order matters: every model is asked for day d BEFORE day d is folded in, so
                // nothing is ever scored on an outcome it has already seen.
                {
                    if (!bo_csv) {
                        std::string bp = log_dir + "/bayes_online.csv";
                        bo_csv = fopen(bp.c_str(), "w");
                        if (bo_csv) fprintf(bo_csv, "pass,day,model,preds,hits,base_up,"
                                                    "elite_obs,p_up_less,p_dn_less,n_up_less\n");
                    }
                    BODay contrib[10];
                    int  hits[10] = {}, preds[10] = {};
                    int  base_up = 0, n_base = 0, elite_obs = 0;
                    for (int i = 0; i < N_IND; i++) {
                        if (!bo_have[i] || !bo_ok1[i]) continue;
                        const float cur = bo_pnl_today[i];
                        if (cur == 0.f || !std::isfinite(cur)) continue;
                        base_up += (cur > 0.f) ? 1 : 0;
                        n_base++;
                        for (size_t m = 0; m < bo_family.size(); m++) {
                            // ONE call per industry-day, formed from the whole elite pool: each
                            // elite's own state gives P(today keeps ITS sign), which converts to
                            // P(today is up) and is averaged. Twenty separate predictions of the
                            // same outcome would be twenty correlated votes, not twenty tests.
                            double p_up = 0.0;
                            int used = 0;
                            for (int e = 0; e < ELITE_POOL; e++) {
                                const float y1 = bo_prev1_el[i][e];
                                if (y1 == 0.f || !std::isfinite(y1)) continue;
                                const int st = (bo_family[m].k == 1)
                                             ? bo_state1(y1) : (bo_ok2[i] ? bo_state2(bo_prev2[i], y1) : -1);
                                if (st < 0) continue;
                                const double pc = bo_family[m].p_continue(st);
                                p_up += (y1 > 0.f) ? pc : (1.0 - pc);
                                used++;
                                contrib[m].n[st]    += 1.0;
                                contrib[m].same[st] += (std::signbit(cur) == std::signbit(y1)) ? 1.0 : 0.0;
                                contrib[m].sum[st]  += cur;
                                if (m == 0) elite_obs++;
                            }
                            if (!used) continue;
                            p_up /= used;
                            preds[m]++;
                            if ((p_up > 0.5) == (cur > 0.f)) hits[m]++;
                        }
                    }
                    for (size_t m = 0; m < bo_family.size(); m++) {
                        if (bo_csv && preds[m] > 0)
                            fprintf(bo_csv, "%d,%d,%s,%d,%d,%d,%d,%.4f,%.4f,%.1f\n",
                                    pass + 1, blk_actual_day[d], bo_family[m].name,
                                    preds[m], hits[m], base_up, elite_obs,
                                    bo_family[m].p_continue(BO_UP_LESS),
                                    bo_family[m].p_continue(BO_DN_LESS),
                                    bo_family[m].n[BO_UP_LESS]);
                        bo_family[m].commit(contrib[m]);
                    }
                    // Flush every day. Buffered, the whole file is lost if a long run is
                    // interrupted -- and this log only exists because it is cheaper than keeping
                    // the raw data. One fflush per day against a ~55 s day is free.
                    if (bo_csv) fflush(bo_csv);
                    (void)n_base;
                    for (int i = 0; i < N_IND; i++) {
                        if (!bo_have[i]) continue;
                        // Shift, and the ORDER here is the whole correctness of the state.
                        // bo_prev2 must lag bo_prev1_el by exactly one day, so it takes
                        // YESTERDAY's slot-0 figure -- which is bo_prev1_el[i][0], since elite 0
                        // IS slot 0 -- and must be read before that array is overwritten. Taking
                        // today's slot-0 instead paired it with today's elites, making the
                        // "accelerated / decelerated" state a same-day cross-section between
                        // slots rather than a two-day trajectory of one.
                        if (bo_ok1[i]) { bo_prev2[i] = bo_prev1_el[i][0]; bo_ok2[i] = true; }
                        for (int e = 0; e < ELITE_POOL; e++) bo_prev1_el[i][e] = bo_el_today[i][e];
                        bo_ok1[i]   = true;
                        bo_have[i]  = false;
                    }
                }

                if (blk_actual_day[d] >= MASTER_START_DAY && !g_no_nn_race) {
                    // Deployed slot-0 portfolio return (training target) + market return (diagnostic).
                    float perf_pf[N_IND], perf_mkt[N_IND];
                    for (int i = 0; i < N_IND; i++) {
                        float bl = blk_results[d][i].baseline;
                        perf_pf[i]  = (bl > 1e-6f) ? (blk_results[d][i].slot0_score / bl - 1.f) : 0.f;
                        perf_mkt[i] = blk_perf[d][i];   // cached market forward return
                    }
                    // Read-only dual grade of the deployed slot-0 (BEFORE step_mt2 reselects the pool):
                    // the same allocation decision scored on both objectives, so the gap is visible.
                    s0_pf  = drift_score_mt2_pts(mt2_scratch->elite(0), in12, perf_pf);
                    s0_mkt = drift_score_mt2_pts(mt2_scratch->elite(0), in12, perf_mkt);
                    mr = step_mt2(*mst, *mt2_scratch, in12, perf_pf,
                                  blk_actual_day[d], total_days, cur_mt2_sigma, &inj);
                }
                mr.slot0_pts_pf  = s0_pf;
                mr.slot0_pts_mkt = s0_mkt;
                blk_master_res[d] = mr;
            }
            // 3. CSV rows for the block (deferred so each row carries its MT2 result)
            if (csv)
                for (int d = 0; d < blk_len; d++)
                    write_csv_row(csv, pass, blk_actual_day[d], blk_results[d],
                                  blk_master_res[d], blk_mkt_ret_c[d], blk_mkt_val_c[d]);
            // 3b. MT1 dataset: the features and the decomposed outcome, per day per industry.
            //     Written before the MT log so a run killed mid-write loses at most one row of
            //     the artefact the horizon decision depends on.
            if (ds_log) {
                for (int d = 0; d < blk_len; d++) {
                    DSLogRecord ds{};
                    ds.pass_num   = (uint32_t)pass;
                    ds.actual_day = (uint32_t)blk_actual_day[d];
                    for (int i = 0; i < N_IND; i++) {
                        memcpy(ds.feat[i], &blk_888[d][i * DS_FEAT], DS_FEAT * sizeof(float));
                        const IndResult& ir = blk_results[d][i];
                        ds.book_prev[i]   = ir.book_prev;
                        ds.book_now[i]    = ir.slot0_score;
                        ds.mkt_move[i]    = ir.baseline    - ir.book_prev;
                        ds.trade_delta[i] = ir.slot0_score - ir.baseline;
                        ds.oi_n_buy[i]     = ir.oi_n_buy;
                        ds.oi_n_sell[i]    = ir.oi_n_sell;
                        ds.oi_buy_aggr[i]  = ir.oi_buy_aggr;
                        ds.oi_sell_aggr[i] = ir.oi_sell_aggr;
                        ds.oi_buy_disp[i]  = ir.oi_buy_disp;
                        ds.oi_buy_val[i]   = ir.oi_buy_val;
                        ds.oi_sell_val[i]  = ir.oi_sell_val;
                        if (blk_day[d])
                            build_mt1c_input(blk_day[d]->sym[i], ir, ds.cfeat[i]);
                        ds.m_pred[i]  = mt1_day_res[i][d].pred0;
                        ds.m_score[i] = mt1_day_res[i][d].score0;
                        ds.c_pred[i]  = mt1c_day_res[i][d].pred0;
                        ds.c_score[i] = mt1c_day_res[i][d].score0;
                        // MT2INet retired in v0.8.1.28; the fields stay so the
                        // dataset layout and its reader are unchanged.
                        ds.i_pred[i]  = 0.f;
                        ds.i_score[i] = 0.f;
                    }
                    write_ds_log_record(ds_log, ds);
                }
            }

            // 4. One MT log record per BLOCK-DAY (V9). Was one per 25-day block, which gave 50
            //    points per pass and made per-day pool dynamics invisible.
            if (mt_log) {
                for (int d = 0; d < blk_len; d++) {
                    if (blk_actual_day[d] < MT1_START_DAY) continue;
                    MTLogRecord rec{};
                    rec.pass_num   = (uint32_t)pass;
                    rec.actual_day = (uint32_t)blk_actual_day[d];
                    for (int i = 0; i < N_IND; i++) {
                        const MT1DayResult& m = mt1_day_res[i][d];
                        // MT1 evo: same architecture, same lifecycle as the standalone
                        // pool this used to read.
                        const MT1PoolScratch& sc = race[RACE_LOG_POOL].pool[i];
                        rec.mt1_pred[i]       = m.pred0;
                        rec.mt1_actual_d[i]   = m.actual;
                        rec.mt1_baseline[i]   = m.baseline;
                        rec.mt1_floor[i]      = m.floor_v;
                        rec.mt1_score0[i]     = m.score0;
                        rec.mt1_score_mean[i] = m.score_mean;
                        rec.mt1_score_best[i] = m.score_best;
                        rec.mt1_score_min[i]  = m.score_min;
                        rec.mt1_pool_stats[i][0] = (float)m.mature;
                        rec.mt1_pool_stats[i][1] = (float)m.culled;
                        rec.mt1_pool_stats[i][2] = (float)m.lineage_max;
                        rec.mt1_pool_stats[i][3] = (float)m.lineage_n;
                        rec.mt1_pool_stats[i][4] = (sc.retire_n > 0)
                            ? (float)(sc.retire_age_sum / (double)sc.retire_n) : 0.f;
                        for (int b = 0; b < 5; b++)
                            rec.mt1_life[i][b] = (float)sc.retire_hist[b];
                    }
                    const MasterResult& lm = blk_master_res[d];
                    rec.mt2_best_pts   = lm.best_pts;
                    rec.mt2_slot0_pts  = lm.elite_mean_pts;
                    rec.mt2_ideal_pts  = lm.ideal_pts;
                    rec.mt2_injected   = 0u;
                    rec.mt2_consensus_flat_pts = lm.consensus_flat_pts;
                    rec.mt2_consensus_wtd_pts  = lm.consensus_wtd_pts;
                    rec.mt2_slot0_pts_pf       = lm.slot0_pts_pf;
                    rec.mt2_slot0_pts_mkt      = lm.slot0_pts_mkt;
                    write_mt_log_record(mt_log, rec);
                }
            }
            // 5. Save (per block ≈ 25 days). Always — under --no-save output_dir is the scratch.
            log_mem_stat(num_workers);
            for (int i = 0; i < N_IND; i++) {
                for (int v = 0; v < RACE_N; v++) {
                    if (RACE[v].search != SRCH_GRAD || race[v].w[i].empty()) continue;
                    save_bin(race_weight_path(output_dir, v, i, pass + 1),
                             race[v].w[i].data(), (int)race[v].w[i].size());
                }
                for (int v = 0; v < RACE_N; v++) {
                    if (RACE[v].search != SRCH_EVO) continue;
                    const MT1PoolScratch& sc = race[v].pool[i];
                    // Per-pass snapshot of the deployed model, for the weight comparison...
                    save_bin(race_weight_path(output_dir, v, i, pass + 1),
                             sc.slot(sc.best_slot), sc.n_params);
                    // ...and the whole pool, which is what the NEXT pass loads to carry on.
                    save_mt1_pool(output_dir, i, sc, RACE[v].tag);
                }
                if (!g_no_nn_race) {
                }
            }
            save_mt2_elites(output_dir, *mt2_scratch);
        };

        for (int day_num = 0; day_num < num_days; day_num++) {
            int actual_day = day_start + day_num;

            // Yield the box to a production cycle if one is running. Between days only — worker
            // threads are parked here, so this idles the whole trainer.
            wait_while_trading();

            // Pass-boundary metric: slot-0 value PASS_JUDGE_DAYS before the end, and on the final
            // day. Logged "Day N" is actual_day + 1, so the window matches days
            // (day_end - PASS_JUDGE_DAYS) .. day_end in log terms.
            const bool judge_win = (day_end - day_start) > PASS_JUDGE_DAYS;
            const DayData* day_ptr  = &all_days[actual_day];
            const DayData* fill_ptr = (actual_day + 1 < total_days) ? &all_days[actual_day + 1] : nullptr;

            // Generate seq_flags for this day
            seq_rng.seed(mix_seed((uint64_t)actual_day * 0xABCDEF01234567ULL));
            for (int si = 0; si < N_SYMS; si++)
                seq_flags[si] = (seq_rng.next() & 1);

            // Set up worker context for this day
            wctx.day_ptr    = day_ptr;
            wctx.fill_ptr   = fill_ptr;
            wctx.results    = results;
            wctx.seq_flags  = seq_flags;
            wctx.actual_day = actual_day;
            wctx.total_avail = total_days;
            wctx.day_num    = day_num;
            wctx.num_days   = num_days;
            wctx.sigma      = cur_sigma;
            wctx.freeze     = false;
            wctx.master_only = master_only;
            wctx.next_ind.store(0, std::memory_order_relaxed);

            // Dispatch workers
            wctx.work_ready.release(num_workers);

            // Wait for all workers to finish
            for (int w = 0; w < num_workers; w++) wctx.work_done.acquire();

            // Update industry OHLCV histories (main thread is canonical)
            for (int i = 0; i < N_IND; i++)
                for (int j = 0; j < IND_SYMS; j++)
                    update_hist_sym(ind_states[i].hist[j], day_ptr->sym[i][j]);

            // Single-day equal-weight market return per industry — used ONLY for the
            // backward-looking feature index mkt_val_hist and CSV logging (NOT the target).
            float mkt_ret[N_IND] = {};
            if (fill_ptr != nullptr) {
                for (int i = 0; i < N_IND; i++) {
                    float sum = 0.f; int valid = 0;
                    for (int j = 0; j < IND_SYMS; j++) {
                        const OHLCV& tod = day_ptr->sym[i][j];
                        const OHLCV& nxt = fill_ptr->sym[i][j];
                        if (tod.valid && nxt.valid && tod.close > 1e-3f) {
                            sum += nxt.close / tod.close - 1.f; valid++;
                        }
                    }
                    mkt_ret[i] = valid > 0 ? sum / valid : 0.f;
                }
            }

            // FORWARD prediction target: equal-weight cumulative relative return over the next
            // MT1_FWD_DAYS sessions (close[t+FWD]/close[t]-1, telescoping). Drives MT1 + MT2.
            // Only defined when the forward window exists (skip the last MT1_FWD_DAYS days).
            const DayData* fwd_ptr = (actual_day + MT1_FWD_DAYS < total_days)
                                       ? &all_days[actual_day + MT1_FWD_DAYS] : nullptr;
            const bool fwd_valid = (fwd_ptr != nullptr);
            float fwd_ret[N_IND] = {};
            if (fwd_valid) {
                for (int i = 0; i < N_IND; i++) {
                    float sum = 0.f; int valid = 0;
                    for (int j = 0; j < IND_SYMS; j++) {
                        const OHLCV& tod = day_ptr->sym[i][j];
                        const OHLCV& fwd = fwd_ptr->sym[i][j];
                        if (tod.valid && fwd.valid && tod.close > 1e-3f) {
                            sum += fwd.close / tod.close - 1.f; valid++;
                        }
                    }
                    fwd_ret[i] = valid > 0 ? sum / valid : 0.f;
                }
            }
            // (median removed — MT1 predicts the ABSOLUTE own-industry forward return;
            //  cross-sectional relativization is MT2's job via opt_tier on perf below.)

            // Build the 888-feature DUAL vector: [37 market-index ‖ 37 portfolio(slot-0)] features
            // per industry, from histories BEFORE today's append (backward-looking).
            float today888[888];
            build_master_features(mst->mkt_val_hist, mst->ind_val_hist, mst->ind_hist_count, today888);

            // MT1 + MT2 no longer run per day — MT1 trains in block-alternating phases (post-forward
            // buffer) and MT2 in the post-block M phase. This day's features + context are buffered
            // in the pending ring below and fed into a block once its forward window closes.

            // Append today's cumulative indices. BOTH now COMPOUND daily returns so a 10-day forward
            // ratio is a meaningful cumulative return:
            //   market index    = prev × (1 + mkt_ret)
            //   portfolio index = prev × (1 + slot0_return),  slot0_return = slot0_score/baseline − 1
            float today_mkt_val[N_IND], today_pf_val[N_IND];
            for (int i = 0; i < N_IND; i++) {
                float prev_mkt = (mst->ind_hist_count > 0)
                    ? (mst->ind_hist_count < IND_HIST_CAP ? mst->mkt_val_hist[i][mst->ind_hist_count - 1]
                                                          : mst->mkt_val_hist[i][IND_HIST_CAP - 1])
                    : static_cast<float>(IND_STARTING_CASH);
                today_mkt_val[i] = prev_mkt * (1.f + mkt_ret[i]);
                float prev_pf = (mst->ind_hist_count > 0)
                    ? (mst->ind_hist_count < IND_HIST_CAP ? mst->ind_val_hist[i][mst->ind_hist_count - 1]
                                                          : mst->ind_val_hist[i][IND_HIST_CAP - 1])
                    : static_cast<float>(IND_STARTING_CASH);
                if (judge_win) {
                    if (actual_day == day_end - PASS_JUDGE_DAYS - 1) judge_start[i] = results[i].baseline;
                    if (actual_day == day_end - 1)                   judge_end[i]   = results[i].baseline;
                }
                // The portfolio index now compounds the BOOK's return, not the trading ratio.
                // It used to be slot0_score/baseline - 1, which is the trade delta over the
                // reference book -- so the curve MT1 saw ended pass 1 near -7% while the book it
                // was supposed to describe ended +80%. MT1's portfolio-half features were built
                // on that curve, so it could see neither the book's level nor the market move on
                // it. Same denominator as the market index (both compound a close-to-close
                // return from IND_STARTING_CASH), so the two halves stay on one scale.
                float book_ret = (results[i].book_prev > 1e-6f)
                    ? (results[i].slot0_score / results[i].book_prev - 1.f) : 0.f;
                today_pf_val[i] = prev_pf * (1.f + book_ret);
                if (mst->ind_hist_count < IND_HIST_CAP) {
                    mst->mkt_val_hist[i][mst->ind_hist_count] = today_mkt_val[i];
                    mst->ind_val_hist[i][mst->ind_hist_count] = today_pf_val[i];
                } else {
                    memmove(mst->mkt_val_hist[i], mst->mkt_val_hist[i] + 1, (IND_HIST_CAP - 1) * sizeof(float));
                    mst->mkt_val_hist[i][IND_HIST_CAP - 1] = today_mkt_val[i];
                    memmove(mst->ind_val_hist[i], mst->ind_val_hist[i] + 1, (IND_HIST_CAP - 1) * sizeof(float));
                    mst->ind_val_hist[i][IND_HIST_CAP - 1] = today_pf_val[i];
                }
            }
            if (mst->ind_hist_count < IND_HIST_CAP) mst->ind_hist_count++;

            // Feed today straight into the MT1/MT2/log step. Nothing is buffered: the target is
            // this session's realised P&L, which `results` already holds, so a day is mature the
            // moment it is computed. The 21-deep ring that used to sit here existed only to
            // wait out a 10-day-forward return and a 20-day-forward volatility, and both are gone.
            memcpy(blk_888[0],       today888,      sizeof(today888));
            memcpy(blk_results[0],   results,       sizeof(IndResult) * N_IND);
            memcpy(blk_mkt_ret_c[0], mkt_ret,       sizeof(mkt_ret));
            memcpy(blk_mkt_val_c[0], today_mkt_val, sizeof(today_mkt_val));
            memcpy(blk_perf[0],      fwd_ret,       sizeof(fwd_ret));   // MT2 market diagnostic only
            blk_day[0]        = day_ptr;
            blk_actual_day[0] = actual_day;
            process_block(1);
        }

        // Save MT1/MT2 after each pass (industry elites already saved by step_industry)
        {
            log_msg("Pass " + std::to_string(pass+1) + " complete — saving MT1/MT2 elites");
            for (int i = 0; i < N_IND; i++) {
            }
            save_mt2_elites(output_dir, *mt2_scratch);

            // Judge this pass against the standing champion and seed the next one. Skipped in
            // --master-only (StockNN is frozen, so the metric is meaningless) and for passes
            // shorter than the judging window.
            // Never in control mode: crowning a champion from random weights would write those
            // weights into champion/, where the NEXT real run would seed from them.
            if (!master_only && !g_control_untrained && (day_end - day_start) > PASS_JUDGE_DAYS)
                pass_boundary(output_dir, pass + 1, day_start, day_end,
                              judge_start, judge_end, /*seed_next=*/pass + 1 < passes);

            // ── how well did the gradient MT1 predict the TRADING OUTCOME? ──────────────
            //
            // Both numbers are out-of-sample by construction: the net predicts a day before it
            // trains on it, and within a day it predicts every order set before training on any
            // of them.
            //
            // within-day rank is the one the per-model inputs exist for. All MT1_BP_MODELS share
            // the day's market move, so ranking them is ranking the TRADING and nothing else.
            // 0.000 means the order features carry no information about which order set wins.
            //
            // slot-0 corr is the deployed model's prediction against its realised P&L across
            // days. That series still contains the common market move, so it is the easier
            // number and the one an allocator would consume -- do not read it as evidence about
            // the trading.
            // Gate on whether anything was MEASURED, not on race[0]'s rank count: race[0] is the
            // MT1 control, whose predictions tie on all 200 by design, so bp_rank_corr returns NaN
            // every day and its rank_n stays 0 -- gating on it would suppress the whole table.
            bool race_measured = false;
            for (int v = 0; v < RACE_N; v++) race_measured |= race[v].distinct_n > 0;
            if (race_measured) {
                char m[256];
                log_msg("   ===== MT1 race, pass " + std::to_string(pass + 1) +
                        " : 3 architectures x 2 search methods =====");
                log_msg("   within-day rank = Spearman across the 200 order sets scored that "
                        "industry-day.");
                log_msg("   They share the day's market move, so it ranks the TRADING alone.");
                log_msg("   MT1 rows MUST read ~0.0000: their 74 inputs carry no order intent, so "
                        "all 200");
                log_msg("   models look identical to them. Non-zero there means the measurement "
                        "leaks.");
                log_msg("   MT1S-stk evo is a SEED REPLICATE of MT1S-sum evo -- the gap between "
                        "them is noise.");
                log_msg("   grdC = gradient CARRIED across the pass boundary; grad = restarted. "
                        "evo always carries.");
                {
                    char c[160];
                    snprintf(c, sizeof(c),
                             "   grdC trains from day %d in passes 2+ (stale-Adam re-adaptation); "
                             "SCORED from day %d like everything else.",
                             MT1_BP_WARM_DAY, MT1_BP_START_DAY);
                    log_msg(c);
                    log_msg("   In PASS 1 grdC has nothing to carry and the same start as grad, "
                            "so their gap there is the noise floor.");
                log_msg("   alloc bp/day is the PAPER-READINESS read: top4 vs flat is what "
                        "switching off --flat-allocation would have earned.");
                }
                {
                    char c[256];
                    auto line = [&](const char* lab, const RowCount& rc, int per_day) {
                        if (rc.n == 0) return;
                        const double dm = rc.sum / (double)rc.n;
                        snprintf(c, sizeof(c),
                                 "   distinct rows/ind-day  %-14s %7.1f of %4d  (%.1f%% duplicated)",
                                 lab, dm, per_day, 100.0 * (1.0 - dm / per_day));
                        log_msg(c);
                    };
                    line("feat74  (MT1)", rows_feat74,   MT1_BP_MODELS);
                    line("cfeat146(MT1C)", rows_cfeat146, MT1_BP_MODELS);
                    line("per-sym (MT1S-stk)", rows_persym, MT1_BP_MODELS * MT1C_SYMS);
                    log_msg("   feat74 must read 1.0: no order intent, so all 200 models look "
                            "identical to MT1.");
                }
                for (int v = 0; v < RACE_N; v++) {
                    const RaceState& r = race[v];
                    if (r.distinct_n == 0) continue;   // never measured this pass
                    // rank_n == 0 is the CONTROL's correct outcome, not a missing row: all 200
                    // predictions tie every day, so no day yields a finite Spearman. Print it
                    // explicitly -- dropping the row would hide the one result the control exists
                    // to show, along with its alloc and corr lines.
                    char rk[48];
                    if (r.rank_n > 0) snprintf(rk, sizeof(rk), "%+.4f", r.rank_sum / (double)r.rank_n);
                    else              snprintf(rk, sizeof(rk), "  n/a  ");
                    char ls[24];
                    if (r.loss_n > 0) snprintf(ls, sizeof(ls), "%.5f", r.loss_sum / (double)r.loss_n);
                    else              snprintf(ls, sizeof(ls), "   --  ");
                    snprintf(m, sizeof(m),
                             "   %-14s loss %s | within-day rank %s over %ld ind-days%s",
                             RACE[v].name, ls, rk, r.rank_n,
                             r.rank_n == 0 ? "  (all tied every day: cannot rank)" : "");
                    log_msg(m);
                    if (r.distinct_n > 0) {
                        const double dm = r.distinct_sum / (double)r.distinct_n;
                        const double tie_frac = 1.0 - dm / (double)MT1_BP_MODELS;
                        // (k/n)^2 bounds how far mishandled ties can move a Spearman, with k the
                        // mean tie-group width. Printed next to the rank so the two are read
                        // together rather than the bias being assumed negligible.
                        const double kbar = tie_frac > 0 ? (double)MT1_BP_MODELS / dm : 1.0;
                        snprintf(m, sizeof(m),
                                 "        distinct preds %.1f/%d (%.1f%% tied, tie-bias bound ~%.1e)",
                                 dm, MT1_BP_MODELS, 100.0 * tie_frac,
                                 (kbar / MT1_BP_MODELS) * (kbar / MT1_BP_MODELS));
                        log_msg(m);
                    }
                    if (r.alloc_n > 0) {
                        // Basis points per day. The long-only line is the paper-readiness read:
                        // production allocates long-only, so "top4 - flat" is literally what
                        // switching off --flat-allocation would have earned. long-short cancels
                        // the market move common to all twelve and is the cleaner read on whether
                        // the ranking carries information at all.
                        const double n = (double)r.alloc_n;
                        const double fl = 1e4 * r.alloc_flat / n;
                        const double t4 = 1e4 * r.alloc_top4 / n;
                        const double t6 = 1e4 * r.alloc_top6 / n;
                        const double b4 = 1e4 * r.alloc_bot4 / n;
                        snprintf(m, sizeof(m),
                                 "        alloc bp/day: flat %+.2f | top4 %+.2f (%+.2f vs flat) | "
                                 "top6 %+.2f (%+.2f) | long-short %+.2f",
                                 fl, t4, t4 - fl, t6, t6 - fl, t4 - b4);
                        log_msg(m);
                    }
                    std::string traj = "        by day:";
                    for (size_t b = 0; b < r.bk_n.size(); b++) {
                        if (r.bk_n[b] < 12) continue;
                        char c[64];
                        snprintf(c, sizeof(c), " %d:%+.4f",
                                 MT1_BP_START_DAY + (int)b * BP_BUCKET,
                                 r.bk_sum[b] / (double)r.bk_n[b]);
                        traj += c;
                    }
                    if (traj.size() > 16) log_msg(traj);
                    // across-day: the deployed model against realised industry P&L, averaged over
                    // industries. This series still contains the common market move, so it is the
                    // easier number and the one an allocator would consume -- not evidence about
                    // the trading.
                    double rsum = 0.0; int rn = 0;
                    for (int i = 0; i < N_IND; i++) {
                        if (r.dn[i] < 3) continue;
                        const double n = (double)r.dn[i];
                        const double cov = r.xy[i] - r.sx[i] * r.sy[i] / n;
                        const double vx  = r.xx[i] - r.sx[i] * r.sx[i] / n;
                        const double vy  = r.yy[i] - r.sy[i] * r.sy[i] / n;
                        if (vx > 0.0 && vy > 0.0) { rsum += cov / sqrt(vx * vy); rn++; }
                    }
                    if (rn > 0) {
                        snprintf(m, sizeof(m),
                                 "        across-day corr %+.4f (mean of %d industries)",
                                 rsum / rn, rn);
                        log_msg(m);
                    }
                }
            }
        }
    }

    // Shutdown workers
    wctx.shutdown.store(true);
    wctx.work_ready.release(num_workers);
    for (int w = 0; w < num_workers; w++) wctx.work_done.acquire();
    for (auto& t : workers) t.join();

    if (csv)    fclose(csv);
    if (mt_log) fclose(mt_log);
    if (ds_log) fclose(ds_log);

    if (!nosave_scratch.empty()) {
        std::error_code ec;
        fs::remove_all(nosave_scratch, ec);
        log_msg(ec ? "WARNING: could not remove scratch " + nosave_scratch
                   : "--no-save: removed scratch " + nosave_scratch);
    }

    if (bo_csv) { fclose(bo_csv); bo_csv = nullptr; }

    log_msg("Training complete.");
    return 0;
}
