// training_v4.cpp — C++ port of training_v4.py
// Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
// Run:   ./build/training_v4_cpp --output models [--load-dir DIR] [--start-day N] [--stop-day N]
//        [--passes N] [--sigma F] [--master-sigma F] [--sigma-decay F] [--workers N]

#define TRAINER_VERSION "0.6.2.4"

#include <algorithm>
#include <atomic>
#include <cassert>
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

#include "mt1_scoring.h"   // MT1 decode helpers + compute_mt1_scores (shared with tests/test_mt1.cpp)

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
static constexpr int   MT1_COMP_SLOTS      = 200;                           // slots per pool
static constexpr int   HT_PARENTS          = ELITE_COUNT + WAVG_COUNT;      // 20 (17 elites + 3 wavg)
static constexpr int   HT_MUTS             = MT1_COMP_SLOTS - HT_PARENTS;   // 180 mutations
// MT1_RANGE_CEIL_MULT / MT1_LOGIT_CAP / MT1_SOFTPLUS_CLAMP live in mt1_scoring.h
static constexpr int   MT1_DIR_DAYS        = 10;  // scoring window (all pools): score each model over last N days, linear-weighted sum (oldest=1.0 → today=2.0)
static constexpr int   MT1_DIR_MIN_CORRECT  = 3;   // direction collapse floor: freeze the pool only when the best model got < N of the window's days' direction right (genuine collapse; random ≈ 5/10, so this rarely trips). Skill is optimized by the two-half selection, not this gate.
// Direction constant-collapse INJECTION (v0.4.2.0). The v0.4.1.0 run showed 9/12 deployed direction
// models collapsed to a CONSTANT predictor (all-up or all-down every day), each scoring exactly the
// class-balanced no-skill baseline (7.50) with no way out — MT1 tail pools were the only pools in the
// system with no re-diversification trigger. This detects that signature and kicks the direction tail:
// fire when, for MT1_DIR_CONST_TRIP consecutive daily checks, the deployed direction model predicts a
// single direction across its whole MT1_DIR_DAYS window (all up OR all down) AND ≥1 of those window
// days is wrong (i.e. it is not perfectly riding a genuine trend).
static constexpr int   MT1_DIR_CONST_TRIP   = 3;            // consecutive constant+imperfect checks → inject
static constexpr int   MT1_DIR_INJ_COOLDOWN = MT1_DIR_DAYS; // checks to wait before another injection
static constexpr float MT1_DIR_INJ_BLEND    = 0.5f;         // injected tail = BLEND×best + (1−BLEND)×random

// Direction pool: forward accumulation (v0.5.0.0). The record layout, its scoring, and the
// lifecycle constants live in mt1_scoring.h so the C++ tests grade the same arithmetic production
// does. Only the pieces the trainer alone needs stay here.
static constexpr int   MT1_DIR_WAVG_K[3]   = {5, 10, 15};  // ephemeral blend parents (top-5/10/15)
static constexpr int   DIR_META_VERSION    = 1;      // mt1_{ind}_tail_dir_meta.bin sidecar version

static constexpr int   HIST_WINDOW         = 15;

static constexpr float IND_STARTING_CASH   = 25000.0f;
static constexpr float MST_STARTING_CASH   = 300000.0f;
static constexpr float IND_UNIT_PRICE      = 25000.0f;
static constexpr float MAX_SINGLE_STOCK_PCT = 0.60f;
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

// MT1: per-industry preprocessor (12 pools). Sizes live with the heads/tails split below —
// HEADNN_PARAMS + 4 × TAILNN_PARAMS. There is no single MT1NN_PARAMS constant by design.
static constexpr int   MT1_START_DAY       = 25;
static constexpr int   MT1_FWD_DAYS        = 10;       // prediction horizon: target = cumulative relative return over next N sessions
static constexpr int   MT1_BLOCK_DAYS      = 25;       // heads/tails block-alternating training: days per block (T1/H/T2/M phases replay the same block)
static constexpr bool  MT2_FEED_DIRECTION  = true;     // MT2 input from direction-pool slot0 (true) vs composite slot0 (false)
// MT1_SCALE_DOLLARS lives in mt1_scoring.h
static constexpr float MT1_FLOOR_COLD      = 250.f;    // cold-start floor for rolling buffer
static constexpr int   MT1_ROLLING_DAYS    = 10;       // days in per-industry |actual_d| buffer
// Trailing window for the read-only out-of-sample skill scores (Increment A). Long because these
// grade a genuinely unseen model against a constant baseline, so they need to be low-variance to be
// worth reading — and because Increment C would weight the composite by them.
static constexpr int   MT1_SKILL_DAYS      = 250;
// ── Heads/tails split (Part C dual head): TWO 37→28 trunks (market ‖ portfolio) → concat56,
//    + specialized 1-output tails (56→1). Input per industry = 74 = [market37 ‖ portfolio37]. ──
// Head buffer = mkt sub-head (a1..c2, 998) then pf sub-head (a1..c2, 998) = 1996. HD_* offsets
// are relative to each sub-head base (0 for mkt, HEADNN_SUB for pf).
static constexpr int   HEADNN_SUB    = 998;              // one MT1Head trunk (37→28)
static constexpr int   HEADNN_PARAMS = 2 * HEADNN_SUB;   // dual trunk = 1996
static constexpr int   TAILNN_PARAMS = 1803;
// The composed production model is head0 + tail0[4]. load_bin validates by exact element count
// and falls back to random init SILENTLY, so a drift here costs a whole run. Pinned on the Python
// side by tests/test_models.py::TestMT1NN::test_param_count (998 / 1996 / 1803 / 9208).
static_assert(HEADNN_SUB == 998 && HEADNN_PARAMS == 1996 && TAILNN_PARAMS == 1803,
              "MT1 head/tail sizes drifted from models.py");
static_assert(HEADNN_PARAMS + 4 * TAILNN_PARAMS == 9208,
              "composed MT1NN size drifted from models.MT1NN");
static constexpr int HD_A1_W=0,   HD_A1_B=400;   static constexpr int HD_A2_W=420, HD_A2_B=820;   // 20×20, 20×20
static constexpr int HD_B1_W=840, HD_B1_B=900;   static constexpr int HD_B2_W=906, HD_B2_B=930;   // 6×10, 4×6
static constexpr int HD_C1_W=934, HD_C1_B=969;   static constexpr int HD_C2_W=974, HD_C2_B=994;   // 5×7, 4×5  (ends 998)
// Tail buffer = d1..d4 with d1 now 56-wide input (concat of both trunks), d4→1.
static constexpr int TL_D1_W=0,    TL_D1_B=1232; static constexpr int TL_D2_W=1254, TL_D2_B=1606; // 22×56, 16×22
static constexpr int TL_D3_W=1622, TL_D3_B=1782; static constexpr int TL_D4_W=1792, TL_D4_B=1802; // 10×16, 1×10  (ends 1803)

// MT2 injection: fire when ≥75% of pool scores below threshold (worst ~15% of days)
static constexpr float MT2_INJ_THRESHOLD = -7.0f;
static constexpr int   MT2_INJ_MIN_BELOW = (int)(N_SLOTS * 0.75f);  // 150/200
// MT2 pool-consensus diagnostic: look-behind window for the reliability-weighted ensemble
static constexpr int   MT2_LB_DAYS       = 5;

// MT2NN: FC[48→36→36] ‖ LSTM[4→36×2layers] → concat72 → 66→60→54→48
static constexpr int   MT2NN_PARAMS   = 34572;
// FC branch
static constexpr int MT2_FC1_W  =     0;  static constexpr int MT2_FC1_B  =  1728;  // 36×48
static constexpr int MT2_FC2_W  =  1764;  static constexpr int MT2_FC2_B  =  3060;  // +36, 36×36
// LSTM L1 (input_size=4, hidden=36): wih[144×4], whh[144×36], bih[144], bhh[144]
static constexpr int MT2_L1_WIH =  3096;  static constexpr int MT2_L1_WHH =  3672;
static constexpr int MT2_L1_BIH =  8856;  static constexpr int MT2_L1_BHH =  9000;
// LSTM L2 (input_size=36, hidden=36): wih[144×36], whh[144×36], bih[144], bhh[144]
static constexpr int MT2_L2_WIH =  9144;  static constexpr int MT2_L2_WHH = 14328;
static constexpr int MT2_L2_BIH = 19512;  static constexpr int MT2_L2_BHH = 19656;
// Taper (biases follow weights immediately for each layer → kaiming_init works)
static constexpr int MT2_T1_W  = 19800;  static constexpr int MT2_T1_B  = 24552;  // 72×66
static constexpr int MT2_T2_W  = 24618;  static constexpr int MT2_T2_B  = 28578;  // +66, 66×60
static constexpr int MT2_T3_W  = 28638;  static constexpr int MT2_T3_B  = 31878;  // +60, 60×54
static constexpr int MT2_OUT_W = 31932;  static constexpr int MT2_OUT_B = 34524;  // +54, 54×48+48=34572

static bool g_no_save = false;  // --no-save: skip all model writes (diagnostic mode)
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

// MT1 decode helpers (mt1_conf / mt1_conf4 / mt1_range_pct / mt1_delta_t) live in mt1_scoring.h

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
static void master_forward(const float* W, const float* today444, float* out48) {
    float h1[444], h2[444], h3[312], h4[180];
    sgemv_relu(W + MAST_FC1_W, W + MAST_FC1_B, today444, h1, 444, 444);
    sgemv_relu(W + MAST_FC2_W, W + MAST_FC2_B, h1,       h2, 444, 444);
    sgemv_relu(W + MAST_FC3_W, W + MAST_FC3_B, h2,       h3, 312, 444);
    sgemv_relu(W + MAST_FC4_W, W + MAST_FC4_B, h3,       h4, 180, 312);
    sgemv_only(W + MAST_OUT_W, W + MAST_OUT_B, h4,    out48,  48, 180);
}

// MT1NN forward (Part C): dual head 74→56, tails 56→4 (raw logits; activations at score time)

// One 37→28 sub-trunk (mirrors models.py MT1Head). W points at the sub-head's base.
static void head_subforward(const float* W, const float* in37, float* concat28) {
    const float* xb = in37;        // daily  [0:10]
    const float* xc = in37 + 10;   // decade [10:17]
    const float* xa = in37 + 17;   // vol+poly [17:37]
    float a1[20], a2[20], b1[6], b2[4], c1[5], c2[4];
    sgemv_relu(W + HD_A1_W, W + HD_A1_B, xa, a1, 20, 20);
    sgemv_relu(W + HD_A2_W, W + HD_A2_B, a1, a2, 20, 20);
    sgemv_relu(W + HD_B1_W, W + HD_B1_B, xb, b1,  6, 10);
    sgemv_relu(W + HD_B2_W, W + HD_B2_B, b1, b2,  4,  6);
    sgemv_relu(W + HD_C1_W, W + HD_C1_B, xc, c1,  5,  7);
    sgemv_relu(W + HD_C2_W, W + HD_C2_B, c1, c2,  4,  5);
    memcpy(concat28,      a2, 20 * sizeof(float));
    memcpy(concat28 + 20, b2,  4 * sizeof(float));
    memcpy(concat28 + 24, c2,  4 * sizeof(float));
}

// Dual head trunk: in74 = [market37 ‖ portfolio37] → concat56 (mirrors models.py MT1DualHead).
// mkt sub-head at W[0:HEADNN_SUB] over in74[0:37]; pf sub-head at W[HEADNN_SUB:] over in74[37:74].
static void head_forward(const float* W, const float* in74, float* concat56) {
    head_subforward(W,               in74,      concat56);       // market trunk → concat56[0:28]
    head_subforward(W + HEADNN_SUB,  in74 + 37, concat56 + 28);  // portfolio trunk → concat56[28:56]
}

// Specialized 1-output tail: concat56 → 1 raw logit (mirrors models.py MT1Tail).
static void tail_forward(const float* W, const float* concat56, float* out1) {
    float d1[22], d2[16], d3[10];
    sgemv_relu(W + TL_D1_W, W + TL_D1_B, concat56, d1, 22, 56);
    sgemv_relu(W + TL_D2_W, W + TL_D2_B, d1,  d2, 16, 22);
    sgemv_relu(W + TL_D3_W, W + TL_D3_B, d2,  d3, 10, 16);
    sgemv_only(W + TL_D4_W, W + TL_D4_B, d3, out1,  1, 10);
}

// Composed MT1: dual head + 4 tails → 4 raw logits (production/composite path; head_forward once,
// then 4 tail_forwards share the concat56).
static void mt1_composed_forward(const float* head_w, const float* const tail_w[4],
                                 const float* in74, float* out4) {
    float concat56[56];
    head_forward(head_w, in74, concat56);
    for (int t = 0; t < 4; t++) tail_forward(tail_w[t], concat56, out4 + t);
}

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
// in48: 12 industries × 4 MT1 outputs (raw activations, no normalization).
static void mt2_forward(const float* W, const float* in48, float* out48) {
    // FC branch
    float fc1[36], fc2[36];
    sgemv_relu(W + MT2_FC1_W, W + MT2_FC1_B, in48, fc1, 36, 48);
    sgemv_relu(W + MT2_FC2_W, W + MT2_FC2_B, fc1,  fc2, 36, 36);

    // LSTM branch: 12 steps × 4 features, 2 layers, hidden=36
    float h1[36]={}, c1[36]={}, hn1[36], cn1[36];
    float h2[36]={}, c2[36]={}, hn2[36], cn2[36];
    float gates[4*36];
    for (int t = 0; t < 12; t++) {
        const float* x_t = in48 + t * 4;
        lstm_step(W+MT2_L1_WIH, W+MT2_L1_WHH, W+MT2_L1_BIH, W+MT2_L1_BHH,
                  x_t, h1, c1, hn1, cn1, gates, 36, 4);
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

struct MT1Scratch {
    // Deterministic per-day mutation seeds — shared scratch for the head + tail pools.
    uint64_t mut_seeds[HT_MUTS];  // 180 = MT1_COMP_SLOTS - HT_PARENTS

    // ── Heads/tails pools. Head = shared trunk (HEADNN_PARAMS); 4 tails = specialized 1-output
    //    (TAILNN_PARAMS). Production model = composed head0 + tail0[4]. ──
    float*   head_elite;          // [MT1_COMP_PARENTS × HEADNN_PARAMS]
    float*   tail_elite[4];       // 4 × [MT1_COMP_PARENTS × TAILNN_PARAMS]  (dir/acc/rng/cfd)
    float*   head_new;            // selection scratch
    float*   tail_new[4];
    float*   head_mut;            // mutation scratch
    float*   tail_mut;
    float*   head_hist;           // [HIST_DAYS × HIST_PER_DAY × HEADNN_PARAMS]
    float*   tail_hist[4];        // 4 × [HIST_DAYS × HIST_PER_DAY × TAILNN_PARAMS]
    int      head_hist_head{0},  head_hist_count{0};
    int      tail_hist_head[4]{}, tail_hist_count[4]{};
    float*   head0_buf;           // best head (production)
    float*   tail0_buf[4];        // best 4 tails (production)

    // ── Leak-free (out-of-sample) evaluation, Increment A ──
    // head0/tail0 exactly as they stood at the START of the current block, captured before the
    // T1/H/T2 phases see any of the block's days. Every day in the block is therefore unseen by this
    // snapshot, so the activations it emits are an honest out-of-sample read on the deployed model.
    // Everything else logged for MT1 is in-sample by construction: the model that emits day d's
    // activation was selected on a scoring window CONTAINING day d.
    float*   head0_snap;
    float*   tail0_snap[4];

    // Trailing ring of out-of-sample (prediction, target) pairs backing the per-channel skill scores.
    struct SkillEntry { float conf, delta_d, range_pct, conf4, actual_d, acc_floor; };
    SkillEntry   skill_buf[MT1_SKILL_DAYS]{};
    int          skill_head{0}, skill_count{0};

    // ── Direction pool: 200 persistent individuals (v0.5.0.0) ──
    // tail_elite[0] holds only HT_PARENTS models because the other pools regenerate their 180
    // mutations from seeds every day. Persistent identity means every slot's weights must actually
    // exist, so direction gets its own full-width buffer and leaves tail_elite[0] unused.
    float*       dir_pool;                    // [MT1_COMP_SLOTS × TAILNN_PARAMS]
    DirSlotMeta  dir_meta[MT1_COMP_SLOTS]{};
    int          dir_best_slot{0};            // deployed model; ranked in place, never shuffled to slot 0
    uint16_t     dir_next_lineage{0};         // next lineage id to issue
    uint16_t     dir_barred[16]{};            // lineages currently barred from breeding
    // Sizing: at most floor(1/CAP)-1 = 7 lineages can strictly exceed the cap at once, but under
    // hysteresis a barred lineage stays barred down to RESUME, so up to floor(1/RESUME)-1 = 9 can
    // be barred simultaneously. 16 leaves headroom if either threshold is lowered again.
    int          dir_barred_count{0};
    // Retirement-age histogram (buckets 8-15 / 16-31 / 32-63 / 64-127 / 128+) plus a running mean.
    // ~150k retirements over a pass, so a histogram rather than one row per event.
    uint32_t     dir_retire_hist[5]{};
    double       dir_retire_age_sum{0.0};
    uint32_t     dir_retire_n{0};
    int          dir_culled_today{0};

    // Multi-day direction scoring buffer (last MT1_DIR_DAYS entries of features + actual_d)
    struct DirDayEntry { float feat74[74]; float actual_d; float vol_d; };
    DirDayEntry  dir_day_buf[MT1_DIR_DAYS]{};
    int          dir_day_head{0};
    int          dir_day_count{0};
    int          dir_streak{0};    // consecutive collapse days for this industry (freeze floor)
    int          dir_cooldown{0};  // remaining cooldown days after injection

    // Direction constant-collapse injection state (v0.4.2.0). Updated once per block-day in the M
    // phase (process_block), so it lives OUTSIDE MT1DataState — it is never rewound by the per-phase
    // snapshot/restore. deploy_* is a ring of the last MT1_DIR_DAYS deployed (head0+tail0) direction
    // calls and whether each was correct.
    bool         deploy_up[MT1_DIR_DAYS]{};       // deployed dir call that day: true = "up" (conf ≥ 0.5)
    bool         deploy_correct[MT1_DIR_DAYS]{};  // was that call correct vs the matured target
    int          deploy_head{0}, deploy_count{0};
    int          dir_const_streak{0};             // consecutive constant+imperfect checks
    int          dir_inj_cooldown{0};             // checks remaining before another injection is allowed

    // Rolling per-industry circular buffers
    float    rolling_actual  [MT1_ROLLING_DAYS]{};  // |actual_d|           → acc floor
    // Was |actual_d − comp0_delta| feeding the retired band ceiling; now the realized-vol target
    // itself, whose rolling mean gives the vol_floor (and 2×floor = the naive vol predictor).
    float    rolling_vol[MT1_ROLLING_DAYS]{};
    int      rolling_head{0};
    int      rolling_count{0};

    MT1Scratch() {
        size_t hep = (size_t)HT_PARENTS * HEADNN_PARAMS;
        size_t tep = (size_t)HT_PARENTS * TAILNN_PARAMS;
        size_t hhp = (size_t)HIST_DAYS * HIST_PER_DAY * HEADNN_PARAMS;
        size_t thp = (size_t)HIST_DAYS * HIST_PER_DAY * TAILNN_PARAMS;
        head_elite = new float[hep](); head_new = new float[hep](); head_mut = new float[HEADNN_PARAMS]();
        head_hist  = new float[hhp](); head0_buf = new float[HEADNN_PARAMS]();
        tail_mut   = new float[TAILNN_PARAMS]();
        head0_snap = new float[HEADNN_PARAMS]();
        dir_pool   = new float[(size_t)MT1_COMP_SLOTS * TAILNN_PARAMS]();
        for (int p = 0; p < 4; p++) {
            tail_elite[p] = new float[tep](); tail_new[p] = new float[tep]();
            tail_hist[p]  = new float[thp](); tail0_buf[p] = new float[TAILNN_PARAMS]();
            tail0_snap[p] = new float[TAILNN_PARAMS]();
        }
    }
    ~MT1Scratch() {
        delete[] head_elite; delete[] head_new; delete[] head_mut; delete[] head_hist; delete[] head0_buf;
        delete[] tail_mut;   delete[] head0_snap; delete[] dir_pool;
        for (int p = 0; p < 4; p++) { delete[] tail_elite[p]; delete[] tail_new[p]; delete[] tail_hist[p]; delete[] tail0_buf[p]; delete[] tail0_snap[p]; }
    }
    float* head_e(int slot)            { return head_elite    + (size_t)slot * HEADNN_PARAMS; }
    float* tail_e(int p, int slot)     { return tail_elite[p] + (size_t)slot * TAILNN_PARAMS; }
    float* dir_w(int slot)             { return dir_pool      + (size_t)slot * TAILNN_PARAMS; }
    float* head_new_e(int slot)        { return head_new      + (size_t)slot * HEADNN_PARAMS; }
    float* tail_new_e(int p, int slot) { return tail_new[p]   + (size_t)slot * TAILNN_PARAMS; }
    float* head_hist_slot(int d, int pos)         { return head_hist    + ((size_t)(d*HIST_PER_DAY+pos))*HEADNN_PARAMS; }
    float* tail_hist_slot(int c, int d, int pos)  { return tail_hist[c] + ((size_t)(d*HIST_PER_DAY+pos))*TAILNN_PARAMS; }

    void push_dir_day(const float* f74, float ad, float vd) {
        memcpy(dir_day_buf[dir_day_head].feat74, f74, 74 * sizeof(float));
        dir_day_buf[dir_day_head].actual_d = ad;
        dir_day_buf[dir_day_head].vol_d    = vd;
        dir_day_head = (dir_day_head + 1) % MT1_DIR_DAYS;
        if (dir_day_count < MT1_DIR_DAYS) dir_day_count++;
    }
    const DirDayEntry& dir_day(int k) const {
        int oldest = (dir_day_head - dir_day_count + MT1_DIR_DAYS) % MT1_DIR_DAYS;
        return dir_day_buf[(oldest + k) % MT1_DIR_DAYS];
    }
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
    float    lb_in48[MT2_LB_DAYS][48]{};
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

static void init_master_weights(float* W, PCG32& rng) {
    kaiming_init(W + MAST_FC1_W, 444, 444, rng);
    kaiming_init(W + MAST_FC2_W, 444, 444, rng);
    kaiming_init(W + MAST_FC3_W, 312, 444, rng);
    kaiming_init(W + MAST_FC4_W, 180, 312, rng);
    kaiming_init(W + MAST_OUT_W,  48, 180, rng);
}

static void init_head_weights(float* W, PCG32& rng) {
    // Two identical 37→28 sub-trunks (mkt at base 0, pf at base HEADNN_SUB).
    for (int s = 0; s < 2; s++) {
        float* Ws = W + s * HEADNN_SUB;
        kaiming_init(Ws + HD_A1_W, 20, 20, rng);
        kaiming_init(Ws + HD_A2_W, 20, 20, rng);
        kaiming_init(Ws + HD_B1_W,  6, 10, rng);
        kaiming_init(Ws + HD_B2_W,  4,  6, rng);
        kaiming_init(Ws + HD_C1_W,  5,  7, rng);
        kaiming_init(Ws + HD_C2_W,  4,  5, rng);
    }
}

static void init_tail_weights(float* W, PCG32& rng) {
    kaiming_init(W + TL_D1_W, 22, 56, rng);
    kaiming_init(W + TL_D2_W, 16, 22, rng);
    kaiming_init(W + TL_D3_W, 10, 16, rng);
    kaiming_init(W + TL_D4_W,  1, 10, rng);
}

static void init_mt2_weights(float* W, PCG32& rng) {
    kaiming_init(W + MT2_FC1_W, 36, 48, rng);
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
        o.valid  = true;
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
    float baseline, slot0_score, best_delta;
    float top_hold, top_cash;
    int   new_streak;
    float elite_max_val, elite_min_val, elite_mean_val;
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
                                   int ind_i, float* elite_buf);
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
    load_or_init_industry(models_dir, load_dir, ind_i, scratch.elite_buf);
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
        seed_rng.seed((uint64_t)actual_day * 1000007ULL + (uint64_t)ind_i * 13ULL);
        for (int i = 0; i < N_SLOTS - ELITE_POOL; i++)
            mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

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

            // Fill day data (next-day)
            float nd_open = fill_sym[j].valid ? fill_sym[j].open  : day_sym[j].close;
            float nd_low  = fill_sym[j].valid ? fill_sym[j].low   : day_sym[j].low;
            float nd_high = fill_sym[j].valid ? fill_sym[j].high  : day_sym[j].high;

            bool low_first = seq_flags[ind_i * IND_SYMS + j];

            // Partial sell at open
            if (sell_qty > 1e-6f && port.holdings[j] > 1e-6f) {
                float amt = std::min(sell_qty, port.holdings[j]);
                port.holdings[j] -= amt;
                port.cash        += sell_net(amt, nd_open);
                local_sell       += amt;
            }

            // Gap-up sell_all at open
            if (port.holdings[j] > 1e-6f && nd_open >= sell_all_price) {
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, nd_open);
                local_sell       += amt;
            }

            // High-first: intraday sell_all fires before buy
            if (!low_first && port.holdings[j] > 1e-6f &&
                nd_low < sell_all_price && sell_all_price < nd_high) {
                float slipped = sell_all_price * (1.f - SLIPPAGE_RATE);
                float amt = port.holdings[j];
                port.holdings[j] = 0.f;
                port.cash        += sell_net(amt, slipped);
                local_sell       += amt;
            }

            // Stop loss
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
                local_sell       += amt;
            }
        }

        slot_scores[slot] = compute_value_ind(port, day_sym, fill_sym);
        trade_count[slot] = (int)((local_buy + local_sell) > 1e-6f);
        buy_exec  += local_buy;
        sell_exec += local_sell;
    }

    // ── Score history models ─────────────────────────────────────────────────
    Portfolio hist_ports[HIST_DAYS * HIST_PER_DAY] = {};
    float hist_scores[HIST_DAYS * HIST_PER_DAY] = {};
    int n_hist = scratch.hist_count * HIST_PER_DAY;
    for (int h = 0; h < n_hist; h++) {
        Portfolio& port = hist_ports[h];
        port.cash = ref_cash;
        for (int j = 0; j < IND_SYMS; j++) {
            port.holdings[j]    = ref_hold[j];
            port.stop_prices[j] = ref_stop[j];
        }
        const float* W = scratch.hist(h / HIST_PER_DAY, h % HIST_PER_DAY);
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
                float amt = std::min(sell_qty, port.holdings[j]);
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
        hist_scores[h] = compute_value_ind(port, day_sym, fill_sym);
        (void)local_buy; (void)local_sell;
    }

    // ── Score, flags, floor check ────────────────────────────────────────────
    float best_score  = *std::max_element(slot_scores, slot_scores + N_SLOTS);
    float best_delta  = best_score - baseline;
    float worst_delta = *std::min_element(slot_scores, slot_scores + N_SLOTS) - baseline;

    // Elite stats (slots 0..ELITE_COUNT-1 portfolio values)
    float elite_max_val = *std::max_element(slot_scores, slot_scores + ELITE_COUNT);
    float elite_min_val = *std::min_element(slot_scores, slot_scores + ELITE_COUNT);
    float elite_mean_val = 0.f;
    for (int s = 0; s < ELITE_COUNT; s++) elite_mean_val += slot_scores[s];
    elite_mean_val /= ELITE_COUNT;

    log_msg(std::string("[") + IND_SHORT[ind_i] + "] Day " +
            std::to_string(actual_day + 1) + "/" + std::to_string(total_avail) +
            " | best Δ" + (best_delta >= 0 ? "+" : "") + std::to_string((int)best_delta) +
            " worst Δ" + (worst_delta >= 0 ? "+" : "") + std::to_string((int)worst_delta) +
            " | buys=" + std::to_string((int)buy_exec) +
            " sells=" + std::to_string((int)sell_exec) +
            " | prod=$" + std::to_string((int)baseline));

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
        return {baseline, baseline, 0.f, 0.f, 0.f, 0,
                IND_STARTING_CASH, IND_STARTING_CASH, IND_STARTING_CASH};
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
            PCG32 div_rng; div_rng.seed((uint64_t)actual_day * 99991ULL + ind_i);
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

    // Report: top_hold = slot0 holdings value, top_cash = slot0 cash
    float top_hold = 0.f;
    for (int j = 0; j < IND_SYMS; j++) {
        float price = fill_sym[j].valid ? fill_sym[j].close :
                      day_sym[j].valid  ? day_sym[j].close  : 0.f;
        top_hold += slot0_own.holdings[j] * price;
    }

    // Save updated elites and history back to disk
    if (!g_no_save) {
        save_industry_elites(models_dir, ind_i, scratch.elite_buf);
        save_ind_history(models_dir, ind_i, scratch);
    }

    IndResult res;
    res.baseline      = baseline;
    res.slot0_score   = slot_scores[0];
    res.best_delta    = best_delta;
    res.top_hold      = top_hold;
    res.top_cash      = slot0_own.cash;
    res.new_streak    = new_streak;
    res.elite_max_val = elite_max_val;
    res.elite_min_val = elite_min_val;
    res.elite_mean_val= elite_mean_val;
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

static const char* const MT1_POOL_NAMES[4] = {"dir", "acc", "rng", "cfd"};

// MT1ScoreBreakdown + compute_mt1_scores live in mt1_scoring.h

// mt1_win_weight: linear scoring-window weight. Today (newest, di=day_count-1) = 2.0;
// a day MT1_DIR_DAYS-1 steps back = 1.0. Anchored to MT1_DIR_DAYS (absolute age), so a day's
// weight is fixed by how old it is, not by how full the window is during warmup.
static inline float mt1_win_weight(int di, int day_count) {
    if (MT1_DIR_DAYS <= 1) return 2.f;
    int   age = day_count - 1 - di;                       // 0 = today
    float w   = 2.f - (float)age / (float)(MT1_DIR_DAYS - 1);
    return w < 1.f ? 1.f : w;                             // clamp (defensive; age should be < window)
}

// ── Out-of-sample per-channel skill scores (read-only diagnostic, Increment A) ──────────────────
// For each of the four channels: the fraction of a CONSTANT predictor's squared error that the
// out-of-sample model removes.  1 = perfect, 0 = no better than the constant, negative = worse.
// Every channel is expressed in the same unit, so they are directly comparable — this is the
// quantity Increment C would weight the composite by, computed here without acting on it.
//
// Baselines are the trailing mean of the target over the SAME window, so they are leak-free in the
// same sense the predictions are: the snapshot model never saw any of these days.
static void mt1_skill_scores(const MT1Scratch& sc, float out4[4]) {
    out4[0] = out4[1] = out4[2] = out4[3] = 0.f;
    const int n = sc.skill_count;
    if (n < MT1_SKILL_DAYS / 5) return;      // too thin to mean anything yet

    // Pass 1 — baseline constants (trailing means of each channel's target).
    double m_y = 0, m_mag = 0, m_err = 0, m_ideal = 0;
    for (int k = 0; k < n; k++) {
        const auto& e = sc.skill_buf[k];
        float ad   = e.actual_d;
        float mag  = fabsf(e.delta_d);
        float err  = fabsf(fabsf(ad) - mag);
        float effd = fmaxf(mag, e.acc_floor);
        float r    = fmaxf(e.range_pct * effd, 1e-6f);
        m_y     += (ad >= 0.f) ? 1.0 : 0.0;
        m_mag   += fabs((double)ad);
        m_err   += err;
        m_ideal += 1.0 / (1.0 + (double)(err / r) * (err / r));
    }
    m_y /= n; m_mag /= n; m_err /= n; m_ideal /= n;

    // Pass 2 — model SSE vs baseline SSE, per channel.
    double num[4] = {0,0,0,0}, den[4] = {0,0,0,0};
    for (int k = 0; k < n; k++) {
        const auto& e = sc.skill_buf[k];
        float ad   = e.actual_d;
        float mag  = fabsf(e.delta_d);
        float err  = fabsf(fabsf(ad) - mag);
        float effd = fmaxf(mag, e.acc_floor);
        float r    = fmaxf(e.range_pct * effd, 1e-6f);
        double y   = (ad >= 0.f) ? 1.0 : 0.0;
        double ideal = 1.0 / (1.0 + (double)(err / r) * (err / r));
        // 0 direction: Brier vs predicting the trailing base rate
        num[0] += (e.conf - y) * (e.conf - y);            den[0] += (m_y - y) * (m_y - y);
        // 1 magnitude: |delta| vs predicting the trailing mean |target|
        num[1] += (mag - fabs((double)ad)) * (mag - fabs((double)ad));
        den[1] += (m_mag - fabs((double)ad)) * (m_mag - fabs((double)ad));
        // 2 range: does the band track the realised residual better than a constant band?
        num[2] += (r - err) * (r - err);                  den[2] += (m_err - err) * (m_err - err);
        // 3 confidence: does conf4 track the range geometry better than a constant?
        num[3] += (e.conf4 - ideal) * (e.conf4 - ideal);  den[3] += (m_ideal - ideal) * (m_ideal - ideal);
    }
    for (int c = 0; c < 4; c++)
        out4[c] = (den[c] > 1e-12) ? (float)(1.0 - num[c] / den[c]) : 0.f;
}


// ── MT1 component pool step ──────────────────────────────────────────────────────

struct MT1CompResult { float best, slot0, mean, min_v, mean_correct_dbl; };


// ── Heads/tails pool step (Increment 2 part 2) ───────────────────────────────────
// Generic evolutionary selection over ONE head or tail pool. Mirrors step_mt1_component's
// select / mutate / wavg / history machinery exactly, but is parameterized by param count (P)
// and a per-model score callback, so the head pool (composite fitness, variable trunk + frozen
// tails) and the four tail pools (component fitness, frozen trunk + one variable tail) share a
// single implementation. `score_model(const float* W)` iterates the trailing window internally
// and returns the windowed score plus (for the direction tail) correct-count keys and a cull
// flag. NO injection slots — the shared head propagates cross-component learning (what the old
// composite-to-pool cascade did), so parents = 20 (17 elites + 3 wavg) and the freed capacity
// goes into more elite mutations (180 total). Unused until Increment 3 wires the block loop.
struct MT1PoolScore { float score; bool culled; int n_correct; int n_correct_dbl; int today; };

template <typename ScoreFn>
static MT1CompResult step_mt1_pool(
    int P, float* elite_buf, float* new_buf, float* mut_buf,
    float* hist_buf, int& hist_head, int& hist_count,
    uint64_t* mut_seeds, int ind_i, int actual_day, float sigma, int day_count,
    ScoreFn score_model)
{
    const int pool_elite = HT_PARENTS;              // 20 (no injection slots)
    const int pool_slots = MT1_COMP_SLOTS;          // 200
    const int pool_muts  = pool_slots - pool_elite; // 180

    // Deterministic mutation seeds (per day + industry; pool distinguished by its own buffers)
    {
        PCG32 seed_rng;
        seed_rng.seed((uint64_t)actual_day * 987017ULL + (uint64_t)ind_i * 10007ULL + 22222ULL);
        for (int i = 0; i < pool_muts; i++)
            mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    // Weighted children table: concentrate breeding on proven elites. With injection removed the
    // former immigrant children (15) plus the 5 reclaimed parent slots are redistributed here, so
    // every real parent breeds more. Sums to 180 = pool_muts across 20 parents (avg 9).
    static const int kChildren[HT_PARENTS] = {
        16, 13, 13, 12, 12,                     // slot 0 (prod) + top-4 elites = 66
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,     // slots 5-16 direct elites     = 96
        6, 6, 6                                 // slots 17-19 wavg blends       = 18
    };
    int parent_of_mut[MT1_COMP_SLOTS - HT_PARENTS];
    {
        int idx = 0;
        for (int p = 0; p < pool_elite && idx < pool_muts; p++)
            for (int c = 0; c < kChildren[p] && idx < pool_muts; c++)
                parent_of_mut[idx++] = p;
        for (; idx < pool_muts; idx++) parent_of_mut[idx] = idx % pool_elite;
    }

    auto elite = [&](int slot) { return elite_buf + (size_t)slot * P; };
    auto new_e = [&](int slot) { return new_buf  + (size_t)slot * P; };
    auto hist_slot = [&](int pos) { return hist_buf + (size_t)pos * P; };
    auto get_weights = [&](int slot, float* dst) {
        if (slot < pool_elite) {
            memcpy(dst, elite(slot), P * sizeof(float));
        } else {
            int mut_i  = slot - pool_elite;
            int parent = parent_of_mut[mut_i];
            memcpy(dst, elite(parent), P * sizeof(float));
            apply_gaussian(dst, P, sigma, mut_seeds[mut_i]);
        }
    };

    // Score every slot (elites in place, mutations materialized into mut_buf).
    std::vector<float> cat_sc(pool_slots, 0.f);
    std::vector<int>   nc(pool_slots, 0), ncd(pool_slots, 0), tdy(pool_slots, 0);
    int dir_max_correct = 0;
    if (day_count > 0) {
        for (int slot = 0; slot < pool_slots; slot++) {
            const float* wptr;
            if (slot < pool_elite) wptr = elite(slot);
            else { get_weights(slot, mut_buf); wptr = mut_buf; }
            MT1PoolScore r = score_model(wptr);
            cat_sc[slot] = r.culled ? -1e30f : r.score;
            if (!r.culled) {
                nc[slot] = r.n_correct; ncd[slot] = r.n_correct_dbl; tdy[slot] = r.today;
                if (r.n_correct > dir_max_correct) dir_max_correct = r.n_correct;
            }
        }
    }

    // Pool statistics (exclude culled slots)
    float mean_cat = 0.f, best_cat = -1e30f, min_cat = 1e30f;
    int   live_count = 0;
    for (int s = 0; s < pool_slots; s++)
        if (cat_sc[s] > -1e29f) {
            mean_cat += cat_sc[s];
            if (cat_sc[s] > best_cat) best_cat = cat_sc[s];
            if (cat_sc[s] < min_cat)  min_cat  = cat_sc[s];
            live_count++;
        }
    if (live_count > 0) mean_cat /= live_count;
    if (best_cat < -1e29f) best_cat = 0.f;
    if (min_cat  >  9e29f) min_cat  = 0.f;
    float slot0_cat = cat_sc[0] > -1e29f ? cat_sc[0] : 0.f;

    // mean_correct_dbl and the collapse-freeze both belonged to the direction pool, which left
    // this function in v0.5.0.0. Nothing here is direction-specific any more.
    float mean_cdb = 0.f;

    // Score history candidates (no culling — history is the unconditional safety net)
    int n_hist = hist_count * HIST_PER_DAY;
    std::vector<float> hist_sc(HIST_DAYS * HIST_PER_DAY, 0.f);
    std::vector<int>   hist_nc(HIST_DAYS * HIST_PER_DAY, 0), hist_td(HIST_DAYS * HIST_PER_DAY, 0);
    int total_hist = HIST_DAYS * HIST_PER_DAY;
    int oldest     = (hist_head * HIST_PER_DAY - n_hist + total_hist) % total_hist;
    for (int k = 0; k < n_hist; k++) {
        int abs_pos = (oldest + k) % total_hist;
        MT1PoolScore r = score_model(hist_slot(abs_pos));
        hist_sc[k] = r.score; hist_nc[k] = r.n_correct; hist_td[k] = r.today;
    }

    // Build candidate list + elite ORDER (identical selection to step_mt1_component).
    struct Cand { float score; int n_correct; int today; bool is_hist; int idx; };
    std::vector<Cand> cands; cands.reserve(pool_slots + n_hist);
    for (int s = 0; s < pool_slots; s++) cands.push_back({cat_sc[s], nc[s], tdy[s], false, s});
    for (int k = 0; k < n_hist; k++)     cands.push_back({hist_sc[k], hist_nc[k], hist_td[k], true, k});

    std::vector<int> order; order.reserve(ELITE_COUNT);
    std::vector<int> by_score(cands.size());
    for (int i = 0; i < (int)cands.size(); i++) by_score[i] = i;
    std::sort(by_score.begin(), by_score.end(),
              [&](int a, int b){ return cands[a].score > cands[b].score; });

    for (int i : by_score) { if ((int)order.size() >= ELITE_COUNT) break; order.push_back(i); }

    // Assign chosen candidates into new_buf in elite order.
    for (int rank = 0; rank < (int)order.size(); rank++) {
        const Cand& c = cands[order[rank]];
        if (!c.is_hist) get_weights(c.idx, new_e(rank));
        else            memcpy(new_e(rank), hist_slot((oldest + c.idx) % total_hist), P * sizeof(float));
    }
    for (int rank = (int)order.size(); rank < ELITE_COUNT; rank++)
        memcpy(new_e(rank), new_e(0), P * sizeof(float));

    // Wavg blends (slots 17,18,19): equal-weight average of top 5, 10, 15 direct elites.
    static constexpr int wavg_k[3] = {5, 10, 15};
    for (int b = 0; b < WAVG_COUNT; b++) {
        float* dst  = new_e(ELITE_COUNT + b);
        int    k    = wavg_k[b];
        float  inv_k = 1.f / (float)k;
        memset(dst, 0, P * sizeof(float));
        for (int e = 0; e < k; e++)
            for (int p = 0; p < P; p++) dst[p] += new_e(e)[p] * inv_k;
    }

    // Commit all 20 parent slots (17 elites + 3 wavg; no injection slots).
    memcpy(elite_buf, new_buf, (size_t)HT_PARENTS * P * sizeof(float));

    // Save top 7 direct elites + 3 wavg blends to 5-day history.
    for (int k = 0; k < HIST_ELITE; k++)
        memcpy(hist_slot(hist_head * HIST_PER_DAY + k), elite(k), P * sizeof(float));
    for (int k = 0; k < HIST_WAVG; k++)
        memcpy(hist_slot(hist_head * HIST_PER_DAY + HIST_ELITE + k), elite(ELITE_COUNT + k), P * sizeof(float));
    hist_head = (hist_head + 1) % HIST_DAYS;
    if (hist_count < HIST_DAYS) hist_count++;

    return {best_cat, slot0_cat, mean_cat, min_cat, mean_cdb};
}

// ── Direction pool: forward accumulation (v0.5.0.0) ──────────────────────────────────────────────
// One step = one day (or one replayed day, under --dir-reps). Unlike step_mt1_pool this does NOT
// rebuild the pool: all 200 models persist, each predicts today, records the outcome in its own
// rolling 16-bit register, and only the worst MT1_DIR_CULL_PCT of MATURE models are replaced.
// Ordering is (primary, secondary, tertiary) descending — see mt1_dir_record_scores.
struct MT1DirResult {
    float best_p, mean_p, min_p;      // pool primary-score stats
    float mean_s, mean_t;             // pool mean secondary / tertiary
    int   culled, mature, best_slot;
    float max_lineage_share;
    int   distinct_lineages;
};

static MT1DirResult step_mt1_dir_pool(
    MT1Scratch& sc, int ind_i, int actual_day, int rep, float sigma,
    const float* concat56, float actual_d)
{
    MT1DirResult R{};
    const int P = TAILNN_PARAMS, N = MT1_COMP_SLOTS;
    const bool target_up = (actual_d >= 0.f);

    // 1. Every slot predicts today; the outcome shifts into its record. This is the only place a
    //    model's evidence grows — there is no replay, so a model's score IS its own track record.
    for (int s = 0; s < N; s++) {
        float o1; tail_forward(sc.dir_w(s), concat56, &o1);
        bool correct = ((mt1_conf(o1) >= 0.5f) == target_up);
        DirSlotMeta& m = sc.dir_meta[s];
        m.hist = (uint16_t)((m.hist << 1) | (correct ? 1u : 0u));
        if (m.n_pred < 0xFFFFu) m.n_pred++;
    }

    // 2. Rank. Mature = old enough to be culled or to breed.
    std::vector<int> mature;
    mature.reserve(N);
    float sp, ss, st, sum_p = 0.f, sum_s = 0.f, sum_t = 0.f;
    R.best_p = -1.f; R.min_p = 2.f; R.best_slot = 0;
    for (int s = 0; s < N; s++) {
        mt1_dir_record_scores(sc.dir_meta[s].hist, sc.dir_meta[s].n_pred, &sp, &ss, &st);
        sum_p += sp; sum_s += ss; sum_t += st;
        if (sp > R.best_p) R.best_p = sp;
        if (sp < R.min_p)  R.min_p  = sp;
        if (sc.dir_meta[s].n_pred >= (uint16_t)MT1_DIR_MIN_AGE) mature.push_back(s);
    }
    R.mean_p = sum_p / (float)N; R.mean_s = sum_s / (float)N; R.mean_t = sum_t / (float)N;
    R.mature = (int)mature.size();

    std::sort(mature.begin(), mature.end(), [&](int a, int b) {
        return mt1_dir_better(sc.dir_meta[a], sc.dir_meta[b]);
    });

    // Deployed model = best MATURE model; ranked in place, never shuffled into slot 0, because
    // moving weights between slots would have to drag identity with them for no benefit. Falls
    // back to slot 0 only before any model has matured.
    R.best_slot = mature.empty() ? 0 : mature.front();
    sc.dir_best_slot = R.best_slot;

    if (mature.size() < 4) { sc.dir_culled_today = 0; return R; }   // too thin to select on

    // 3. Lineage shares, and the breeding bar. Counting is over all 200 slots, not just mature.
    uint16_t lin_id[MT1_COMP_SLOTS]; int lin_cnt[MT1_COMP_SLOTS]; int n_lin = 0;
    for (int s = 0; s < N; s++) {
        uint16_t L = sc.dir_meta[s].lineage;
        int j = 0; for (; j < n_lin; j++) if (lin_id[j] == L) { lin_cnt[j]++; break; }
        if (j == n_lin) { lin_id[n_lin] = L; lin_cnt[n_lin] = 1; n_lin++; }
    }
    R.distinct_lineages = n_lin;
    R.max_lineage_share = 0.f;
    for (int j = 0; j < n_lin; j++) {
        float share = (float)lin_cnt[j] / (float)N;
        if (share > R.max_lineage_share) R.max_lineage_share = share;
    }
    auto is_barred = [&](uint16_t L) {
        for (int i = 0; i < sc.dir_barred_count; i++) if (sc.dir_barred[i] == L) return true;
        return false;
    };
    for (int j = 0; j < n_lin; j++) {
        float share = (float)lin_cnt[j] / (float)N;
        bool  barred = is_barred(lin_id[j]);
        // Hysteresis: bar above CAP, release only below RESUME. Equal thresholds make a lineage
        // hovering at the cap flip state almost every rep, and a bar that lasts one rep suppresses
        // nothing.
        bool  over = barred ? (share > MT1_DIR_LINEAGE_RESUME) : (share > MT1_DIR_LINEAGE_CAP);
        if (over && !barred && sc.dir_barred_count < (int)(sizeof(sc.dir_barred)/sizeof(sc.dir_barred[0]))) {
            sc.dir_barred[sc.dir_barred_count++] = lin_id[j];
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   direction lineage " +
                    std::to_string(lin_id[j]) + " BARRED from breeding (share " +
                    std::to_string((int)(100.f * share)) + "%)");
        } else if (!over && barred) {
            for (int i = 0; i < sc.dir_barred_count; i++)
                if (sc.dir_barred[i] == lin_id[j]) {
                    sc.dir_barred[i] = sc.dir_barred[--sc.dir_barred_count];
                    break;
                }
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   direction lineage " +
                    std::to_string(lin_id[j]) + " re-enabled (share " +
                    std::to_string((int)(100.f * share)) + "%)");
        }
    }

    // 4. Parents: top MT1_DIR_ELITE_PCT of mature whose lineage is not barred, plus three ephemeral
    //    weighted-average blends of the top 5/10/15. The blends are breeding TEMPLATES, not pool
    //    residents — a synthetic average has no track record, so it could never mature, and giving
    //    it a slot would just park an unscoreable model in the pool.
    std::vector<int> parents;
    int want = (int)(mature.size() * MT1_DIR_ELITE_PCT + 0.5f);
    if (want < 1) want = 1;
    for (size_t i = 0; i < mature.size() && (int)parents.size() < want; i++)
        if (!is_barred(sc.dir_meta[mature[i]].lineage)) parents.push_back(mature[i]);
    if (parents.empty()) parents.push_back(mature.front());   // every lineage barred: breed anyway
    const int n_direct = (int)parents.size();
    const int n_par    = n_direct + 3;                        // + wavg(top5), wavg(top10), wavg(top15)

    // 5. Cull the worst mature models, recording each retirement age.
    int n_cull = (int)(mature.size() * MT1_DIR_CULL_PCT + 0.5f);
    if (n_cull < 1) n_cull = 1;
    if (n_cull > (int)mature.size() - n_direct) n_cull = (int)mature.size() - n_direct;
    if (n_cull < 0) n_cull = 0;
    R.culled = n_cull;
    sc.dir_culled_today = n_cull;

    PCG32 rng;
    rng.seed((uint64_t)actual_day * 987017ULL + (uint64_t)ind_i * 10007ULL +
             (uint64_t)rep * 1300081ULL + 4242ULL);

    for (int k = 0; k < n_cull; k++) {
        int dead = mature[mature.size() - 1 - k];
        uint16_t age = sc.dir_meta[dead].n_pred;
        int b = (age < 16) ? 0 : (age < 32) ? 1 : (age < 64) ? 2 : (age < 128) ? 3 : 4;
        sc.dir_retire_hist[b]++;
        sc.dir_retire_age_sum += (double)age;
        sc.dir_retire_n++;

        // 6. Backfill round-robin across parents — flat, not the weighted kChildren table the other
        //    pools use (which hands slot 0 sixteen of 180 children and drives the monoculture).
        int pi = k % n_par;
        uint16_t child_lineage;
        if (pi < n_direct) {
            memcpy(sc.dir_w(dead), sc.dir_w(parents[pi]), P * sizeof(float));
            child_lineage = sc.dir_meta[parents[pi]].lineage;
        } else {
            int kk = MT1_DIR_WAVG_K[pi - n_direct];
            if (kk > (int)mature.size()) kk = (int)mature.size();
            float* dst = sc.dir_w(dead);
            memset(dst, 0, P * sizeof(float));
            for (int t = 0; t < kk; t++) {
                const float* src = sc.dir_w(mature[t]);
                for (int p = 0; p < P; p++) dst[p] += src[p];
            }
            const float inv = 1.f / (float)kk;
            for (int p = 0; p < P; p++) dst[p] *= inv;
            child_lineage = sc.dir_next_lineage++;   // a blend is a genuinely new genotype
        }
        apply_gaussian(sc.dir_w(dead), P, sigma,
                       ((uint64_t)rng.next() << 32) | rng.next());
        sc.dir_meta[dead].hist    = 0;   // fresh individual: no record, immune until MT1_DIR_MIN_AGE
        sc.dir_meta[dead].n_pred  = 0;
        sc.dir_meta[dead].lineage = child_lineage;
    }
    return R;
}

// Evolve ONE tail pool (comp: 0=dir 1=acc 2=rng 3=cfd) with the head + other 3 tails frozen at
// their production bests. The frozen-head concat and the frozen tails' raw logits are computed
// once per window day (head is stationary through the whole tail phase); each candidate tail
// substitutes only out4[comp]. comp is 1..3 (acc/vol/cfd) — DIRECTION LEFT THIS FUNCTION in
// v0.5.0.0 for step_mt1_dir_pool, taking its class-balanced day weights, two-half selection and
// collapse floor with it. The balanced weighting was the mechanism that let any constant predictor
// tie at the no-skill baseline, which is what the forward-accumulation redesign exists to escape.
static MT1CompResult step_mt1_tail(
    int comp, int ind_i, MT1Scratch& scratch,
    int actual_day, float sigma, float acc_floor, float vol_floor)
{
    const int dc = scratch.dir_day_count;

    // Plain linear recency weights (oldest = 1.0 -> today = 2.0).
    float day_weight[MT1_DIR_DAYS] = {};
    for (int di = 0; di < dc; di++) day_weight[di] = mt1_win_weight(di, dc);

    // Frozen-head concat + frozen tail raw logits per window day.
    float concat[MT1_DIR_DAYS][56];
    float frozen4[MT1_DIR_DAYS][4];
    for (int di = 0; di < dc; di++) {
        const auto& e = scratch.dir_day(di);
        head_forward(scratch.head0_buf, e.feat74, concat[di]);
        for (int c = 0; c < 4; c++) tail_forward(scratch.tail0_buf[c], concat[di], &frozen4[di][c]);
    }

    auto score_model = [&](const float* W) -> MT1PoolScore {
        MT1PoolScore R{0.f, false, 0, 0, 0};
        if (dc == 0) return R;
        for (int di = 0; di < dc; di++) {
            const auto& e = scratch.dir_day(di);
            bool is_today = (di == dc - 1);
            float out4[4] = {frozen4[di][0], frozen4[di][1], frozen4[di][2], frozen4[di][3]};
            float o1; tail_forward(W, concat[di], &o1); out4[comp] = o1;

            // Range-ceiling cull removed: sc_rng is now single-peaked and self-limiting on the wide
            // side (m < 1 → score = m), so a hard cull adds nothing — and its threshold was computed
            // from TODAY's target (see floors()), which leaked the answer into the constraint.

            float ds;
            (void)is_today;
            if (comp == 1) {
                float delta_d = mt1_delta_t(out4[1]) * MT1_SCALE_DOLLARS;
                float err     = fabsf(fabsf(e.actual_d) - fabsf(delta_d));
                ds = acc_floor / (err + acc_floor);
            } else {
                auto sb = compute_mt1_scores(e.actual_d, e.vol_d, out4, acc_floor, vol_floor);
                ds = (comp == 2) ? sb.range : sb.confidence;
            }
            R.score += ds * day_weight[di];
        }
        // Direction flip cull removed. It culled any model whose confidence never crossed 0.5 —
        // i.e. exactly the constant predictor that the class-balanced day_weight above is DESIGNED
        // to score at the no-skill baseline (dir_W/2 = 7.50). Removing the floor left the pool free
        // to sit below it, and it did: ~35% balanced accuracy, systematically inverted, for all 5
        // passes. Class balancing already handles constant predictors correctly — they score exactly
        // 7.50 and simply never win — so no cull is needed to prevent that degeneracy.
        return R;
    };

    return step_mt1_pool(TAILNN_PARAMS,
        scratch.tail_elite[comp], scratch.tail_new[comp], scratch.tail_mut,
        scratch.tail_hist[comp], scratch.tail_hist_head[comp], scratch.tail_hist_count[comp],
        scratch.mut_seeds, ind_i, actual_day, sigma, dc, score_model);
}

// Evolve the head (shared trunk) pool with all 4 tails frozen at their production bests.
// Fitness = windowed composite of head_slot → concat → 4 frozen tails → 4 outputs.
static MT1CompResult step_mt1_head(
    int ind_i, MT1Scratch& scratch,
    int actual_day, float sigma, float acc_floor, float vol_floor)
{
    const int dc = scratch.dir_day_count;
    float day_weight[MT1_DIR_DAYS] = {};
    for (int di = 0; di < dc; di++) day_weight[di] = mt1_win_weight(di, dc);

    auto score_model = [&](const float* W) -> MT1PoolScore {
        MT1PoolScore R{0.f, false, 0, 0, 0};
        if (dc == 0) return R;
        float concat56[56], out4[4];
        for (int di = 0; di < dc; di++) {
            const auto& e = scratch.dir_day(di);
            head_forward(W, e.feat74, concat56);
            for (int c = 0; c < 4; c++) tail_forward(scratch.tail0_buf[c], concat56, &out4[c]);
            R.score += compute_mt1_scores(e.actual_d, e.vol_d, out4, acc_floor, vol_floor).composite
                       * day_weight[di];
        }
        return R;
    };

    return step_mt1_pool(HEADNN_PARAMS,
        scratch.head_elite, scratch.head_new, scratch.head_mut,
        scratch.head_hist, scratch.head_hist_head, scratch.head_hist_count,
        scratch.mut_seeds, ind_i, actual_day, sigma, dc, score_model);
}

// ── MT1 composite blend-pool step ────────────────────────────────────────────────

// Regenerate blend b into dst (deterministic per actual_day, ind_i, b)



// ── MT1 orchestrator ─────────────────────────────────────────────────────────────


// ── MT2 training step (replaces step_master) ────────────────────────────────────

static MasterResult step_mt2(MasterState& state, MT2Scratch& scratch,
                              const float in48[48], const float actual_perf[N_IND],
                              int actual_day, int total_avail,
                              float sigma, bool* injected_out) {
    if (actual_day < MASTER_START_DAY) {
        if (injected_out) *injected_out = false;
        return {0.f, 0.f, 0.f, 0.f, 0.f};
    }

    {
        PCG32 seed_rng; seed_rng.seed((uint64_t)actual_day * 777017ULL + 99999ULL);
        for (int i = 0; i < N_SLOTS - ELITE_POOL; i++)
            scratch.mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    // Compute optimal tiers retroactively
    int opt_tier[N_IND] = {};
    {
        int pos_idx[N_IND]; int n_pos = 0;
        for (int i = 0; i < N_IND; i++)
            if (actual_perf[i] >= 0.f) pos_idx[n_pos++] = i;
        std::sort(pos_idx, pos_idx + n_pos,
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
        mt2_forward(W, in48, out48);

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
            std::sort(positives, positives + n_pos,
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
            std::sort(pos_idx, pos_idx + n_pos, [&perf](int a, int b){ return perf[a] < perf[b]; });
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
            std::sort(idx, idx + N_IND, [&ord](int a, int b){ return ord[a] > ord[b]; });
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
                    mt2_forward(scratch.elite(e), scratch.lb_in48[d], out48b);
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
        for (int i = 0; i < 48; i++) scratch.lb_in48[scratch.lb_head][i] = in48[i];
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
            mt2_forward(W, in48, out48);
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
        PCG32 div_rng; div_rng.seed((uint64_t)actual_day * 55555ULL + 77777ULL);
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
    return {best_pts_v, elite_max_pts, elite_min_pts, elite_mean_pts, ideal_pts,
            consensus_flat_pts, consensus_wtd_pts};
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

static void save_master_elites(const std::string& dir, const float* elite_buf) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        std::string path = elite_path(dir, "master", slot);
        if (!save_bin(path, elite_buf + (size_t)slot * MASTERNN_PARAMS, MASTERNN_PARAMS))
            log_msg("WARNING: could not save " + path);
    }
}

static void load_or_init_industry(const std::string& dir, const std::string& load_dir,
                                   int ind_i, float* elite_buf) {
    PCG32 rng; rng.seed((uint64_t)ind_i * 987654321ULL + 123456789ULL);
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        float* e = elite_buf + (size_t)slot * STOCKNN_PARAMS;
        bool loaded = false;
        if (!load_dir.empty()) {
            std::string p = elite_path(load_dir, g_ind_names[ind_i].c_str(), slot);
            loaded = load_bin(p, e, STOCKNN_PARAMS);
        }
        if (!loaded) {
            std::string p = elite_path(dir, g_ind_names[ind_i].c_str(), slot);
            loaded = load_bin(p, e, STOCKNN_PARAMS);
        }
        if (!loaded) {
            log_msg(std::string("[") + IND_SHORT[ind_i] + "]   Slot " +
                    std::to_string(slot) + ": random init");
            init_stock_weights(e, rng);
        }
    }
}

static void load_or_init_master(const std::string& dir, const std::string& load_dir,
                                 float* elite_buf) {
    PCG32 rng; rng.seed(0xDEADBEEFCAFEBABEULL);
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        float* e = elite_buf + (size_t)slot * MASTERNN_PARAMS;
        bool loaded = false;
        if (!load_dir.empty()) {
            std::string p = elite_path(load_dir, "master", slot);
            loaded = load_bin(p, e, MASTERNN_PARAMS);
        }
        if (!loaded) {
            std::string p = elite_path(dir, "master", slot);
            loaded = load_bin(p, e, MASTERNN_PARAMS);
        }
        if (!loaded) {
            log_msg("[master  ]   Slot " + std::to_string(slot) + ": random init");
            init_master_weights(e, rng);
        }
    }
}

// ── MT1/MT2 persistence ──────────────────────────────────────────────────────────



// ── Heads/tails: per-phase data-state snapshot/restore (Increment 3A) ─────────────
// Block-alternating training replays the same MT1_BLOCK_DAYS days once per phase (T1/H/T2).
// For fitness to be comparable across those replays, every phase must see the identical
// scoring window (dir_day_buf), adaptive floors (rolling buffers), and collapse counters.
// Those are DATA (feature+target+floor state), captured here and restored before each replay.
// The model pools (head/tail elites + their weight histories) are NOT captured — they carry the
// evolution forward across phases; only the data window is rewound.
struct MT1DataState {
    MT1Scratch::DirDayEntry dir_day_buf[MT1_DIR_DAYS];
    int   dir_day_head, dir_day_count, dir_streak, dir_cooldown;
    float rolling_actual[MT1_ROLLING_DAYS], rolling_vol[MT1_ROLLING_DAYS];
    int   rolling_head, rolling_count;
};

static void snapshot_mt1_data(const MT1Scratch& s, MT1DataState& d) {
    memcpy(d.dir_day_buf, s.dir_day_buf, sizeof(d.dir_day_buf));
    d.dir_day_head = s.dir_day_head; d.dir_day_count = s.dir_day_count;
    d.dir_streak   = s.dir_streak;   d.dir_cooldown  = s.dir_cooldown;
    memcpy(d.rolling_actual,   s.rolling_actual,   sizeof(d.rolling_actual));
    memcpy(d.rolling_vol, s.rolling_vol, sizeof(d.rolling_vol));
    d.rolling_head = s.rolling_head; d.rolling_count = s.rolling_count;
}

static void restore_mt1_data(MT1Scratch& s, const MT1DataState& d) {
    memcpy(s.dir_day_buf, d.dir_day_buf, sizeof(d.dir_day_buf));
    s.dir_day_head = d.dir_day_head; s.dir_day_count = d.dir_day_count;
    s.dir_streak   = d.dir_streak;   s.dir_cooldown  = d.dir_cooldown;
    memcpy(s.rolling_actual,   d.rolling_actual,   sizeof(d.rolling_actual));
    memcpy(s.rolling_vol, d.rolling_vol, sizeof(d.rolling_vol));
    s.rolling_head = d.rolling_head; s.rolling_count = d.rolling_count;
}

// ── Heads/tails: model persistence (Increment 3A) ────────────────────────────────
// BREAKING file format for the head/tail redesign. Additive filenames (head_*/tail_*) so it can
// coexist with the old branched files until Increment 3C retires the old save/load path.
//   head:  mt1_{ind}_head_elite_{0..19}.bin, mt1_{ind}_head_hist.bin, mt1_{ind}_head_0.bin
//   tail:  mt1_{ind}_tail_{name}_elite_{0..19}.bin, _hist.bin, _0.bin  (name = dir/acc/rng/cfd)
//   data:  mt1_{ind}_ht_dir.bin (dir_day window + streak/cooldown; rolling rebuilds after load)
static void save_mt1_ht(const std::string& dir, int ind_i, const MT1Scratch& scratch) {
    const char* ind = g_ind_names[ind_i].c_str();
    char path[512];

    // Head pool: HT_PARENTS parent slots + history + production best
    for (int slot = 0; slot < HT_PARENTS; slot++) {
        snprintf(path, sizeof(path), "%s/mt1_%s_head_elite_%d.bin", dir.c_str(), ind, slot);
        if (!save_bin(path, scratch.head_elite + (size_t)slot * HEADNN_PARAMS, HEADNN_PARAMS))
            log_msg(std::string("WARNING: could not save ") + path);
    }
    snprintf(path, sizeof(path), "%s/mt1_%s_head_hist.bin", dir.c_str(), ind);
    if (FILE* f = fopen(path, "wb")) {
        int meta[2] = {scratch.head_hist_head, scratch.head_hist_count};
        fwrite(meta, sizeof(int), 2, f);
        fwrite(scratch.head_hist, sizeof(float),
               (size_t)HIST_DAYS * HIST_PER_DAY * HEADNN_PARAMS, f);
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/mt1_%s_head_0.bin", dir.c_str(), ind);
    save_bin(path, scratch.head0_buf, HEADNN_PARAMS);

    // Tail pools. Direction (c == 0) persists all MT1_COMP_SLOTS individuals plus its metadata
    // sidecar, because every slot is a real model now rather than a seed-regenerated mutation.
    // It has no weight-history ring: persistent identity supersedes it — a model good enough to
    // bring back from history is a model that was never culled.
    for (int c = 0; c < 4; c++) {
        const int n_slots = (c == 0) ? MT1_COMP_SLOTS : HT_PARENTS;
        for (int slot = 0; slot < n_slots; slot++) {
            snprintf(path, sizeof(path), "%s/mt1_%s_tail_%s_elite_%d.bin",
                     dir.c_str(), ind, MT1_POOL_NAMES[c], slot);
            const float* src = (c == 0) ? (scratch.dir_pool + (size_t)slot * TAILNN_PARAMS)
                                        : (scratch.tail_elite[c] + (size_t)slot * TAILNN_PARAMS);
            if (!save_bin(path, src, TAILNN_PARAMS))
                log_msg(std::string("WARNING: could not save ") + path);
        }
        if (c != 0) {
            snprintf(path, sizeof(path), "%s/mt1_%s_tail_%s_hist.bin", dir.c_str(), ind, MT1_POOL_NAMES[c]);
            if (FILE* f = fopen(path, "wb")) {
                int meta[2] = {scratch.tail_hist_head[c], scratch.tail_hist_count[c]};
                fwrite(meta, sizeof(int), 2, f);
                fwrite(scratch.tail_hist[c], sizeof(float),
                       (size_t)HIST_DAYS * HIST_PER_DAY * TAILNN_PARAMS, f);
                fclose(f);
            }
        }
        snprintf(path, sizeof(path), "%s/mt1_%s_tail_%s_0.bin", dir.c_str(), ind, MT1_POOL_NAMES[c]);
        save_bin(path, scratch.tail0_buf[c], TAILNN_PARAMS);
    }

    // Direction metadata sidecar. It has to be a separate file: save_bin/load_bin are raw headerless
    // float arrays validated by exact element count, so appending anything to a weight file makes
    // load_bin reject it and silently fall back to random init for that slot.
    snprintf(path, sizeof(path), "%s/mt1_%s_tail_dir_meta.bin", dir.c_str(), ind);
    if (FILE* f = fopen(path, "wb")) {
        int meta[4] = {DIR_META_VERSION, MT1_COMP_SLOTS,
                       (int)scratch.dir_next_lineage, scratch.dir_best_slot};
        fwrite(meta, sizeof(int), 4, f);
        fwrite(scratch.dir_meta, sizeof(DirSlotMeta), MT1_COMP_SLOTS, f);
        fclose(f);
    }

    // Shared data window (dir_day buffer + streak/cooldown; rolling not persisted — it rebuilds)
    snprintf(path, sizeof(path), "%s/mt1_%s_ht_dir.bin", dir.c_str(), ind);
    if (FILE* f = fopen(path, "wb")) {
        int meta[4] = {scratch.dir_day_head, scratch.dir_day_count,
                       scratch.dir_streak, scratch.dir_cooldown};
        fwrite(meta, sizeof(int), 4, f);
        fwrite(scratch.dir_day_buf, sizeof(MT1Scratch::DirDayEntry), MT1_DIR_DAYS, f);
        fclose(f);
    }
}

static void load_or_init_mt1_ht(const std::string& dir, const std::string& load_dir,
                                 int ind_i, MT1Scratch& scratch) {
    const char* ind = g_ind_names[ind_i].c_str();
    PCG32 rng; rng.seed((uint64_t)(ind_i + 2 * N_IND) * 777777777ULL + 271828182ULL);
    char path[512];

    auto try_load = [&](const char* fmt_suffix, float* dst, int n, auto... args) -> bool {
        for (const std::string* sd : {&load_dir, &dir}) {
            if (sd->empty()) continue;
            snprintf(path, sizeof(path), fmt_suffix, sd->c_str(), ind, args...);
            if (load_bin(path, dst, n)) return true;
        }
        return false;
    };

    // Head pool elites
    for (int slot = 0; slot < HT_PARENTS; slot++) {
        float* e = scratch.head_elite + (size_t)slot * HEADNN_PARAMS;
        if (!try_load("%s/mt1_%s_head_elite_%d.bin", e, HEADNN_PARAMS, slot))
            init_head_weights(e, rng);
    }
    // Head history
    scratch.head_hist_head = 0; scratch.head_hist_count = 0;
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_head_hist.bin", sd->c_str(), ind);
        FILE* f = fopen(path, "rb"); if (!f) continue;
        int meta[2] = {};
        if (fread(meta, sizeof(int), 2, f) == 2) {
            scratch.head_hist_head  = std::max(0, std::min(meta[0], HIST_DAYS - 1));
            scratch.head_hist_count = std::max(0, std::min(meta[1], HIST_DAYS));
        }
        fread(scratch.head_hist, sizeof(float), (size_t)HIST_DAYS * HIST_PER_DAY * HEADNN_PARAMS, f);
        fclose(f); break;
    }
    // Head production best (fall back to elite slot 0 so it is always valid)
    if (!try_load("%s/mt1_%s_head_0.bin", scratch.head0_buf, HEADNN_PARAMS))
        memcpy(scratch.head0_buf, scratch.head_elite, HEADNN_PARAMS * sizeof(float));

    // Tail pools. Direction loads MT1_COMP_SLOTS individuals into dir_pool and skips the history
    // ring entirely; the other three are unchanged.
    for (int c = 0; c < 4; c++) {
        const int n_slots = (c == 0) ? MT1_COMP_SLOTS : HT_PARENTS;
        for (int slot = 0; slot < n_slots; slot++) {
            float* e = (c == 0) ? (scratch.dir_pool + (size_t)slot * TAILNN_PARAMS)
                                : (scratch.tail_elite[c] + (size_t)slot * TAILNN_PARAMS);
            if (!try_load("%s/mt1_%s_tail_%s_elite_%d.bin", e, TAILNN_PARAMS, MT1_POOL_NAMES[c], slot))
                init_tail_weights(e, rng);
        }
        scratch.tail_hist_head[c] = 0; scratch.tail_hist_count[c] = 0;
        if (c != 0) {
            for (const std::string* sd : {&load_dir, &dir}) {
                if (sd->empty()) continue;
                snprintf(path, sizeof(path), "%s/mt1_%s_tail_%s_hist.bin", sd->c_str(), ind, MT1_POOL_NAMES[c]);
                FILE* f = fopen(path, "rb"); if (!f) continue;
                int meta[2] = {};
                if (fread(meta, sizeof(int), 2, f) == 2) {
                    scratch.tail_hist_head[c]  = std::max(0, std::min(meta[0], HIST_DAYS - 1));
                    scratch.tail_hist_count[c] = std::max(0, std::min(meta[1], HIST_DAYS));
                }
                fread(scratch.tail_hist[c], sizeof(float), (size_t)HIST_DAYS * HIST_PER_DAY * TAILNN_PARAMS, f);
                fclose(f); break;
            }
        }
        const float* fallback = (c == 0) ? scratch.dir_pool : scratch.tail_elite[c];
        if (!try_load("%s/mt1_%s_tail_%s_0.bin", scratch.tail0_buf[c], TAILNN_PARAMS, MT1_POOL_NAMES[c]))
            memcpy(scratch.tail0_buf[c], fallback, TAILNN_PARAMS * sizeof(float));
    }

    // Direction metadata sidecar. Absent (or a version we don't know) → every slot starts as a
    // fresh individual with its own lineage, which is exactly the cold-start state. Tolerating a
    // missing file the same way the ht_dir loader does keeps old model directories loadable.
    for (int s = 0; s < MT1_COMP_SLOTS; s++)
        scratch.dir_meta[s] = DirSlotMeta{0, 0, (uint16_t)s, 0};
    scratch.dir_next_lineage = (uint16_t)MT1_COMP_SLOTS;
    scratch.dir_best_slot    = 0;
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_tail_dir_meta.bin", sd->c_str(), ind);
        FILE* f = fopen(path, "rb"); if (!f) continue;
        int meta[4] = {};
        if (fread(meta, sizeof(int), 4, f) == 4 && meta[0] == DIR_META_VERSION &&
            meta[1] == MT1_COMP_SLOTS) {
            if (fread(scratch.dir_meta, sizeof(DirSlotMeta), MT1_COMP_SLOTS, f)
                    == (size_t)MT1_COMP_SLOTS) {
                scratch.dir_next_lineage = (uint16_t)meta[2];
                scratch.dir_best_slot    = std::max(0, std::min(meta[3], MT1_COMP_SLOTS - 1));
            }
        }
        fclose(f); break;
    }

    // Shared data window
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_ht_dir.bin", sd->c_str(), ind);
        FILE* f = fopen(path, "rb"); if (!f) continue;
        int meta[4] = {};
        int meta_read = (int)fread(meta, sizeof(int), 4, f);
        if (meta_read >= 2 &&
            meta[1] >= 0 && meta[1] <= MT1_DIR_DAYS &&
            meta[0] >= 0 && meta[0] < MT1_DIR_DAYS) {
            scratch.dir_day_head  = meta[0];
            scratch.dir_day_count = meta[1];
            if (meta_read >= 4) { scratch.dir_streak = meta[2]; scratch.dir_cooldown = meta[3]; }
            fread(scratch.dir_day_buf, sizeof(MT1Scratch::DirDayEntry), MT1_DIR_DAYS, f);
        }
        fclose(f); break;
    }
}

// ── Heads/tails: one block of alternating training (Increment 3B) ─────────────────
// Runs the T1 / H / T2 phases over one cached block of up to MT1_BLOCK_DAYS days. Features and
// targets are precomputed by the caller (market-derived → identical across phases); only the MT1
// data window (dir_day + rolling floors) is rewound between phases via MT1DataState, while the
// head/tail pools carry evolution forward. Production bests (head0_buf/tail0_buf) update at each
// phase boundary so the next phase freezes the just-trained side. Post-block: the composed
// head0+tail0 IS the production MT1 (feeds MT2 for every day of this block). Returns block stats.
//
// `per_day` (optional, length >= block_len) receives the pool stats for EVERY block day rather than
// just the last: tail stats from the T2 phase, head stats from the H phase. Both phases replay the
// same days indexed by d, so the two line up without extra bookkeeping. Previously tail_res/head_res
// were simply overwritten each day and only the final day survived into the log — 50 points per
// pass, which is why the pool collapse was only visible in aggregate.
static MT1Result run_mt1_block(
    int ind_i, MT1Scratch& scratch, int block_len,
    const float (*in74_block)[74], const float* actual_d_block, const float* vol_d_block,
    const int* actual_day_block,
    float dir_sigma, float acc_sigma, float rng_sigma, float cfd_sigma, float head_sigma,
    MT1Result* per_day = nullptr)
{
    MT1Result res{};
    if (block_len <= 0) return res;

    // tw[] point at the (stable) tail0 buffers; contents change as bests update, addresses do not.
    const float* tw[4] = {scratch.tail0_buf[0], scratch.tail0_buf[1],
                          scratch.tail0_buf[2], scratch.tail0_buf[3]};

    // Both floors are plain trailing means of their own targets — strictly backward-looking, so
    // nothing about today's answer reaches the scale a model is graded against. (The retired band
    // ceiling once took fmaxf(mean, today) and widened exactly when the model erred.)
    auto floors = [&](float& acc_floor, float& vol_floor) {
        acc_floor = MT1_FLOOR_COLD / 2.f;
        vol_floor = MT1_FLOOR_COLD / 2.f;
        if (scratch.rolling_count > 0) {
            float sa = 0.f, sv = 0.f;
            for (int k = 0; k < scratch.rolling_count; k++) { sa += scratch.rolling_actual[k]; sv += scratch.rolling_vol[k]; }
            acc_floor = sa / (float)scratch.rolling_count / 2.f;
            vol_floor = sv / (float)scratch.rolling_count / 2.f;
        }
    };
    auto advance_rolling = [&](const float* in74, float actual_d, float vol_d) {
        (void)in74;
        scratch.rolling_actual[scratch.rolling_head] = fabsf(actual_d);
        scratch.rolling_vol[scratch.rolling_head]    = fabsf(vol_d);
        scratch.rolling_head = (scratch.rolling_head + 1) % MT1_ROLLING_DAYS;
        if (scratch.rolling_count < MT1_ROLLING_DAYS) scratch.rolling_count++;
    };

    MT1DataState snap; snapshot_mt1_data(scratch, snap);
    MT1CompResult tail_res[4]{}, head_res{};
    const float tsig[4] = {dir_sigma, acc_sigma, rng_sigma, cfd_sigma};

    // Copy one day's pool stats into an MT1Result (used for both the block summary and per_day).
    auto set_tails = [](MT1Result& r, const MT1CompResult t[4]) {
        r.best_dir = t[0].best; r.slot0_dir = t[0].slot0; r.mean_dir = t[0].mean; r.min_dir = t[0].min_v; r.mean_dir_cdbl = t[0].mean_correct_dbl;
        r.best_acc = t[1].best; r.slot0_acc = t[1].slot0; r.mean_acc = t[1].mean; r.min_acc = t[1].min_v;
        r.best_rng = t[2].best; r.slot0_rng = t[2].slot0; r.mean_rng = t[2].mean; r.min_rng = t[2].min_v;
        r.best_cfd = t[3].best; r.slot0_cfd = t[3].slot0; r.mean_cfd = t[3].mean; r.min_cfd = t[3].min_v;
    };
    auto set_head = [](MT1Result& r, const MT1CompResult& h) {
        r.best_score = h.best; r.slot0_score = h.slot0; r.mean_score = h.mean; r.min_score = h.min_v;
    };

    // record_per_day: only the FINAL tail pass (T2) writes per-day tail stats, so what is logged is
    // the state the block actually ends on. T1 is an intermediate pass over the same days.
    // Replay phase for the ACC/RNG/CFD tails only. Direction left the replay regime in v0.5.0.0 —
    // it accumulates a forward record instead (see dir_phase below), so c starts at 1.
    auto tail_phase = [&](bool record_per_day) {
        for (int d = 0; d < block_len; d++) {
            scratch.push_dir_day(in74_block[d], actual_d_block[d], vol_d_block[d]);
            if (actual_day_block[d] >= MT1_START_DAY) {
                float af, vf; floors(af, vf);
                for (int c = 1; c < 4; c++)
                    tail_res[c] = step_mt1_tail(c, ind_i, scratch, actual_day_block[d], tsig[c], af, vf);
                if (record_per_day && per_day) set_tails(per_day[d], tail_res);
            }
            advance_rolling(in74_block[d], actual_d_block[d], vol_d_block[d]);
        }
        for (int c = 1; c < 4; c++)
            memcpy(scratch.tail0_buf[c], scratch.tail_e(c, 0), TAILNN_PARAMS * sizeof(float));
    };

    // Direction phase: replay the block g_dir_reps times. Each rep re-walks the same days from the
    // start, so the model INPUTS reset naturally; the models, their 16-bit records, ages and
    // lineages live in MT1Scratch and deliberately carry across reps. Nothing here touches
    // MT1DataState (dir_day_buf / rolling buffers) — those belong to the replay pools.
    //
    // Reps restore evolutionary throughput lost to the gentle 8.3% cull, at the cost of running
    // 20 epochs over 25 days. That is a real overfitting risk; --dir-reps exists so it can be swept
    // and read off the out-of-sample instrument rather than assumed.
    auto dir_phase = [&](bool record_per_day) {
        float concat56[56];
        for (int r = 0; r < g_dir_reps; r++) {
            for (int d = 0; d < block_len; d++) {
                if (actual_day_block[d] < MT1_START_DAY) continue;
                head_forward(scratch.head0_buf, in74_block[d], concat56);
                MT1DirResult dr = step_mt1_dir_pool(scratch, ind_i, actual_day_block[d], r,
                                                    dir_sigma, concat56, actual_d_block[d]);
                if (record_per_day && r == g_dir_reps - 1 && per_day) {
                    per_day[d].best_dir      = dr.best_p;
                    per_day[d].slot0_dir     = dr.best_p;
                    per_day[d].mean_dir      = dr.mean_p;
                    per_day[d].min_dir       = dr.min_p;
                    per_day[d].mean_dir_cdbl = dr.mean_t;
                    per_day[d].dir_mature      = (float)dr.mature / (float)MT1_COMP_SLOTS;
                    per_day[d].dir_culled      = (float)dr.culled;
                    per_day[d].dir_lineage_max = dr.max_lineage_share;
                    per_day[d].dir_lineage_n   = (float)dr.distinct_lineages;
                    per_day[d].dir_mean_sec    = dr.mean_s;
                }
            }
        }
        memcpy(scratch.tail0_buf[0], scratch.dir_w(scratch.dir_best_slot),
               TAILNN_PARAMS * sizeof(float));
    };

    // T1: freeze head0 + tail0 (block-start bests); evolve the 4 tail pools.
    tail_phase(false);
    restore_mt1_data(scratch, snap);

    // H: freeze tail0 (from T1); evolve the head pool.
    for (int d = 0; d < block_len; d++) {
        scratch.push_dir_day(in74_block[d], actual_d_block[d], vol_d_block[d]);
        if (actual_day_block[d] >= MT1_START_DAY) {
            float af, vf; floors(af, vf);
            head_res = step_mt1_head(ind_i, scratch, actual_day_block[d], head_sigma, af, vf);
            if (per_day) set_head(per_day[d], head_res);
        }
        advance_rolling(in74_block[d], actual_d_block[d], vol_d_block[d]);
    }
    memcpy(scratch.head0_buf, scratch.head_e(0), HEADNN_PARAMS * sizeof(float));
    restore_mt1_data(scratch, snap);

    // T2: freeze new head0 + tail0; evolve tails once more (they get the last word on the new head).
    tail_phase(true);
    // No restore after T2 — data-state stays advanced through the block (block end).

    // D: direction pool, g_dir_reps passes over the block against the block's final head.
    dir_phase(true);

    // Log record from the composed production model on the last block day. (Per-day slot-0
    // activations are captured by the caller's MT2 M phase, which already runs the post-block
    // composed model over every block day.)
    float o4[4]; mt1_composed_forward(scratch.head0_buf, tw, in74_block[block_len - 1], o4);
    res.slot0_conf = mt1_conf(o4[0]);          res.slot0_delta_t = mt1_delta_t(o4[1]);
    res.slot0_range_pct = mt1_range_pct(o4[2]); res.slot0_conf4  = mt1_conf4(o4[3]);
    res.dir0_conf = res.slot0_conf; res.dir0_delta_t = res.slot0_delta_t;
    res.dir0_range_pct = res.slot0_range_pct; res.dir0_conf4 = res.slot0_conf4;
    // "composite" stats = head pool (its fitness IS the composite objective); components = tails.
    set_head(res, head_res);
    set_tails(res, tail_res);
    return res;
}

static void save_mt2_elites(const std::string& dir, MT2Scratch& scratch) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        char p[512]; snprintf(p, sizeof(p), "%s/mt2_elite_%d.bin", dir.c_str(), slot);
        if (!save_bin(p, scratch.elite_buf + (size_t)slot * MT2NN_PARAMS, MT2NN_PARAMS))
            log_msg(std::string("WARNING: could not save ") + p);
    }
    // Save MT2 history
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
    PCG32 rng; rng.seed(0xCAFED00DBEEF1234ULL);
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
static constexpr uint32_t MT_LOG_VERSION = 10u;

static bool write_mt_log_header(FILE* f) {
    uint32_t hdr[4] = {MT_LOG_MAGIC, MT_LOG_VERSION, (uint32_t)N_IND, 0u};
    return fwrite(hdr, sizeof(uint32_t), 4, f) == 4;
}

struct MTLogRecord {
    uint32_t pass_num, actual_day;
    // MT1 composite pool stats
    float mt1_best[N_IND],     mt1_slot0[N_IND],     mt1_mean[N_IND],     mt1_min[N_IND];
    // MT1 direction component pool stats
    float mt1_dir_best[N_IND], mt1_dir_slot0[N_IND], mt1_dir_mean[N_IND], mt1_dir_min[N_IND];
    // MT1 range component pool stats
    float mt1_rng_best[N_IND], mt1_rng_slot0[N_IND], mt1_rng_mean[N_IND], mt1_rng_min[N_IND];
    // MT1 accuracy component pool stats
    float mt1_acc_best[N_IND], mt1_acc_slot0[N_IND], mt1_acc_mean[N_IND], mt1_acc_min[N_IND];
    // MT1 confidence (out[3]) component pool stats
    float mt1_cfd_best[N_IND], mt1_cfd_slot0[N_IND], mt1_cfd_mean[N_IND], mt1_cfd_min[N_IND];
    // MT1 direction: mean n_correct_dbl (primary sort key denominator; logged for plotting)
    float mt1_dir_correct_dbl[N_IND];
    // MT2
    float    mt2_best_pts, mt2_slot0_pts, mt2_ideal_pts;
    uint8_t  mt2_injected;
    uint8_t  pad[3];
    uint8_t  mt1_dir_injected[N_IND];
    float    mt1_slot0_act[N_IND][4];   // V7: raw slot0 activations (conf, delta_t, range_pct, conf4) ×12
    float    mt2_consensus_flat_pts;    // V7: pool-consensus allocation score (flat vote)
    float    mt2_consensus_wtd_pts;     // V7: pool-consensus allocation score (look-behind weighted)
    // V7 (log v7): dual-graded deployed slot-0 pts — portfolio (trained objective) vs market (proxy).
    float    mt2_slot0_pts_pf;          // deployed slot0 graded on slot-0 portfolio delta
    float    mt2_slot0_pts_mkt;         // deployed slot0 graded on market forward return (diagnostic)
    // V9: the MT1 target itself (portfolio forward return × MT1_SCALE_DOLLARS), per industry.
    // Pairs with mt1_slot0_act[] to make the deployed model's calls gradable offline.
    float    mt1_actual_d[N_IND];
    // V10: out-of-sample twin of mt1_slot0_act — the same four activations, but from the head0/tail0
    // SNAPSHOT taken at block start, so the model that produced them never saw this day. Pair with
    // mt1_actual_d[] for an honest, leak-free grade; mt1_slot0_act[] is in-sample by construction.
    float    mt1_oos_act[N_IND][4];
    // V10: per-channel skill over the trailing MT1_SKILL_DAYS out-of-sample predictions
    // (dir, acc, rng, cfd) — fraction of a constant predictor's squared error removed.
    float    mt1_skill[N_IND][4];
    // V11: direction-pool lifecycle (v0.5.0.0 forward accumulation).
    // stats: {mature_frac, culled_today, max_lineage_share, distinct_lineages, mean_secondary,
    //         mean_retirement_age}
    float    mt1_dir_stats[N_IND][6];
    // life: cumulative retirement-age histogram, buckets 8-15 / 16-31 / 32-63 / 64-127 / 128+.
    // Cumulative rather than per-day because ~150k models retire over a pass — the trajectory is
    // what matters, and it comes free from logging the running counts.
    float    mt1_dir_life[N_IND][5];
};
static_assert(sizeof(MTLogRecord) == 2212, "MTLogRecord must be 2212 bytes");

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
        fprintf(csv, ",%+10.2f,%+10.2f,%+10.2f",
                res[i].elite_max_val, res[i].elite_min_val, res[i].elite_mean_val);
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
        "          [--workers N] [--master-only] [--preserve-stock-data] [--no-save]\n"
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
static void drift_mt1_floors(MT1Scratch& sc, float* acc_floor, float* vol_floor) {
    *acc_floor     = MT1_FLOOR_COLD / 2.f;
    *vol_floor     = MT1_FLOOR_COLD / 2.f;
    if (sc.rolling_count > 0) {
        float sum_a = 0.f, sum_r = 0.f;
        for (int k = 0; k < sc.rolling_count; k++) { sum_a += sc.rolling_actual[k]; sum_r += sc.rolling_vol[k]; }
        *acc_floor = sum_a / (float)sc.rolling_count / 2.f;
        // Backward-looking only — the today_r term (computed from TODAY's target) leaked the answer
        // into the threshold; dropped here to match run_mt1_block's floors().
        *vol_floor = sum_r / (float)sc.rolling_count / 2.f;
    }
}

// Windowed composite score for one composed MT1 model (head + 4 tails): linear recency-weighted
// sum of per-day composite over the trailing dir_day window (heads/tails production objective).
static float drift_score_mt1_comp(const float* head_w, const float* const tail_w[4],
                                  MT1Scratch& sc, float acc_floor, float vol_floor) {
    float out4[4], total = 0.f;
    for (int di = 0; di < sc.dir_day_count; di++) {
        const auto& e = sc.dir_day(di);
        mt1_composed_forward(head_w, tail_w, e.feat74, out4);
        total += compute_mt1_scores(e.actual_d, e.vol_d, out4, acc_floor, vol_floor).composite
                 * mt1_win_weight(di, sc.dir_day_count);
    }
    return total;
}

// One day of production-style heads/tails upkeep for the drift track (mirror of upkeep.py):
// daily tail phase (freeze head0 + tail0, evolve the 4 tail pools) + a head phase every
// MT1_BLOCK_DAYS days (freeze tails, evolve the head pool). Updates head0/tail0 + rolling.
static void drift_mt1_day(int ind_i, MT1Scratch& sc, float actual_d, float vol_d,
                          const float in74[74],
                          int actual_day, float sigma, bool do_head) {
    sc.push_dir_day(in74, actual_d, vol_d);
    if (actual_day < MT1_START_DAY) return;
    float acc_floor, vol_floor;
    drift_mt1_floors(sc, &acc_floor, &vol_floor);
    for (int c = 0; c < 4; c++)
        step_mt1_tail(c, ind_i, sc, actual_day, sigma, acc_floor, vol_floor);
    for (int c = 0; c < 4; c++)
        memcpy(sc.tail0_buf[c], sc.tail_e(c, 0), TAILNN_PARAMS * sizeof(float));
    if (do_head) {
        step_mt1_head(ind_i, sc, actual_day, sigma, acc_floor, vol_floor);
        memcpy(sc.head0_buf, sc.head_e(0), HEADNN_PARAMS * sizeof(float));
    }
    const float* tw[4] = {sc.tail0_buf[0], sc.tail0_buf[1], sc.tail0_buf[2], sc.tail0_buf[3]};
    sc.rolling_actual[sc.rolling_head]   = fabsf(actual_d);
    sc.rolling_vol[sc.rolling_head] = fabsf(vol_d);
    sc.rolling_head = (sc.rolling_head + 1) % MT1_ROLLING_DAYS;
    if (sc.rolling_count < MT1_ROLLING_DAYS) sc.rolling_count++;
}

// Tier-classification pts for one MT2 model (mirrors the step_mt2 grading at lines ~2843).
static float drift_score_mt2_pts(const float* W, const float in48[48], const float actual_perf[N_IND]) {
    int opt_tier[N_IND] = {0};
    {
        int pos_idx[N_IND]; int n_pos = 0;
        for (int i = 0; i < N_IND; i++) if (actual_perf[i] >= 0.f) pos_idx[n_pos++] = i;
        std::sort(pos_idx, pos_idx + n_pos, [&](int a, int b){ return actual_perf[a] < actual_perf[b]; });
        if (n_pos == 1) { opt_tier[pos_idx[0]] = 3; }
        else if (n_pos == 2) { opt_tier[pos_idx[0]] = 2; opt_tier[pos_idx[1]] = 3; }
        else if (n_pos > 0) {
            int base = n_pos/3, rem = n_pos%3, n1 = base+(rem>=1?1:0), n2 = base+(rem>=2?1:0);
            for (int r = 0; r < n_pos; r++) { int ind = pos_idx[r];
                opt_tier[ind] = (r < n1) ? 1 : (r < n1+n2) ? 2 : 3; }
        }
    }
    float out48[48]; mt2_forward(W, in48, out48);
    int tier[N_IND];
    for (int i = 0; i < N_IND; i++) { const float* lg = out48 + i*4; int b = 0;
        for (int k = 1; k < 4; k++) if (lg[k] > lg[b]) b = k; tier[i] = b; }
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
// mirroring main()'s per-day body. Fills in48/actual_perf for band scoring by the caller.
// Returns false if today has no forward target (last MT1_FWD_DAYS days — not gradable).
static bool drift_advance_day(int run_day_num, int actual_day, int total_days,
                              std::vector<DayData>& all_days,
                              IndustryState* ind_states, MT1Scratch* mt1_scr,
                              MT2Scratch& mt2_scr, MasterState& mst, WorkerScratch& wscr,
                              const std::string& scratch_dir, const std::string& seed_dir,
                              float sigma, float mt2_sigma,
                              float out_in48[48], float out_actual_perf[N_IND]) {
    const DayData* day_ptr  = &all_days[actual_day];
    const DayData* fill_ptr = (actual_day + 1 < total_days) ? &all_days[actual_day + 1] : nullptr;
    const DayData* fwd_ptr  = (actual_day + MT1_FWD_DAYS < total_days) ? &all_days[actual_day + MT1_FWD_DAYS] : nullptr;
    const bool fwd_valid = (fwd_ptr != nullptr);

    bool seq_flags[N_SYMS];
    PCG32 seq_rng; seq_rng.seed((uint64_t)actual_day * 0xABCDEF01234567ULL);
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

    if (fwd_valid) for (int i = 0; i < N_IND; i++) {
        const float* in74 = today888 + i * 74;
        float actual_d = fwd_ret[i] * MT1_SCALE_DOLLARS;   // absolute (matches main loop)
        // The drift study walks day-by-day and has no forward-vol ring, so it feeds |fwd_ret| as a
        // stand-in scale for the vol channel. That makes the vol term a constant-ish baseline here
        // rather than a real target — fine, because the drift track measures COMPOSITE drift of the
        // deployed model against a frozen band, not the vol channel's own skill.
        float vol_d = fabsf(actual_d);
        bool do_head = (run_day_num % MT1_BLOCK_DAYS == 0);
        drift_mt1_day(i, mt1_scr[i], actual_d, vol_d, in74, actual_day, sigma, do_head);
        // MT2 feed from the composed production MT1 (head0 + tail0); one model, so composite ==
        // direction feed (MT2_FEED_DIRECTION moot).
        MT1Scratch& sc = mt1_scr[i];
        const float* tw[4] = {sc.tail0_buf[0], sc.tail0_buf[1], sc.tail0_buf[2], sc.tail0_buf[3]};
        float o4[4]; mt1_composed_forward(sc.head0_buf, tw, in74, o4);
        float conf = mt1_conf(o4[0]);
        out_in48[i*4+0] = conf;
        // Magnitude only. The delta tail is graded signlessly (err = ||actual| - |delta||), so its
        // own sign is never selected on, and folding conf's sign in here randomises the one channel
        // that carries size information — while conf is already passed separately as channel 0.
        // Keep them orthogonal and let MT2 combine them.
        out_in48[i*4+1] = fabsf(mt1_delta_t(o4[1]));
        out_in48[i*4+2] = mt1_range_pct(o4[2]);
        // Channel 3 is UNGRADED as of v0.5.0.0, so its tail is a frozen arbitrary function of
        // the input. Forwarding that to MT2 is worse than forwarding nothing — it is structured
        // noise MT2 can fit. Send a constant instead, so the channel is inert while the tail
        // stays in the model and every file layout is unchanged.
        out_in48[i*4+3] = MT1_UNGRADED_FEED;
    }

    if (fwd_valid && actual_day >= MASTER_START_DAY) {
        bool inj = false;
        step_mt2(mst, mt2_scr, out_in48, out_actual_perf, actual_day, total_days, mt2_sigma, &inj);
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
                         IndustryState* ind_states, MT1Scratch* mt1_scr, MT2Scratch& mt2_scr,
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
        // Reset MT1 rolling/window state IN PLACE (MT1Scratch owns heap buffers — never
        // reassign it: the implicit copy-assign would shallow-copy and double-free).
        MT1Scratch& sc = mt1_scr[i];
        sc.rolling_head = sc.rolling_count = 0;
        memset(sc.rolling_actual,   0, sizeof(sc.rolling_actual));
        memset(sc.rolling_vol, 0, sizeof(sc.rolling_vol));
        memset(sc.dir_day_buf,      0, sizeof(sc.dir_day_buf));
        sc.dir_day_head = sc.dir_day_count = 0;
        sc.dir_streak = sc.dir_cooldown = 0;
        sc.head_hist_head = sc.head_hist_count = 0;
        for (int p = 0; p < 4; p++) { sc.tail_hist_head[p] = 0; sc.tail_hist_count[p] = 0; }
        load_or_init_mt1_ht(scratch_dir, seed_dir, i, sc);
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
    // Heads/tails: a single composed production model feeds MT2, so the MT2_FEED_DIRECTION toggle
    // is moot here (no separate composite/direction feeds).
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
    auto mt1_scr    = std::make_unique<MT1Scratch[]>(N_IND);
    auto mt2_scr    = std::make_unique<MT2Scratch>();
    auto mst        = std::make_unique<MasterState>();
    auto wscr       = std::make_unique<WorkerScratch>();
    // Band = 8 composed MT1 models per industry: head-pool slots 0-7 each combined with the frozen
    // best tails (the head pool IS the composite optimizer). Track = composed head0 + tail0.
    std::vector<float> mt1_band_head((size_t)N_IND * DRIFT_BAND_N * HEADNN_PARAMS);
    std::vector<float> mt1_band_tail((size_t)N_IND * 4 * TAILNN_PARAMS);
    std::vector<float> mt2_band((size_t)DRIFT_BAND_N * MT2NN_PARAMS);

    for (int freeze_day = first_freeze; freeze_day <= last_freeze; freeze_day += DRIFT_WEEK_LEN) {
        int run_start = freeze_day - DRIFT_WARMUP_DAYS;
        drift_reseed(scratch_dir, seed_dir, ind_states.get(), mt1_scr.get(), *mt2_scr,
                     *mst, all_days, run_start);

        int run_day_num = 0;
        float in48[48], actual_perf[N_IND];
        // Warm-up: fill rolling buffers up to (and including) the freeze day.
        for (int d = run_start; d <= freeze_day; d++)
            drift_advance_day(run_day_num++, d, total_days, all_days, ind_states.get(),
                              mt1_scr.get(), *mt2_scr, *mst, *wscr, scratch_dir, seed_dir,
                              sigma, mt2_sigma, in48, actual_perf);

        // Freeze the band: MT1 = head-pool slots 0-7 + the frozen best tails; MT2 = elite slots 0-7.
        for (int i = 0; i < N_IND; i++) {
            MT1Scratch& sc = mt1_scr[i];
            for (int b = 0; b < DRIFT_BAND_N; b++)
                memcpy(&mt1_band_head[((size_t)i * DRIFT_BAND_N + b) * HEADNN_PARAMS],
                       sc.head_e(b), HEADNN_PARAMS * sizeof(float));
            for (int c = 0; c < 4; c++)
                memcpy(&mt1_band_tail[((size_t)i * 4 + c) * TAILNN_PARAMS],
                       sc.tail0_buf[c], TAILNN_PARAMS * sizeof(float));
        }
        for (int b = 0; b < DRIFT_BAND_N; b++)
            memcpy(&mt2_band[(size_t)b * MT2NN_PARAMS], mt2_scr->elite(b), MT2NN_PARAMS * sizeof(float));

        // Belt: roll the track forward DRIFT_BELT_DAYS, scoring weeks 2-5.
        for (int age_day = 0; age_day < DRIFT_BELT_DAYS; age_day++) {
            int actual_day = freeze_day + 1 + age_day;
            bool gradable = drift_advance_day(run_day_num++, actual_day, total_days, all_days,
                                              ind_states.get(), mt1_scr.get(), *mt2_scr, *mst, *wscr,
                                              scratch_dir, seed_dir, sigma, mt2_sigma, in48, actual_perf);
            int age_week = age_day / DRIFT_WEEK_LEN;          // 0..5
            if (age_week < DRIFT_LAG_WEEKS || !gradable) continue;

            DriftLogRecord rec{};
            rec.freeze_day   = (uint32_t)freeze_day;
            rec.calendar_day = (uint32_t)actual_day;
            rec.bucket       = (uint32_t)age_week;

            // MT1 composite: per-industry track (composed head0+tail0) vs 8-model band, same floors.
            for (int i = 0; i < N_IND; i++) {
                MT1Scratch& sc = mt1_scr[i];
                float acc_floor, vol_floor;
                drift_mt1_floors(sc, &acc_floor, &vol_floor);
                const float* btw[4] = {&mt1_band_tail[((size_t)i*4+0)*TAILNN_PARAMS],
                                       &mt1_band_tail[((size_t)i*4+1)*TAILNN_PARAMS],
                                       &mt1_band_tail[((size_t)i*4+2)*TAILNN_PARAMS],
                                       &mt1_band_tail[((size_t)i*4+3)*TAILNN_PARAMS]};
                float bmean = 0.f, bmax = -1e30f, bmin = 1e30f;
                for (int b = 0; b < DRIFT_BAND_N; b++) {
                    float s = drift_score_mt1_comp(&mt1_band_head[((size_t)i*DRIFT_BAND_N + b)*HEADNN_PARAMS],
                                                   btw, sc, acc_floor, vol_floor);
                    bmean += s; bmax = fmaxf(bmax, s); bmin = fminf(bmin, s);
                }
                bmean /= DRIFT_BAND_N;
                const float* ttw[4] = {sc.tail0_buf[0], sc.tail0_buf[1], sc.tail0_buf[2], sc.tail0_buf[3]};
                float track = drift_score_mt1_comp(sc.head0_buf, ttw, sc, acc_floor, vol_floor);
                float floor = fmaxf(1e-3f, 0.05f * fabsf(bmean));   // band-width adaptive floor (§4)
                rec.mt1_rel[i]       = drift_rel_metric(track, bmean, bmax, bmin, floor);
                rec.mt1_band_mean[i] = bmean;
                rec.mt1_track[i]     = track;
            }

            // MT2: track (slot0) vs 8-model band, same in48/actual_perf.
            {
                float bmean = 0.f, bmax = -1e30f, bmin = 1e30f;
                for (int b = 0; b < DRIFT_BAND_N; b++) {
                    float s = drift_score_mt2_pts(&mt2_band[(size_t)b*MT2NN_PARAMS], in48, actual_perf);
                    bmean += s; bmax = fmaxf(bmax, s); bmin = fminf(bmin, s);
                }
                bmean /= DRIFT_BAND_N;
                float track = drift_score_mt2_pts(mt2_scr->elite(0), in48, actual_perf);
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
    float dir_sigma = -1.f, rng_sigma = -1.f, acc_sigma = -1.f, cfd_sigma = -1.f, mt2_sigma_arg = -1.f;
    bool master_only = false, preserve_stock = false;
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
        else if (arg == "--dir-sigma" && a+1<argc) { dir_sigma    = atof(argv[++a]); }
        else if (arg == "--rng-sigma" && a+1<argc) { rng_sigma    = atof(argv[++a]); }
        else if (arg == "--acc-sigma" && a+1<argc) { acc_sigma    = atof(argv[++a]); }
        else if (arg == "--cfd-sigma" && a+1<argc) { cfd_sigma    = atof(argv[++a]); }
        else if (arg == "--mt2-sigma" && a+1<argc) { mt2_sigma_arg= atof(argv[++a]); }
        else if (arg == "--workers"  && a+1<argc) { num_workers=atoi(argv[++a]);}
        else if (arg == "--master-only") master_only = true;
        else if (arg == "--preserve-stock-data") preserve_stock = true;
        else if (arg == "--no-save") g_no_save = true;
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
    if (dir_sigma     < 0.f) dir_sigma     = master_sigma;
    if (rng_sigma     < 0.f) rng_sigma     = master_sigma * (2.f / 3.f);
    if (acc_sigma     < 0.f) acc_sigma     = master_sigma;
    if (cfd_sigma     < 0.f) cfd_sigma     = master_sigma * 0.5f;
    if (mt2_sigma_arg < 0.f) mt2_sigma_arg = master_sigma / 6.f;  // v0.2.6.0: 0.002→0.001 (tighten MT2 pool)

    log_msg(std::string("training_v4_cpp v") + TRAINER_VERSION +
            "  account=" + (account.empty() ? "(diagnostic)" : account));

    if (!load_universe_json("universe.json")) return 1;

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
    auto mt1_scratches = std::make_unique<MT1Scratch[]>(N_IND);   // 12 × ~272 KB ≈ 3.3 MB
    auto mt2_scratch  = std::make_unique<MT2Scratch>();            // ~5.5 MB

    // Open CSV log (goes to log_dir, not output_dir)
    std::string csv_path = log_dir + "/training_log.csv";
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (csv) {
        fprintf(csv, "pass,day");
        for (int i = 0; i < N_IND; i++)
            fprintf(csv, ",%s_elite_max,%s_elite_min,%s_elite_mean",
                    g_ind_names[i].c_str(), g_ind_names[i].c_str(), g_ind_names[i].c_str());
        fprintf(csv, ",mt2_elite_max_pts,mt2_elite_min_pts,mt2_elite_mean_pts,mt2_ideal_pts");
        fprintf(csv, ",mt2_consensus_flat_pts,mt2_consensus_wtd_pts");
        fprintf(csv, ",mt2_slot0_pts_pf,mt2_slot0_pts_mkt");
        for (int i = 0; i < N_IND; i++)
            fprintf(csv, ",%s_mkt_ret,%s_mkt_val",
                    g_ind_names[i].c_str(), g_ind_names[i].c_str());
        fprintf(csv, "\n");
    }

    // Open binary MT log (goes to log_dir, not output_dir)
    std::string mt_log_path = log_dir + "/mt_training_log.bin";
    FILE* mt_log = fopen(mt_log_path.c_str(), "wb");
    if (mt_log) { write_mt_log_header(mt_log); fflush(mt_log); }

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
        float cur_dir_sigma = dir_sigma    * decay;
        float cur_rng_sigma = rng_sigma    * decay;
        float cur_acc_sigma = acc_sigma    * decay;
        float cur_cfd_sigma = cfd_sigma    * decay;
        float cur_mt2_sigma = mt2_sigma_arg* decay;
        log_msg("===== PASS " + std::to_string(pass+1) + "/" + std::to_string(passes) +
                " | sigma=" + std::to_string(cur_sigma).substr(0,6) +
                " | mst=" + std::to_string(cur_mst_sigma).substr(0,6) +
                " | dir=" + std::to_string(cur_dir_sigma).substr(0,6) +
                " | rng=" + std::to_string(cur_rng_sigma).substr(0,6) +
                " | acc=" + std::to_string(cur_acc_sigma).substr(0,6) +
                " | cfd=" + std::to_string(cur_cfd_sigma).substr(0,6) +
                " | mt2=" + std::to_string(cur_mt2_sigma).substr(0,6) + " =====");

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
        for (int i = 0; i < N_IND; i++)
            load_or_init_mt1_ht(output_dir, load_dir, i, mt1_scratches[i]);
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
        // Each fwd-valid day's market-derived features + targets are cached; every MT1_BLOCK_DAYS
        // cached days a block is processed (MT1 T1/H/T2 phases, then MT2's M phase, then flush CSV
        // + MT log + save). StockNN, OHLCV/market histories and non-fwd-day CSV rows stay per-day.
        // blk_888  = dual [market‖portfolio] MT1 features per day
        // blk_fwd  = MT1 target: PORTFOLIO forward return over MT1_FWD_DAYS (Part C; matured via the
        //            pending ring below, since the portfolio forward value isn't known until t+FWD)
        // blk_perf = market forward return (MT2 read-only diagnostic; MT2 trains on same-day slot-0)
        static float blk_888[MT1_BLOCK_DAYS][888];
        static float blk_fwd[MT1_BLOCK_DAYS][N_IND];
        static float blk_vol[MT1_BLOCK_DAYS][N_IND];   // forward MT1_VOL_DAYS realized portfolio vol
        static float blk_perf[MT1_BLOCK_DAYS][N_IND];
        static float blk_mkt_ret_c[MT1_BLOCK_DAYS][N_IND];
        static float blk_mkt_val_c[MT1_BLOCK_DAYS][N_IND];
        static IndResult blk_results[MT1_BLOCK_DAYS][N_IND];
        int blk_actual_day[MT1_BLOCK_DAYS];
        int blk_fill = 0;

        // ── Forward-buffer for the MT1 PORTFOLIO target (Part C) ──
        // The portfolio forward value at t+MT1_FWD_DAYS isn't known until StockNN has simulated
        // that far, so each day's full context is buffered here and only fed into a block once its
        // forward window closes (i.e. MT1_FWD_DAYS days later). MT2 still trains on the same-day
        // slot-0 delta; both stay aligned on the matured day t.
        struct PendingDay {
            int actual_day;
            float feat888[888];
            IndResult results[N_IND];
            float mkt_ret[N_IND], mkt_val[N_IND], mkt_fwd[N_IND], pf_base[N_IND];
        };
        // Ring depth is MT1_VOL_DAYS (>= MT1_FWD_DAYS) so a day matures only once BOTH its
        // forward return (t+FWD) and its forward realized volatility (t+VOL) are known. The
        // return target still uses the t+FWD entry; the vol target walks the whole span.
        static PendingDay pend[MT1_VOL_DAYS + 1];
        int pend_head = 0, pend_count = 0;

        auto process_block = [&](int blk_len) {
            if (blk_len <= 0) return;
            // 1. MT1 block-alternating training per industry (T1/H/T2 over the cached block)
            MT1Result blk_mt1_res[N_IND];
            static MT1Result mt1_day_res[N_IND][MT1_BLOCK_DAYS];   // per-block-day pool stats (V9 log)
            static float     mt1_day_actual[MT1_BLOCK_DAYS][N_IND];
            memset(mt1_day_res, 0, sizeof(mt1_day_res));
            // 0. Freeze the deployed model BEFORE any of this block's days touch it. The T1/H/T2
            //    phases below select on windows that contain these days, so everything they produce
            //    is in-sample; this snapshot is the only leak-free read available.
            for (int i = 0; i < N_IND; i++) {
                MT1Scratch& sc = mt1_scratches[i];
                memcpy(sc.head0_snap, sc.head0_buf, HEADNN_PARAMS * sizeof(float));
                for (int c = 0; c < 4; c++)
                    memcpy(sc.tail0_snap[c], sc.tail0_buf[c], TAILNN_PARAMS * sizeof(float));
            }
            for (int i = 0; i < N_IND; i++) {
                float in74_i[MT1_BLOCK_DAYS][74];
                float actd_i[MT1_BLOCK_DAYS];
                float vold_i[MT1_BLOCK_DAYS];
                int   aday_i[MT1_BLOCK_DAYS];
                for (int d = 0; d < blk_len; d++) {
                    memcpy(in74_i[d], &blk_888[d][i * 74], 74 * sizeof(float));
                    actd_i[d] = blk_fwd[d][i] * MT1_SCALE_DOLLARS;
                    vold_i[d] = blk_vol[d][i];
                    aday_i[d] = blk_actual_day[d];
                    mt1_day_actual[d][i] = actd_i[d];
                }
                blk_mt1_res[i] = run_mt1_block(i, mt1_scratches[i], blk_len, in74_i, actd_i, vold_i, aday_i,
                                               cur_dir_sigma, cur_acc_sigma, cur_rng_sigma,
                                               cur_cfd_sigma, cur_mst_sigma, mt1_day_res[i]);
            }
            // 2. MT2 M phase: replay block days with the post-block composed MT1 (head0+tail0).
            //    MT2 is now GRADED/TRAINED on the deployed slot-0 StockNN portfolio delta
            //    (perf_pf = slot0_score/baseline − 1) — the quantity that actually earns — instead
            //    of the coincident market forward return (blk_perf, kept as a read-only diagnostic).
            MasterResult blk_master_res[MT1_BLOCK_DAYS];
            static float mt1_day_act[MT1_BLOCK_DAYS][N_IND][4];   // per-day deployed slot-0 activations
            static float mt1_oos_day[MT1_BLOCK_DAYS][N_IND][4];   // per-day OOS (block-start snapshot)
            static float mt1_skill_day[MT1_BLOCK_DAYS][N_IND][4]; // per-day OOS skill scores
            bool dir_inject_flag[N_IND] = {};   // set by the constant-collapse detector below
            // Per-day trip record for the collapse detector. dir_inject_flag is block-level (the
            // injection runs once, after this loop); this keeps the DAY the streak tripped, so the
            // log can show injection cadence rather than just "somewhere in this block".
            bool dir_inject_day[MT1_BLOCK_DAYS][N_IND] = {};
            for (int d = 0; d < blk_len; d++) {
                MasterResult mr{}; bool inj = false;
                float s0_pf = 0.f, s0_mkt = 0.f;
                // Deployed (post-block composed) MT1 activations for EVERY block day — these feed the
                // V9 log as well as MT2's in48, so they are computed from MT1_START_DAY, not only from
                // MASTER_START_DAY where MT2 switches on.
                float in48[48];
                for (int i = 0; i < N_IND; i++) {
                    MT1Scratch& sc = mt1_scratches[i];
                    const float* tw[4] = {sc.tail0_buf[0], sc.tail0_buf[1], sc.tail0_buf[2], sc.tail0_buf[3]};
                    float o4[4]; mt1_composed_forward(sc.head0_buf, tw, &blk_888[d][i * 74], o4);
                    float conf = mt1_conf(o4[0]);
                    in48[i*4 + 0] = conf;
                    // Magnitude only — see the matching note in the drift-study in48 build. The
                    // delta tail's sign is ungraded, so multiplying conf's sign in here only
                    // corrupts the size channel; conf travels separately as channel 0.
                    in48[i*4 + 1] = fabsf(mt1_delta_t(o4[1]));
                    in48[i*4 + 2] = mt1_range_pct(o4[2]);
                    // Ungraded channel — constant, see the drift-study in48 build.
                    in48[i*4 + 3] = MT1_UNGRADED_FEED;
                    mt1_day_act[d][i][0] = conf;
                    mt1_day_act[d][i][1] = mt1_delta_t(o4[1]);
                    mt1_day_act[d][i][2] = in48[i*4 + 2];
                    mt1_day_act[d][i][3] = in48[i*4 + 3];

                    // ── Leak-free twin: same day, same features, but through the block-start
                    //    snapshot, which has never seen this day. Diagnostic only — it feeds the
                    //    log and the skill ring, never in48, MT2, or any selection.
                    const float* stw[4] = {sc.tail0_snap[0], sc.tail0_snap[1],
                                           sc.tail0_snap[2], sc.tail0_snap[3]};
                    float oo4[4];
                    mt1_composed_forward(sc.head0_snap, stw, &blk_888[d][i * 74], oo4);
                    float oos_conf  = mt1_conf(oo4[0]);
                    float oos_delta = mt1_delta_t(oo4[1]) * MT1_SCALE_DOLLARS;
                    float oos_rng   = mt1_range_pct(oo4[2]);
                    float oos_cfd   = mt1_conf4(oo4[3]);
                    mt1_oos_day[d][i][0] = oos_conf;
                    mt1_oos_day[d][i][1] = mt1_delta_t(oo4[1]);   // fraction, matching mt1_day_act[1]
                    mt1_oos_day[d][i][2] = oos_rng;
                    mt1_oos_day[d][i][3] = oos_cfd;
                    if (blk_actual_day[d] >= MT1_START_DAY) {
                        // Post-block floors (the rolling buffers hold the whole block by now) rather
                        // than day-d floors. acc_floor only sets the band anchor for the range
                        // channel's skill term, so the approximation is immaterial to a diagnostic.
                        float af_s, rc_s; drift_mt1_floors(sc, &af_s, &rc_s);
                        MT1Scratch::SkillEntry& se = sc.skill_buf[sc.skill_head];
                        se.conf = oos_conf; se.delta_d = oos_delta; se.range_pct = oos_rng;
                        se.conf4 = oos_cfd; se.actual_d = mt1_day_actual[d][i]; se.acc_floor = af_s;
                        sc.skill_head = (sc.skill_head + 1) % MT1_SKILL_DAYS;
                        if (sc.skill_count < MT1_SKILL_DAYS) sc.skill_count++;
                    }
                    mt1_skill_scores(sc, mt1_skill_day[d][i]);

                    // ── Direction constant-collapse detector ──
                    // Push this day's deployed direction call into the ring, then, once the ring holds
                    // a full MT1_DIR_DAYS window, test it: constant = every window day called the same
                    // direction; imperfect = ≥1 window day wrong. MT1_DIR_CONST_TRIP consecutive
                    // constant+imperfect checks flags the industry for injection (below).
                    if (blk_actual_day[d] >= MT1_START_DAY) {
                        bool up      = (conf >= 0.5f);
                        bool correct = (up == (mt1_day_actual[d][i] >= 0.f));
                        sc.deploy_up[sc.deploy_head]      = up;
                        sc.deploy_correct[sc.deploy_head] = correct;
                        sc.deploy_head = (sc.deploy_head + 1) % MT1_DIR_DAYS;
                        if (sc.deploy_count < MT1_DIR_DAYS) sc.deploy_count++;
                        if (sc.dir_inj_cooldown > 0) sc.dir_inj_cooldown--;
                        if (sc.deploy_count == MT1_DIR_DAYS) {
                            bool constant = true, imperfect = false;
                            for (int w = 0; w < MT1_DIR_DAYS; w++) {
                                if (sc.deploy_up[w] != sc.deploy_up[0]) constant = false;
                                if (!sc.deploy_correct[w])              imperfect = true;
                            }
                            sc.dir_const_streak = (constant && imperfect) ? sc.dir_const_streak + 1 : 0;
                            if (sc.dir_const_streak >= MT1_DIR_CONST_TRIP && sc.dir_inj_cooldown == 0) {
                                dir_inject_flag[i]  = true;
                                dir_inject_day[d][i] = true;   // V10: log the DAY it tripped
                                sc.dir_const_streak = 0;
                                sc.dir_inj_cooldown = MT1_DIR_INJ_COOLDOWN;
                            }
                        }
                    }
                }
                if (blk_actual_day[d] >= MASTER_START_DAY) {
                    // Deployed slot-0 portfolio return (training target) + market return (diagnostic).
                    float perf_pf[N_IND], perf_mkt[N_IND];
                    for (int i = 0; i < N_IND; i++) {
                        float bl = blk_results[d][i].baseline;
                        perf_pf[i]  = (bl > 1e-6f) ? (blk_results[d][i].slot0_score / bl - 1.f) : 0.f;
                        perf_mkt[i] = blk_perf[d][i];   // cached market forward return
                    }
                    // Read-only dual grade of the deployed slot-0 (BEFORE step_mt2 reselects the pool):
                    // the same allocation decision scored on both objectives, so the gap is visible.
                    s0_pf  = drift_score_mt2_pts(mt2_scratch->elite(0), in48, perf_pf);
                    s0_mkt = drift_score_mt2_pts(mt2_scratch->elite(0), in48, perf_mkt);
                    mr = step_mt2(*mst, *mt2_scratch, in48, perf_pf,
                                  blk_actual_day[d], total_days, cur_mt2_sigma, &inj);
                }
                mr.slot0_pts_pf  = s0_pf;
                mr.slot0_pts_mkt = s0_mkt;
                blk_master_res[d] = mr;
            }
            // 2b. Direction constant-collapse injection: re-diversify flagged direction pools.
            //     Under forward accumulation this targets the WORST mature individuals rather than
            //     elite ranks — with persistent identity there are no rank slots to overwrite, and
            //     replacing the bottom of the pool is what "re-diversify without discarding the
            //     good models" actually means. Each replacement becomes a fresh individual: no
            //     record, its own lineage, immune until MT1_DIR_MIN_AGE, exactly like a cull birth.
            for (int i = 0; i < N_IND; i++) {
                if (!dir_inject_flag[i]) continue;
                MT1Scratch& sc = mt1_scratches[i];
                PCG32 inj_rng;
                inj_rng.seed((uint64_t)blk_actual_day[blk_len - 1] * 77003ULL + (uint64_t)i * 131ULL + 55555ULL);

                std::vector<int> mature;
                for (int s = 0; s < MT1_COMP_SLOTS; s++)
                    if (sc.dir_meta[s].n_pred >= (uint16_t)MT1_DIR_MIN_AGE) mature.push_back(s);
                if (mature.empty()) continue;
                std::sort(mature.begin(), mature.end(), [&](int a, int b) {
                    return mt1_dir_better(sc.dir_meta[a], sc.dir_meta[b]);
                });
                const float* best = sc.dir_w(sc.dir_best_slot);
                int n_inj = (int)mature.size() / 4;               // worst quartile of mature
                if (n_inj < 1) n_inj = 1;
                for (int k = 0; k < n_inj; k++) {
                    int dead = mature[mature.size() - 1 - k];
                    init_tail_weights(sc.tail_mut, inj_rng);      // fresh random tail into scratch
                    float* dst = sc.dir_w(dead);
                    for (int p = 0; p < TAILNN_PARAMS; p++)
                        dst[p] = MT1_DIR_INJ_BLEND * best[p] + (1.f - MT1_DIR_INJ_BLEND) * sc.tail_mut[p];
                    sc.dir_meta[dead].hist    = 0;
                    sc.dir_meta[dead].n_pred  = 0;
                    sc.dir_meta[dead].lineage = sc.dir_next_lineage++;
                }
                log_msg(std::string("[") + IND_SHORT[i] + "]   MT1 direction constant-collapse — injected " +
                        std::to_string(n_inj) + " blended individuals (½ best + ½ random)");
            }
            // 3. CSV rows for the block (deferred so each row carries its MT2 result)
            if (csv)
                for (int d = 0; d < blk_len; d++)
                    write_csv_row(csv, pass, blk_actual_day[d], blk_results[d],
                                  blk_master_res[d], blk_mkt_ret_c[d], blk_mkt_val_c[d]);
            // 4. One MT log record per BLOCK-DAY (V9). Was one per 25-day block, which gave 50
            //    points per pass and made per-day pool dynamics invisible.
            if (mt_log) {
                for (int d = 0; d < blk_len; d++) {
                    if (blk_actual_day[d] < MT1_START_DAY) continue;
                    MTLogRecord rec{};
                    rec.pass_num   = (uint32_t)pass;
                    rec.actual_day = (uint32_t)blk_actual_day[d];
                    for (int i = 0; i < N_IND; i++) {
                        const MT1Result& m = mt1_day_res[i][d];
                        rec.mt1_best[i]  = m.best_score;
                        rec.mt1_slot0[i] = m.slot0_score;
                        rec.mt1_mean[i]  = m.mean_score;
                        rec.mt1_min[i]   = m.min_score;
                        rec.mt1_dir_best[i]        = m.best_dir;
                        rec.mt1_dir_slot0[i]       = m.slot0_dir;
                        rec.mt1_dir_mean[i]        = m.mean_dir;
                        rec.mt1_dir_min[i]         = m.min_dir;
                        rec.mt1_dir_correct_dbl[i] = m.mean_dir_cdbl;
                        rec.mt1_rng_best[i]  = m.best_rng;
                        rec.mt1_rng_slot0[i] = m.slot0_rng;
                        rec.mt1_rng_mean[i]  = m.mean_rng;
                        rec.mt1_rng_min[i]   = m.min_rng;
                        rec.mt1_acc_best[i]  = m.best_acc;
                        rec.mt1_acc_slot0[i] = m.slot0_acc;
                        rec.mt1_acc_mean[i]  = m.mean_acc;
                        rec.mt1_acc_min[i]   = m.min_acc;
                        rec.mt1_cfd_best[i]  = m.best_cfd;
                        rec.mt1_cfd_slot0[i] = m.slot0_cfd;
                        rec.mt1_cfd_mean[i]  = m.mean_cfd;
                        rec.mt1_cfd_min[i]   = m.min_cfd;
                        // Live again as of V10 — this was hardcoded to 0 behind a stale "retired"
                        // comment while the detector was in fact firing hundreds of times a run,
                        // so injection cadence was invisible to every offline tool.
                        rec.mt1_dir_injected[i] = dir_inject_day[d][i] ? 1u : 0u;
                        rec.mt1_slot0_act[i][0] = mt1_day_act[d][i][0];
                        rec.mt1_slot0_act[i][1] = mt1_day_act[d][i][1];
                        rec.mt1_slot0_act[i][2] = mt1_day_act[d][i][2];
                        rec.mt1_slot0_act[i][3] = mt1_day_act[d][i][3];
                        rec.mt1_actual_d[i]     = mt1_day_actual[d][i];
                        for (int c = 0; c < 4; c++) {
                            rec.mt1_oos_act[i][c] = mt1_oos_day[d][i][c];
                            rec.mt1_skill[i][c]   = mt1_skill_day[d][i][c];
                        }
                        // V11 direction lifecycle: per-day stats from the block result, cumulative
                        // retirement counts straight off the scratch (they are running totals).
                        const MT1Scratch& dsc = mt1_scratches[i];
                        rec.mt1_dir_stats[i][0] = m.dir_mature;
                        rec.mt1_dir_stats[i][1] = m.dir_culled;
                        rec.mt1_dir_stats[i][2] = m.dir_lineage_max;
                        rec.mt1_dir_stats[i][3] = m.dir_lineage_n;
                        rec.mt1_dir_stats[i][4] = m.dir_mean_sec;
                        rec.mt1_dir_stats[i][5] = (dsc.dir_retire_n > 0)
                            ? (float)(dsc.dir_retire_age_sum / (double)dsc.dir_retire_n) : 0.f;
                        for (int b = 0; b < 5; b++)
                            rec.mt1_dir_life[i][b] = (float)dsc.dir_retire_hist[b];
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
            // 5. Save (per block ≈ 25 days)
            if (!g_no_save) {
                for (int i = 0; i < N_IND; i++) save_mt1_ht(output_dir, i, mt1_scratches[i]);
                save_mt2_elites(output_dir, *mt2_scratch);
            }
        };

        for (int day_num = 0; day_num < num_days; day_num++) {
            int actual_day = day_start + day_num;
            const DayData* day_ptr  = &all_days[actual_day];
            const DayData* fill_ptr = (actual_day + 1 < total_days) ? &all_days[actual_day + 1] : nullptr;

            // Generate seq_flags for this day
            seq_rng.seed((uint64_t)actual_day * 0xABCDEF01234567ULL);
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
                float slot0_ret = (results[i].baseline > 1e-6f)
                    ? (results[i].slot0_score / results[i].baseline - 1.f) : 0.f;
                today_pf_val[i] = prev_pf * (1.f + slot0_ret);
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

            // Push today into the pending ring (buffers full day-context until its forward matures).
            {
                PendingDay& pd = pend[(pend_head + pend_count) % (MT1_VOL_DAYS + 1)];
                pd.actual_day = actual_day;
                memcpy(pd.feat888, today888,      sizeof(today888));
                memcpy(pd.results, results,       sizeof(IndResult) * N_IND);
                memcpy(pd.mkt_ret, mkt_ret,       sizeof(mkt_ret));
                memcpy(pd.mkt_val, today_mkt_val, sizeof(today_mkt_val));
                memcpy(pd.mkt_fwd, fwd_ret,       sizeof(fwd_ret));   // market forward (MT2 diagnostic)
                memcpy(pd.pf_base, today_pf_val,  sizeof(today_pf_val));
                pend_count++;
            }
            // A day matures once its t+MT1_VOL_DAYS portfolio value (= today's) is realized, which
            // is also when its t+MT1_FWD_DAYS return became available.
            if (pend_count == MT1_VOL_DAYS + 1) {
                PendingDay& t = pend[pend_head];
                // Entry FWD steps ahead of the maturing day carries the t+FWD portfolio value.
                const PendingDay& fwd_e = pend[(pend_head + MT1_FWD_DAYS) % (MT1_VOL_DAYS + 1)];
                for (int i = 0; i < N_IND; i++) {
                    // Realized vol over t+1..t+VOL from consecutive buffered portfolio values.
                    // sqrt(mean(r^2)) of the DAILY returns, scaled like every other MT1 target.
                    double ss = 0.0; int nr = 0;
                    for (int k = 0; k < MT1_VOL_DAYS; k++) {
                        const PendingDay& a = pend[(pend_head + k)     % (MT1_VOL_DAYS + 1)];
                        float bv = a.pf_base[i];
                        float nv = (k + 1 < MT1_VOL_DAYS)
                                 ? pend[(pend_head + k + 1) % (MT1_VOL_DAYS + 1)].pf_base[i]
                                 : today_pf_val[i];
                        if (bv > 1e-6f) { double r = (double)nv / bv - 1.0; ss += r * r; nr++; }
                    }
                    blk_vol[blk_fill][i] = (nr > 0)
                        ? (float)(sqrt(ss / nr) * MT1_SCALE_DOLLARS) : 0.f;
                }
                pend_head = (pend_head + 1) % (MT1_VOL_DAYS + 1);
                pend_count--;
                int d = blk_fill;
                memcpy(blk_888[d], t.feat888, sizeof(t.feat888));
                for (int i = 0; i < N_IND; i++) {
                    blk_fwd[d][i]  = (t.pf_base[i] != 0.f) ? (fwd_e.pf_base[i] / t.pf_base[i] - 1.f) : 0.f;  // MT1 target: portfolio forward
                    blk_perf[d][i] = t.mkt_fwd[i];                                                           // MT2 diagnostic: market forward
                }
                memcpy(blk_mkt_ret_c[d], t.mkt_ret, sizeof(t.mkt_ret));
                memcpy(blk_mkt_val_c[d], t.mkt_val, sizeof(t.mkt_val));
                memcpy(blk_results[d],   t.results, sizeof(IndResult) * N_IND);
                blk_actual_day[d] = t.actual_day;
                blk_fill++;
                if (blk_fill == MT1_BLOCK_DAYS) { process_block(blk_fill); blk_fill = 0; }
            }
        }
        // Flush the final partial block (matured days), then drain the un-matured tail (the last
        // MT1_FWD_DAYS days whose forward window never closed) as plain CSV rows.
        if (blk_fill > 0) { process_block(blk_fill); blk_fill = 0; }
        while (pend_count > 0) {
            PendingDay& t = pend[pend_head];
            pend_head = (pend_head + 1) % (MT1_FWD_DAYS + 1);
            pend_count--;
            if (csv) write_csv_row(csv, pass, t.actual_day, t.results, MasterResult{}, t.mkt_ret, t.mkt_val);
        }

        // Save MT1/MT2 after each pass (industry elites already saved by step_industry)
        if (!g_no_save) {
            log_msg("Pass " + std::to_string(pass+1) + " complete — saving MT1/MT2 elites");
            for (int i = 0; i < N_IND; i++)
                save_mt1_ht(output_dir, i, mt1_scratches[i]);
            save_mt2_elites(output_dir, *mt2_scratch);
        }
    }

    // Shutdown workers
    wctx.shutdown.store(true);
    wctx.work_ready.release(num_workers);
    for (int w = 0; w < num_workers; w++) wctx.work_done.acquire();
    for (auto& t : workers) t.join();

    if (csv)    fclose(csv);
    if (mt_log) fclose(mt_log);
    log_msg("Training complete.");
    return 0;
}
