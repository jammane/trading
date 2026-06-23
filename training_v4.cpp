// training_v4.cpp — C++ port of training_v4.py
// Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
// Run:   ./build/training_v4_cpp --output models [--load-dir DIR] [--start-day N] [--stop-day N]
//        [--passes N] [--sigma F] [--master-sigma F] [--sigma-decay F] [--workers N]

#define TRAINER_VERSION "0.2.0.12"

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

// MT1: 5 separate pools (4 component + 1 composite blend).
// Component pools (dir/acc/rng/cfd): 23 elites (17 direct + 3 wavg + 3 re-injected)
//   + 207 mutations = 230 slots each.  History: 5 days × 10 models (reuses HIST_* constants).
// Composite pool: 200 blends/day (no mutation); 5-day history (10/day); re-injection into
//   component pools each day (top 3 composite → component slots 20–22).
static constexpr int   MT1_REINJECT        = 3;
static constexpr int   MT1_COMP_ELITE      = ELITE_COUNT + WAVG_COUNT + MT1_REINJECT; // 23
static constexpr int   MT1_COMP_SLOTS      = 230;
static constexpr int   MT1_COMP_MUTS       = MT1_COMP_SLOTS - MT1_COMP_ELITE;         // 207
static constexpr int   MT1_BLEND_SLOTS     = 200;
static constexpr float MT1_RANGE_FLOOR     = 1.f;    // $1 — effectively no floor
static constexpr float MT1_RANGE_CEIL_MULT = 4.f;    // ceiling = 4 × mean(last 10 |actual−delta|)
static constexpr float MT1_DIR_BACKFILL    = 0.65f;  // skip direction pool update when best score < this
static constexpr int   MT1_RANGE_INJECT    = 5;      // top range elites → bottom 5 confidence slots (anti-gaming)
static constexpr int   HIST_WINDOW         = 15;

static constexpr float IND_STARTING_CASH   = 25000.0f;
static constexpr float MST_STARTING_CASH   = 300000.0f;
static constexpr float IND_UNIT_PRICE      = 25000.0f;
static constexpr float MAX_SINGLE_STOCK_PCT = 0.60f;
static constexpr float SEC_FEE_RATE        = 0.0000278f;
static constexpr float FINRA_TAF_PER_SHARE = 0.000166f;
static constexpr float FINRA_TAF_MAX       = 8.30f;
static constexpr float SLIPPAGE_RATE       = 0.001f;

static constexpr int   STOCKNN_PARAMS  = 921625;
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

// MT1NN: 37→37→29→20→12→4, per-industry preprocessor (12 pools)
static constexpr int   MT1NN_PARAMS        = 3412;
static constexpr int   MT1_START_DAY       = 25;
static constexpr float MT1_SCALE_DOLLARS   = 10000.f;  // tanh(out[1]) × scale = dollar P&L prediction
static constexpr float MT1_FLOOR_COLD      = 250.f;    // cold-start floor for rolling buffer
static constexpr int   MT1_ROLLING_DAYS    = 10;       // days in per-industry |actual_d| buffer
// MT1 weight layout offsets (kaiming_init writes weight+bias consecutively per layer)
static constexpr int MT1_FC1_W = 0;       static constexpr int MT1_FC1_B = 1369;   // 37×37
static constexpr int MT1_FC2_W = 1406;    static constexpr int MT1_FC2_B = 2479;   // +37, 37×29
static constexpr int MT1_FC3_W = 2508;    static constexpr int MT1_FC3_B = 3088;   // +29, 29×20
static constexpr int MT1_FC4_W = 3108;    static constexpr int MT1_FC4_B = 3348;   // +20, 20×12
static constexpr int MT1_OUT_W = 3360;    static constexpr int MT1_OUT_B = 3408;   // +12, 12×4+4=3412

// MT2 injection: fire when ≥75% of pool scores below threshold (worst ~15% of days)
static constexpr float MT2_INJ_THRESHOLD = -7.0f;
static constexpr int   MT2_INJ_MIN_BELOW = (int)(N_SLOTS * 0.75f);  // 150/200

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

// ── Layer dimensions ───────────────────────────────────────────────────────────

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
static constexpr int STOCK_TODAY_B = 603175;
static constexpr int STOCK_FLAT1_W = 603475;
static constexpr int STOCK_FLAT1_B = 693475;
static constexpr int STOCK_FLAT2_W = 693775;
static constexpr int STOCK_FLAT2_B = 783775;
static constexpr int STOCK_FC1_W   = 784075;
static constexpr int STOCK_FC1_B   = 855175;
static constexpr int STOCK_FC2_W   = 855412;
static constexpr int STOCK_FC2_B   = 896650;
static constexpr int STOCK_FC3_W   = 896824;
static constexpr int STOCK_FC3_B   = 916138;
static constexpr int STOCK_OUT_W   = 916249;
static constexpr int STOCK_OUT_B   = 921577;

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

// StockNN forward — weights[] is STOCKNN_PARAMS floats in the offset layout above.
// hist15x60 is row-major [15][60], row 0 = oldest.  today208 is [208].
// Output out48 is [48] = reshape of [12][4]; activations applied per column.
static void stock_forward(const float* W, const float* hist15x60,
                          const float* today208, float* out48) {
    float x[300], y[300], cat[420], fc1[237], fc2[174], fc3[111];

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

    // Today: cat(x[190], today[208]) = 398 → 300
    memcpy(cat, x, 190 * sizeof(float));
    memcpy(cat + 190, today208, 208 * sizeof(float));
    sgemv_relu(W + STOCK_TODAY_W, W + STOCK_TODAY_B, cat, x, 300, 398);

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

// MT1NN forward: 37→37→29→20→12→4 (raw logits; activations applied at score time)
static void mt1_forward(const float* W, const float* in37, float* out4) {
    float h1[37], h2[29], h3[20], h4[12];
    sgemv_relu(W + MT1_FC1_W, W + MT1_FC1_B, in37, h1, 37, 37);
    sgemv_relu(W + MT1_FC2_W, W + MT1_FC2_B, h1,   h2, 29, 37);
    sgemv_relu(W + MT1_FC3_W, W + MT1_FC3_B, h2,   h3, 20, 29);
    sgemv_relu(W + MT1_FC4_W, W + MT1_FC4_B, h3,   h4, 12, 20);
    sgemv_only(W + MT1_OUT_W, W + MT1_OUT_B, h4,  out4,  4, 12);
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
    // 4 component pool elite buffers [MT1_COMP_ELITE × MT1NN_PARAMS each]
    // pool: 0=dir  1=acc  2=rng  3=cfd
    float*   comp_elites[4];
    float*   new_elites;           // scratch [MT1_COMP_ELITE × MT1NN_PARAMS]
    float*   mut_buf;              // [MT1NN_PARAMS]
    uint64_t mut_seeds[MT1_COMP_MUTS];  // 207

    // Component pool 5-day histories [HIST_DAYS × HIST_PER_DAY × MT1NN_PARAMS each]
    float*   pool_hist[4];
    int      pool_hist_head[4]{};
    int      pool_hist_count[4]{};

    // Composite blend-pool history [HIST_DAYS × HIST_PER_DAY × MT1NN_PARAMS]
    float*   blend_hist;
    int      blend_hist_head{0};
    int      blend_hist_count{0};

    // Re-injection: top MT1_REINJECT composite models written to component slots 20–22 next day
    float*   reinject_buf;
    int      reinject_count{0};

    // Range→confidence anti-gaming injection: top range elites overwrite bottom confidence slots
    float*   rng_inject_buf;
    int      rng_inject_count{0};

    // Best composite model weights (saved periodically as comp_0.bin)
    float*   comp0_buf;

    // Rolling per-industry circular buffers
    float    rolling_actual  [MT1_ROLLING_DAYS]{};  // |actual_d|           → acc floor
    float    rolling_residual[MT1_ROLLING_DAYS]{};  // |actual_d−comp0_δd| → range ceiling
    int      rolling_head{0};
    int      rolling_count{0};

    MT1Scratch() {
        size_t ep  = (size_t)MT1_COMP_ELITE * MT1NN_PARAMS;
        size_t hp  = (size_t)HIST_DAYS * HIST_PER_DAY * MT1NN_PARAMS;
        for (int p = 0; p < 4; p++) {
            comp_elites[p] = new float[ep]();
            pool_hist[p]   = new float[hp]();
        }
        new_elites   = new float[ep]();
        mut_buf      = new float[MT1NN_PARAMS]();
        blend_hist   = new float[hp]();
        reinject_buf    = new float[(size_t)MT1_REINJECT      * MT1NN_PARAMS]();
        rng_inject_buf  = new float[(size_t)MT1_RANGE_INJECT  * MT1NN_PARAMS]();
        comp0_buf       = new float[MT1NN_PARAMS]();
    }
    ~MT1Scratch() {
        for (int p = 0; p < 4; p++) { delete[] comp_elites[p]; delete[] pool_hist[p]; }
        delete[] new_elites; delete[] mut_buf;
        delete[] blend_hist; delete[] reinject_buf; delete[] rng_inject_buf; delete[] comp0_buf;
    }
    float* comp_elite(int pool, int slot) { return comp_elites[pool] + (size_t)slot * MT1NN_PARAMS; }
    float* new_elite(int slot)            { return new_elites         + (size_t)slot * MT1NN_PARAMS; }
    float* pool_hist_slot(int p, int day, int pos) {
        return pool_hist[p] + ((size_t)(day * HIST_PER_DAY + pos)) * MT1NN_PARAMS;
    }
    float* blend_hist_slot(int day, int pos) {
        return blend_hist + ((size_t)(day * HIST_PER_DAY + pos)) * MT1NN_PARAMS;
    }
    float* reinject(int k)    { return reinject_buf   + (size_t)k * MT1NN_PARAMS; }
    float* rng_inject(int k)  { return rng_inject_buf + (size_t)k * MT1NN_PARAMS; }
};

struct MT1Result {
    float best_score, slot0_score, mean_score, min_score;
    float slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4;  // decoded raw activations for MT2
    // per-component pool stats (direction, range, accuracy, confidence)
    float best_dir, slot0_dir, mean_dir, min_dir;
    float best_rng, slot0_rng, mean_rng, min_rng;
    float best_acc, slot0_acc, mean_acc, min_acc;
    float best_cfd, slot0_cfd, mean_cfd, min_cfd;
};


struct MT2Scratch {
    float*   elite_buf;   // [ELITE_POOL × MT2NN_PARAMS]
    float*   new_elites;  // [ELITE_POOL × MT2NN_PARAMS]
    float*   wavg_buf;    // [3 × MT2NN_PARAMS]
    float*   mut_buf;     // [MT2NN_PARAMS]
    uint64_t mut_seeds[N_SLOTS - ELITE_POOL];

    MT2Scratch() {
        size_t ep = (size_t)ELITE_POOL * MT2NN_PARAMS;
        elite_buf  = new float[ep]();
        new_elites = new float[ep]();
        wavg_buf   = new float[3 * MT2NN_PARAMS]();
        mut_buf    = new float[MT2NN_PARAMS]();
    }
    ~MT2Scratch() { delete[] elite_buf; delete[] new_elites; delete[] wavg_buf; delete[] mut_buf; }
    float* elite(int i)     { return elite_buf  + (size_t)i * MT2NN_PARAMS; }
    float* new_elite(int i) { return new_elites + (size_t)i * MT2NN_PARAMS; }
    float* wavg(int i)      { return wavg_buf   + (size_t)i * MT2NN_PARAMS; }
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
    kaiming_init(W + STOCK_TODAY_W, 300, 398, rng);
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

static void init_mt1_weights(float* W, PCG32& rng) {
    kaiming_init(W + MT1_FC1_W, 37, 37, rng);
    kaiming_init(W + MT1_FC2_W, 29, 37, rng);
    kaiming_init(W + MT1_FC3_W, 20, 29, rng);
    kaiming_init(W + MT1_FC4_W, 12, 20, rng);
    kaiming_init(W + MT1_OUT_W,  4, 12, rng);
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

    // today_arr: [208]
    // per sym: 5 raw + 5 delta + 5 normalized = 15 × 12 = 180
    // + 15 cross-sym delta aggs + 13 state
    float today_arr[208] = {};
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
        int base = j * 15;
        // raw(5)
        for (int k = 0; k < 5; k++) today_arr[base + k]     = raw_t[k];
        // delta(5)
        for (int k = 0; k < 5; k++) today_arr[base + 5 + k] = dlt_t[k];
        // normalized(5)
        today_arr[base + 10] = (raw_t[1] - st.lo15) / rng_15;
        today_arr[base + 11] = raw_t[1] / st.avg_c;
        today_arr[base + 12] = st.volatility;
        today_arr[base + 13] = raw_t[4] / st.avg_v;
        today_arr[base + 14] = (raw_t[0] * raw_t[4]) / st.avg_dv;
    }
    // Cross-sym aggs: for each of 5 delta channels: max, min, mean
    for (int k = 0; k < 5; k++) {
        float mx = today_dl[0][k], mn = today_dl[0][k], sm = 0.f;
        for (int j = 0; j < IND_SYMS; j++) {
            if (today_dl[j][k] > mx) mx = today_dl[j][k];
            if (today_dl[j][k] < mn) mn = today_dl[j][k];
            sm += today_dl[j][k];
        }
        today_arr[180 + k * 3 + 0] = mx;
        today_arr[180 + k * 3 + 1] = mn;
        today_arr[180 + k * 3 + 2] = sm / IND_SYMS;
    }
    // State: [cash, holdings[12]]
    today_arr[195] = ref_cash;
    for (int j = 0; j < IND_SYMS; j++) today_arr[196 + j] = ref_hold[j];

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

// Build 444-feature vector for master from per-industry value history.
// out444 layout per industry (37 features × 12 = 444):
//   [0..17]   18 delta lookbacks at LOOKBACKS days
//   [18..20]  3 poly-2 coefs over 5-day window
//   [21..36]  16 poly-3 coefs over 4 windows (10,30,60,90 days)
static void build_master_features(const float ind_val_hist[][IND_HIST_CAP],
                                   int hist_count, float* out444) {
    static constexpr int LOOKBACKS[18] = {1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,90};
    static constexpr int POLY3_WINS[4] = {10,30,60,90};
    memset(out444, 0, 444 * sizeof(float));

    for (int i = 0; i < N_IND; i++) {
        const float* h = ind_val_hist[i];
        float* feat = out444 + i * 37;

        // 18 delta lookbacks
        for (int lt = 0; lt < 18; lt++) {
            int t  = LOOKBACKS[lt];
            float vt  = hist_at(h, hist_count, t);
            float vt1 = hist_at(h, hist_count, t + 1);
            float denom = fabsf(vt1) > 1e-9f ? vt1 : (vt1 >= 0.f ? 1e-9f : -1e-9f);
            feat[lt] = (vt - vt1) / fabsf(denom);
        }

        // poly-2 over 5-day window
        float win5[5];
        fill_window(h, hist_count, 5, win5);
        polyfit2(win5, 5, feat + 18);

        // poly-3 over 4 windows (10,30,60,90)
        float polywin[90];
        for (int wi = 0; wi < 4; wi++) {
            int W = POLY3_WINS[wi];
            fill_window(h, hist_count, W, polywin);
            polyfit3(polywin, W, feat + 21 + wi * 4);
        }
    }
}

// ── step_master ─────────────────────────────────────────────────────────────────

static MasterResult step_master(MasterState& state, MasterScratch& scratch,
                                const IndResult* ind_results,
                                int actual_day, int total_avail,
                                float sigma) {
    // Gate: must have 30+ days of actual market data before master can train.
    if (actual_day < MASTER_START_DAY) return {0.f, 0.f, 0.f, 0.f, 0.f};

    // Compute actual_perf[N_IND] from slot-0 industry results
    float actual_perf[N_IND] = {};
    for (int i = 0; i < N_IND; i++) {
        if (ind_results[i].baseline > 0.f)
            actual_perf[i] = ind_results[i].slot0_score / ind_results[i].baseline - 1.f;
    }

    // Build 444-feature vector (same for all slots — features independent of portfolio state)
    float today444[444];
    build_master_features(state.ind_val_hist, state.ind_hist_count, today444);

    // Snapshot slot-0's portfolio as baseline
    float ref_cash = state.portfolios[0].cash;
    float ref_hold[N_IND];
    for (int i = 0; i < N_IND; i++) ref_hold[i] = state.portfolios[0].holdings[i];
    float baseline = ref_cash;
    for (int i = 0; i < N_IND; i++) baseline += ref_hold[i] * IND_UNIT_PRICE;

    // Reset all portfolios to slot-0's state; zero_counts for slots 1+ inherit slot 0's counts
    for (int s = 0; s < N_SLOTS; s++) {
        state.portfolios[s].cash = ref_cash;
        for (int i = 0; i < N_IND; i++) state.portfolios[s].holdings[i] = ref_hold[i];
    }
    for (int s = 1; s < N_SLOTS; s++)
        memcpy(state.zero_counts[s], state.zero_counts[0], N_IND * sizeof(int));

    // Pregenerate mutation seeds
    float*    mast_mut_buf   = scratch.mut_buf;
    uint64_t* mast_mut_seeds = scratch.mut_seeds;
    {
        PCG32 seed_rng; seed_rng.seed((uint64_t)actual_day * 777017ULL + 99999ULL);
        for (int i = 0; i < N_SLOTS - ELITE_POOL; i++)
            mast_mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    // Compute optimal tiers retroactively from actual_perf (for points scoring)
    // Negatives → tier 0; positives sorted ascending, split bottom-up into thirds.
    // n_pos==1: opt=3; n_pos==2: lower=2,higher=3; n_pos>=3: bottom-up formula.
    int opt_tier[N_IND] = {};
    {
        int pos_idx[N_IND]; int n_pos_opt = 0;
        for (int i = 0; i < N_IND; i++)
            if (actual_perf[i] >= 0.f) pos_idx[n_pos_opt++] = i;
        std::sort(pos_idx, pos_idx + n_pos_opt,
                  [&actual_perf](int a, int b){ return actual_perf[a] < actual_perf[b]; });
        if (n_pos_opt == 1) {
            opt_tier[pos_idx[0]] = 3;
        } else if (n_pos_opt == 2) {
            opt_tier[pos_idx[0]] = 2;
            opt_tier[pos_idx[1]] = 3;
        } else {
            int base = n_pos_opt / 3, rem = n_pos_opt % 3;
            int n1 = base + (rem >= 1 ? 1 : 0);
            int n2 = base + (rem >= 2 ? 1 : 0);
            for (int rank = 0; rank < n_pos_opt; rank++) {
                int ind = pos_idx[rank];
                if      (rank < n1)        opt_tier[ind] = 1;
                else if (rank < n1 + n2)   opt_tier[ind] = 2;
                else                        opt_tier[ind] = 3;
            }
        }
    }

    float ideal_pts = 0.f;
    for (int i = 0; i < N_IND; i++) ideal_pts += (float)opt_tier[i];

    float pred_scores[N_SLOTS] = {};
    float port_vals[N_SLOTS]   = {};
    int   slot_tiers[N_SLOTS][N_IND] = {};
    float out48[48];

    for (int slot = 0; slot < N_SLOTS; slot++) {
        const float* W;
        if (slot < ELITE_POOL) {
            W = scratch.elite(slot);
        } else {
            int mut_i  = slot - ELITE_POOL;
            int parent = mut_i / MUTATIONS_PER_PARENT;
            memcpy(mast_mut_buf, scratch.elite(parent), MASTERNN_PARAMS * sizeof(float));
            apply_gaussian(mast_mut_buf, MASTERNN_PARAMS, sigma, mast_mut_seeds[mut_i]);
            W = mast_mut_buf;
        }

        master_forward(W, today444, out48);

        // Decode tiers: argmax over each group of 4 logits
        int tier[N_IND];
        for (int i = 0; i < N_IND; i++) {
            const float* lg = out48 + i * 4;
            int best = 0;
            for (int k = 1; k < 4; k++) if (lg[k] > lg[best]) best = k;
            tier[i] = best;
            slot_tiers[slot][i] = best;
        }

        // Update consecutive-zero counts for this slot
        for (int i = 0; i < N_IND; i++) {
            if (tier[i] == 0) state.zero_counts[slot][i]++;
            else              state.zero_counts[slot][i] = 0;
        }

        MasterPortfolio& port = state.portfolios[slot];

        // Liquidate industries with 3+ consecutive tier-0 predictions
        for (int i = 0; i < N_IND; i++) {
            if (state.zero_counts[slot][i] >= 3 && port.holdings[i] > 1e-9f) {
                port.cash        += sell_net(port.holdings[i], IND_UNIT_PRICE);
                port.holdings[i]  = 0.f;
            }
        }

        // Compute tier-based allocation (bottom-up, null-padded for n_pos < 3)
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
                int base = n_pos / 3, rem = n_pos % 3;
                int n1 = base + (rem >= 1 ? 1 : 0);
                int n2 = base + (rem >= 2 ? 1 : 0);

                int assigned[N_IND] = {};
                for (int rank = 0; rank < n_pos; rank++) {
                    int ind = positives[rank];
                    if      (rank < n1)        assigned[ind] = 1;
                    else if (rank < n1 + n2)   assigned[ind] = 2;
                    else                        assigned[ind] = 3;
                }

                float total_w = 0.f;
                for (int k = 0; k < n_pos; k++)
                    total_w += TIER_WEIGHTS[assigned[positives[k]]];

                if (total_w > 0.f)
                    for (int k = 0; k < n_pos; k++) {
                        int ind = positives[k];
                        alloc[ind] = TIER_WEIGHTS[assigned[ind]] / total_w * pool;
                    }
            }
        }

        // Apply allocation: liquidate first, then buy
        for (int i = 0; i < N_IND; i++) {
            float cur_v = port.holdings[i] * IND_UNIT_PRICE;
            float tgt_v = alloc[i];
            if (cur_v > tgt_v + 1e-6f) {
                float units = (cur_v - tgt_v) / IND_UNIT_PRICE;
                port.holdings[i] -= units;
                port.cash        += sell_net(units, IND_UNIT_PRICE);
            }
        }
        for (int i = 0; i < N_IND; i++) {
            float cur_v = port.holdings[i] * IND_UNIT_PRICE;
            float tgt_v = alloc[i];
            if (tgt_v > cur_v + 1e-6f) {
                float diff = std::min(tgt_v - cur_v, port.cash);
                float units = diff / IND_UNIT_PRICE;
                port.holdings[i] += units;
                port.cash        -= units * IND_UNIT_PRICE;
            }
        }

        // Apply daily industry returns
        for (int i = 0; i < N_IND; i++)
            port.holdings[i] *= (1.f + actual_perf[i]);

        // Score via points table against retroactive optimal tiers
        float port_val = port.cash;
        for (int i = 0; i < N_IND; i++) port_val += port.holdings[i] * IND_UNIT_PRICE;
        port_vals[slot] = port_val;
        float pts = 0.f;
        for (int i = 0; i < N_IND; i++) {
            int pred = tier[i], opt = opt_tier[i];
            if (opt == 0) {
                if (pred > 0) pts += -2.f - 0.25f * pred;
            } else if (pred == 0) {
                pts -= (float)opt;
            } else if (pred <= opt) {
                pts += (float)pred;
            } else {
                pts += (float)opt - 0.25f * (float)(pred - opt);
            }
        }
        pred_scores[slot] = pts * 1e9f + port_val;
    }

    // Log tier distribution for slot-0
    int tier_counts[4] = {};
    for (int i = 0; i < N_IND; i++) tier_counts[slot_tiers[0][i]]++;
    int best_slot_idx = (int)(std::max_element(pred_scores, pred_scores + N_SLOTS) - pred_scores);
    float best_score  = pred_scores[best_slot_idx];
    float best_pts_v  = (best_score - port_vals[best_slot_idx]) / 1e9f;
    float slot0_pts_v = (pred_scores[0] - port_vals[0]) / 1e9f;

    auto fmt_pts = [](float v) -> std::string {
        char buf[32]; snprintf(buf, sizeof(buf), "%+.2f", v); return buf;
    };
    log_msg(std::string("[master  ] Day ") + std::to_string(actual_day + 1) +
            "/" + std::to_string(total_avail) +
            " | best_pts=" + fmt_pts(best_pts_v) +
            " slot0_pts=" + fmt_pts(slot0_pts_v) +
            " | t0=" + std::to_string(tier_counts[0]) +
            " t1="   + std::to_string(tier_counts[1]) +
            " t2="   + std::to_string(tier_counts[2]) +
            " t3="   + std::to_string(tier_counts[3]));

    // Floor reset
    if (baseline < MST_STARTING_CASH * 0.9f) {
        for (int s = 0; s < N_SLOTS; s++) {
            state.portfolios[s].cash = MST_STARTING_CASH;
            for (int i = 0; i < N_IND; i++) state.portfolios[s].holdings[i] = 0.f;
        }
        memset(state.zero_counts, 0, sizeof(state.zero_counts));
        return {0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Snapshot slot-0's own portfolio and zero_counts before selection mutates them
    MasterPortfolio slot0_own = state.portfolios[0];
    int slot0_zc[N_IND];
    memcpy(slot0_zc, state.zero_counts[0], N_IND * sizeof(int));

    // Selection gate: run selection unless best score is below -1 pt (injection threshold).
    if (best_pts_v >= -1.f) {
        float mean_ps = 0.f;
        for (int s = 0; s < N_SLOTS; s++) mean_ps += pred_scores[s];
        mean_ps /= N_SLOTS;
        float var_ps = 0.f;
        for (int s = 0; s < N_SLOTS; s++) { float d = pred_scores[s]-mean_ps; var_ps += d*d; }
        float pool_floor = mean_ps - sqrtf(var_ps / N_SLOTS);

        std::vector<std::pair<float,int>> surviving;
        for (int s = 0; s < N_SLOTS; s++)
            if (pred_scores[s] >= pool_floor)
                surviving.push_back({pred_scores[s], s});
        if (surviving.empty()) surviving.push_back({pred_scores[0], 0});

        std::sort(surviving.begin(), surviving.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        int n_top = std::min((int)surviving.size(), ELITE_COUNT);

        int   src_rank[ELITE_COUNT] = {};
        float src_val[ELITE_COUNT]  = {};
        for (int k = 0; k < n_top; k++) { src_rank[k] = surviving[k].second; src_val[k] = surviving[k].first; }
        for (int k = n_top; k < ELITE_COUNT; k++) { src_rank[k] = src_rank[0]; src_val[k] = src_val[0]; }

        float w5_w[5], w10_w[10], w15_w[15];
        int n5 = std::min(n_top,5), n10 = std::min(n_top,10), n15 = std::min(n_top,15);
        normalize_weights(src_val, w5_w, n5);
        normalize_weights(src_val, w10_w, n10);
        normalize_weights(src_val, w15_w, n15);

        // New slot-0's zero_counts come from the best-scoring slot
        memcpy(slot0_zc, state.zero_counts[src_rank[0]], N_IND * sizeof(int));

        MasterPortfolio new_mports[ELITE_POOL];
        const MasterPortfolio* mp5[5], *mp10[10], *mp15[15];
        for (int k=0;k<n5;k++)  mp5[k]  = &state.portfolios[src_rank[k]];
        for (int k=0;k<n10;k++) mp10[k] = &state.portfolios[src_rank[k]];
        for (int k=0;k<n15;k++) mp15[k] = &state.portfolios[src_rank[k]];
        MasterPortfolio wp5={}, wp10={}, wp15={};
        wavg_mst_portfolio(mp5,  w5_w,  n5,  wp5);
        wavg_mst_portfolio(mp10, w10_w, n10, wp10);
        wavg_mst_portfolio(mp15, w15_w, n15, wp15);

        for (int k = 0; k < n_top; k++) {
            int slot = src_rank[k];
            if (slot < ELITE_POOL) {
                memcpy(scratch.new_elite(k), scratch.elite(slot), MASTERNN_PARAMS * sizeof(float));
            } else {
                int mut_i  = slot - ELITE_POOL;
                int parent = mut_i / MUTATIONS_PER_PARENT;
                memcpy(scratch.new_elite(k), scratch.elite(parent), MASTERNN_PARAMS * sizeof(float));
                apply_gaussian(scratch.new_elite(k), MASTERNN_PARAMS, sigma, mast_mut_seeds[mut_i]);
            }
            new_mports[k] = state.portfolios[slot];
        }
        for (int k = n_top; k < ELITE_COUNT; k++) {
            memcpy(scratch.new_elite(k), scratch.new_elite(0), MASTERNN_PARAMS * sizeof(float));
            new_mports[k] = new_mports[0];
        }

        int seq[ELITE_COUNT]; for (int k = 0; k < ELITE_COUNT; k++) seq[k] = k;
        wavg_weights_flat(scratch.new_elites, MASTERNN_PARAMS, seq, w5_w,  n5,  scratch.wavg(0));
        wavg_weights_flat(scratch.new_elites, MASTERNN_PARAMS, seq, w10_w, n10, scratch.wavg(1));
        wavg_weights_flat(scratch.new_elites, MASTERNN_PARAMS, seq, w15_w, n15, scratch.wavg(2));

        memcpy(scratch.new_elite(ELITE_COUNT),     scratch.wavg(0), MASTERNN_PARAMS * sizeof(float));
        memcpy(scratch.new_elite(ELITE_COUNT + 1), scratch.wavg(1), MASTERNN_PARAMS * sizeof(float));
        memcpy(scratch.new_elite(ELITE_COUNT + 2), scratch.wavg(2), MASTERNN_PARAMS * sizeof(float));
        new_mports[ELITE_COUNT]   = wp5;
        new_mports[ELITE_COUNT+1] = wp10;
        new_mports[ELITE_COUNT+2] = wp15;

        for (int k = 0; k < ELITE_POOL; k++) {
            memcpy(scratch.elite(k), scratch.new_elite(k), MASTERNN_PARAMS * sizeof(float));
            state.portfolios[k] = new_mports[k];
        }
        for (int mut_i = 0; mut_i < N_SLOTS - ELITE_POOL; mut_i++)
            state.portfolios[ELITE_POOL + mut_i] = state.portfolios[mut_i / MUTATIONS_PER_PARENT];
    } else {
        int half = ELITE_COUNT / 2;
        log_msg("[master  ] best_pts=" + fmt_pts(best_pts_v) + " < -1 — injecting diversity");
        PCG32 div_rng; div_rng.seed((uint64_t)actual_day * 55555ULL + 77777ULL);
        for (int k = half; k < ELITE_COUNT; k++) {
            init_master_weights(scratch.mut_buf, div_rng);
            for (int p = 0; p < MASTERNN_PARAMS; p++)
                scratch.elite(k)[p] = 0.5f * scratch.elite(k-half)[p] + 0.5f * scratch.mut_buf[p];
            state.portfolios[k] = state.portfolios[k - half];
        }
    }

    // Restore slot-0's own portfolio and propagate its zero_counts to the new best
    state.portfolios[0] = slot0_own;
    memcpy(state.zero_counts[0], slot0_zc, N_IND * sizeof(int));

    // Elite pts stats (slots 0..ELITE_COUNT-1)
    float elite_max_pts = -1e9f, elite_min_pts = 1e9f, elite_mean_pts = 0.f;
    for (int s = 0; s < ELITE_COUNT; s++) {
        float p = (pred_scores[s] - port_vals[s]) / 1e9f;
        if (p > elite_max_pts) elite_max_pts = p;
        if (p < elite_min_pts) elite_min_pts = p;
        elite_mean_pts += p;
    }
    elite_mean_pts /= ELITE_COUNT;

    return {best_pts_v, elite_max_pts, elite_min_pts, elite_mean_pts, ideal_pts};
}

// ── MT1 per-industry training step ──────────────────────────────────────────────

static const char* const MT1_POOL_NAMES[4] = {"dir", "acc", "rng", "cfd"};

struct MT1ScoreBreakdown { float composite, direction, range, accuracy, confidence; };

// acc_floor: per-industry adaptive floor for accuracy denom (half the 10-day mean |actual_d|)
// range_ceiling: 4 × mean(last 10 |actual_d - comp0_delta_d|); 1e30 = no ceiling yet
static MT1ScoreBreakdown compute_mt1_scores(
    float actual_d, const float raw4[4], float acc_floor, float range_ceiling)
{
    float conf      = sigmoidf(raw4[0]);
    float delta_d   = tanhf(raw4[1]) * MT1_SCALE_DOLLARS;
    float range_pct = log1pf(expf(raw4[2]));   // softplus
    float conf4     = sigmoidf(raw4[3]);

    float sc_dir = (actual_d >= 0.f) ? conf : (1.f - conf);

    float eff_delta = fmaxf(fabsf(delta_d), MT1_RANGE_FLOOR);
    float r         = range_pct * eff_delta;
    if (range_ceiling < 1e30f) r = fminf(r, range_ceiling);
    float err       = fabsf(actual_d - delta_d);
    float m         = (r > 1e-9f) ? err / r : (err > 0.f ? 1e9f : 0.f);
    float sc_rng    = (m < 1.f) ? m : 0.f;

    float denom  = fmaxf(fabsf(actual_d), acc_floor);
    float sc_acc = fmaxf(0.f, 1.f - err / denom);

    float d      = err;
    float dor    = (r > 1e-9f) ? d / r : (d > 0.f ? 1e9f : 0.f);
    float ideal  = 1.f / (1.f + dor * dor);
    float diff   = conf4 - ideal;
    float sc_cfd = 1.f - diff * diff;

    return {0.50f * sc_dir + 0.33f * sc_rng + 0.17f * sc_acc, sc_dir, sc_rng, sc_acc, sc_cfd};
}

// ── MT1 component pool step ──────────────────────────────────────────────────────

struct MT1CompResult { float best, slot0, mean, min_v; };

static MT1CompResult step_mt1_component(
    int pool_id, int ind_i, MT1Scratch& scratch,
    float actual_d, const float in37[37], int actual_day, float sigma,
    float acc_floor, float range_ceiling)
{
    // Load re-injection models into elite slots 20–22 (slots ELITE_COUNT+WAVG_COUNT .. MT1_COMP_ELITE-1)
    for (int k = 0; k < scratch.reinject_count; k++)
        memcpy(scratch.comp_elite(pool_id, ELITE_COUNT + WAVG_COUNT + k),
               scratch.reinject(k), MT1NN_PARAMS * sizeof(float));

    // Deterministic mutation seeds (per day + industry + pool)
    {
        PCG32 seed_rng;
        seed_rng.seed((uint64_t)actual_day * 987017ULL +
                      (uint64_t)ind_i      * 10007ULL  +
                      (uint64_t)pool_id    *   997ULL  + 11111ULL);
        for (int i = 0; i < MT1_COMP_MUTS; i++)
            scratch.mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    // get_weights: reconstructs slot weights into dst (elite or mutation)
    auto get_weights = [&](int slot, float* dst) {
        if (slot < MT1_COMP_ELITE) {
            memcpy(dst, scratch.comp_elite(pool_id, slot), MT1NN_PARAMS * sizeof(float));
        } else {
            int mut_i  = slot - MT1_COMP_ELITE;
            int parent = mut_i % MT1_COMP_ELITE;
            memcpy(dst, scratch.comp_elite(pool_id, parent), MT1NN_PARAMS * sizeof(float));
            apply_gaussian(dst, MT1NN_PARAMS, sigma, scratch.mut_seeds[mut_i]);
        }
    };

    // Score all 230 main-pool slots on the pool-specific component score
    float cat_sc[MT1_COMP_SLOTS] = {};
    float out4[4];
    for (int slot = 0; slot < MT1_COMP_SLOTS; slot++) {
        if (slot < MT1_COMP_ELITE) {
            mt1_forward(scratch.comp_elite(pool_id, slot), in37, out4);
        } else {
            get_weights(slot, scratch.mut_buf);
            mt1_forward(scratch.mut_buf, in37, out4);
        }
        auto sb = compute_mt1_scores(actual_d, out4, acc_floor, range_ceiling);
        switch (pool_id) {
            case 0: cat_sc[slot] = sb.direction;  break;
            case 1: cat_sc[slot] = sb.accuracy;   break;
            case 2: cat_sc[slot] = sb.range;      break;
            default: cat_sc[slot] = sb.confidence; break;
        }
    }

    // Pool statistics (over all 230 slots)
    float mean_cat = 0.f, best_cat = cat_sc[0], min_cat = cat_sc[0];
    for (int s = 0; s < MT1_COMP_SLOTS; s++) {
        mean_cat += cat_sc[s];
        if (cat_sc[s] > best_cat) best_cat = cat_sc[s];
        if (cat_sc[s] < min_cat)  min_cat  = cat_sc[s];
    }
    mean_cat /= MT1_COMP_SLOTS;
    float slot0_cat = cat_sc[0];  // pre-selection slot 0

    // Direction backfill: if the best model on this day scored < 0.65, the day's signal
    // is too weak (near-random). Keep yesterday's elites on disk by returning early.
    if (pool_id == 0 && best_cat < MT1_DIR_BACKFILL)
        return {best_cat, slot0_cat, mean_cat, min_cat};

    // Score history candidates
    int n_hist = scratch.pool_hist_count[pool_id] * HIST_PER_DAY;
    float hist_cat_sc[HIST_DAYS * HIST_PER_DAY] = {};
    {
        int total = HIST_DAYS * HIST_PER_DAY;
        int oldest = (scratch.pool_hist_head[pool_id] * HIST_PER_DAY - n_hist + total) % total;
        for (int k = 0; k < n_hist; k++) {
            int abs_pos = (oldest + k) % total;
            float* hw = scratch.pool_hist[pool_id] + (size_t)abs_pos * MT1NN_PARAMS;
            mt1_forward(hw, in37, out4);
            auto sb = compute_mt1_scores(actual_d, out4, acc_floor, range_ceiling);
            switch (pool_id) {
                case 0: hist_cat_sc[k] = sb.direction;  break;
                case 1: hist_cat_sc[k] = sb.accuracy;   break;
                case 2: hist_cat_sc[k] = sb.range;      break;
                default: hist_cat_sc[k] = sb.confidence; break;
            }
        }
    }

    // Build candidate list and sort by component score
    struct Cand { float score; bool is_hist; int idx; };
    std::vector<Cand> cands;
    cands.reserve(MT1_COMP_SLOTS + n_hist);
    for (int s = 0; s < MT1_COMP_SLOTS; s++) cands.push_back({cat_sc[s], false, s});
    for (int k = 0; k < n_hist; k++)         cands.push_back({hist_cat_sc[k], true, k});
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.score > b.score; });

    // Select top ELITE_COUNT (17) direct elites into new_elites
    for (int rank = 0; rank < ELITE_COUNT && rank < (int)cands.size(); rank++) {
        const Cand& c = cands[rank];
        if (!c.is_hist) {
            get_weights(c.idx, scratch.new_elite(rank));
        } else {
            int total = HIST_DAYS * HIST_PER_DAY;
            int oldest = (scratch.pool_hist_head[pool_id] * HIST_PER_DAY - n_hist + total) % total;
            int abs_pos = (oldest + c.idx) % total;
            memcpy(scratch.new_elite(rank),
                   scratch.pool_hist[pool_id] + (size_t)abs_pos * MT1NN_PARAMS,
                   MT1NN_PARAMS * sizeof(float));
        }
    }
    // Fill any unfilled direct elite slots from slot 0
    for (int rank = (int)std::min((int)cands.size(), ELITE_COUNT); rank < ELITE_COUNT; rank++)
        memcpy(scratch.new_elite(rank), scratch.new_elite(0), MT1NN_PARAMS * sizeof(float));

    // Wavg blends (slots 17, 18, 19): equal-weight average of top 5, 10, 15 direct elites
    static constexpr int wavg_k[3] = {5, 10, 15};
    for (int b = 0; b < WAVG_COUNT; b++) {
        float* dst  = scratch.new_elite(ELITE_COUNT + b);
        int    k    = wavg_k[b];
        float inv_k = 1.f / (float)k;
        memset(dst, 0, MT1NN_PARAMS * sizeof(float));
        for (int e = 0; e < k; e++)
            for (int p = 0; p < MT1NN_PARAMS; p++) dst[p] += scratch.new_elite(e)[p] * inv_k;
    }

    // Re-injection slots (20–22): copy from current comp_elites (loaded from reinject_buf above)
    for (int k = 0; k < MT1_REINJECT; k++)
        memcpy(scratch.new_elite(ELITE_COUNT + WAVG_COUNT + k),
               scratch.comp_elite(pool_id, ELITE_COUNT + WAVG_COUNT + k),
               MT1NN_PARAMS * sizeof(float));

    // Commit new elite set
    memcpy(scratch.comp_elites[pool_id], scratch.new_elites,
           (size_t)MT1_COMP_ELITE * MT1NN_PARAMS * sizeof(float));

    // Save top 7 direct elites + 3 wavg blends to 5-day history
    {
        int& head  = scratch.pool_hist_head[pool_id];
        int& count = scratch.pool_hist_count[pool_id];
        for (int k = 0; k < HIST_ELITE; k++)
            memcpy(scratch.pool_hist_slot(pool_id, head, k),
                   scratch.comp_elite(pool_id, k), MT1NN_PARAMS * sizeof(float));
        for (int k = 0; k < HIST_WAVG; k++)
            memcpy(scratch.pool_hist_slot(pool_id, head, HIST_ELITE + k),
                   scratch.comp_elite(pool_id, ELITE_COUNT + k), MT1NN_PARAMS * sizeof(float));
        head = (head + 1) % HIST_DAYS;
        if (count < HIST_DAYS) count++;
    }

    // Range pool: snapshot top elites for confidence cross-injection
    if (pool_id == 2) {
        scratch.rng_inject_count = std::min(MT1_RANGE_INJECT, ELITE_COUNT);
        for (int k = 0; k < scratch.rng_inject_count; k++)
            memcpy(scratch.rng_inject(k),
                   scratch.comp_elite(2, k), MT1NN_PARAMS * sizeof(float));
    }

    // Confidence pool: overwrite bottom 5 direct elites with top range elites (anti-gaming)
    if (pool_id == 3 && scratch.rng_inject_count > 0) {
        int start = ELITE_COUNT - scratch.rng_inject_count;  // 17-5 = 12
        for (int k = 0; k < scratch.rng_inject_count; k++)
            memcpy(scratch.comp_elite(3, start + k),
                   scratch.rng_inject(k), MT1NN_PARAMS * sizeof(float));
    }

    return {best_cat, slot0_cat, mean_cat, min_cat};
}

// ── MT1 composite blend-pool step ────────────────────────────────────────────────

// Regenerate blend b into dst (deterministic per actual_day, ind_i, b)
static void gen_mt1_blend(int b, int ind_i, int actual_day, MT1Scratch& scratch, float* dst) {
    PCG32 rng; rng.seed((uint64_t)actual_day * 654321ULL +
                        (uint64_t)ind_i      * 31337ULL  +
                        (uint64_t)b          *  1009ULL);
    int src_pool[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    int src_rank[8];
    for (int p = 0; p < 4; p++) {
        int r1 = (int)(rng.next() % ELITE_COUNT);
        int r2 = (int)(rng.next() % (ELITE_COUNT - 1));
        if (r2 >= r1) r2++;
        src_rank[p*2]   = r1;
        src_rank[p*2+1] = r2;
    }
    float w[8], wsum = 0.f;
    for (int k = 0; k < 8; k++) { w[k] = 20.f - (float)src_rank[k]; wsum += w[k]; }
    float inv = 1.f / wsum;
    for (int k = 0; k < 8; k++) w[k] *= inv;
    memset(dst, 0, MT1NN_PARAMS * sizeof(float));
    for (int k = 0; k < 8; k++) {
        const float* src = scratch.comp_elite(src_pool[k], src_rank[k]);
        for (int p = 0; p < MT1NN_PARAMS; p++) dst[p] += src[p] * w[k];
    }
}

struct MT1BlendResult {
    float best_score, slot0_score, mean_score, min_score;
    float slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4;
    float comp0_residual;  // |actual_d - comp0_delta_d| for rolling_residual
};

static MT1BlendResult step_mt1_composite(
    int ind_i, MT1Scratch& scratch,
    float actual_d, const float in37[37], int actual_day,
    float acc_floor, float range_ceiling)
{
    // Score all MT1_BLEND_SLOTS (200) new blends
    float blend_sc[MT1_BLEND_SLOTS];
    float out4[4];
    float mean_sc = 0.f, best_sc = -1e9f, min_sc = 1e9f;

    for (int b = 0; b < MT1_BLEND_SLOTS; b++) {
        gen_mt1_blend(b, ind_i, actual_day, scratch, scratch.mut_buf);
        mt1_forward(scratch.mut_buf, in37, out4);
        blend_sc[b] = compute_mt1_scores(actual_d, out4, acc_floor, range_ceiling).composite;
        mean_sc += blend_sc[b];
        if (blend_sc[b] > best_sc) best_sc = blend_sc[b];
        if (blend_sc[b] < min_sc)  min_sc  = blend_sc[b];
    }
    mean_sc /= MT1_BLEND_SLOTS;

    // Score history candidates
    int n_hist = scratch.blend_hist_count * HIST_PER_DAY;
    float hist_sc[HIST_DAYS * HIST_PER_DAY] = {};
    {
        int total  = HIST_DAYS * HIST_PER_DAY;
        int oldest = (scratch.blend_hist_head * HIST_PER_DAY - n_hist + total) % total;
        for (int k = 0; k < n_hist; k++) {
            int abs_pos = (oldest + k) % total;
            float* hw = scratch.blend_hist + (size_t)abs_pos * MT1NN_PARAMS;
            mt1_forward(hw, in37, out4);
            hist_sc[k] = compute_mt1_scores(actual_d, out4, acc_floor, range_ceiling).composite;
        }
    }

    // Build sorted candidate list (blend index or history position)
    struct BLCand { float score; bool is_hist; int idx; };
    std::vector<BLCand> cands;
    cands.reserve(MT1_BLEND_SLOTS + n_hist);
    for (int b = 0; b < MT1_BLEND_SLOTS; b++) cands.push_back({blend_sc[b], false, b});
    for (int k = 0; k < n_hist; k++)          cands.push_back({hist_sc[k], true, k});
    std::sort(cands.begin(), cands.end(), [](const BLCand& a, const BLCand& b){ return a.score > b.score; });

    // Helper: retrieve weights for a candidate
    auto get_cand_weights = [&](const BLCand& c, float* dst) {
        if (!c.is_hist) {
            gen_mt1_blend(c.idx, ind_i, actual_day, scratch, dst);
        } else {
            int total  = HIST_DAYS * HIST_PER_DAY;
            int oldest = (scratch.blend_hist_head * HIST_PER_DAY - n_hist + total) % total;
            int abs_pos = (oldest + c.idx) % total;
            memcpy(dst, scratch.blend_hist + (size_t)abs_pos * MT1NN_PARAMS, MT1NN_PARAMS * sizeof(float));
        }
    };

    // Top candidate → production model (comp0_buf)
    float raw0[4] = {};
    get_cand_weights(cands[0], scratch.comp0_buf);
    mt1_forward(scratch.comp0_buf, in37, raw0);
    float comp0_delta_d = tanhf(raw0[1]) * MT1_SCALE_DOLLARS;

    // Save top 3 to reinject_buf (for next day's component pools)
    scratch.reinject_count = std::min(MT1_REINJECT, (int)cands.size());
    for (int k = 0; k < scratch.reinject_count; k++)
        get_cand_weights(cands[k], scratch.reinject(k));

    // Save top HIST_PER_DAY (10) to blend_hist circular buffer
    int n_save = std::min(HIST_PER_DAY, (int)cands.size());
    int& bhead = scratch.blend_hist_head;
    for (int k = 0; k < n_save; k++) {
        float* dst = scratch.blend_hist_slot(bhead, k);
        get_cand_weights(cands[k], dst);
    }
    bhead = (bhead + 1) % HIST_DAYS;
    if (scratch.blend_hist_count < HIST_DAYS) scratch.blend_hist_count++;

    float slot0_conf      = sigmoidf(raw0[0]);
    float slot0_delta_t   = tanhf(raw0[1]);
    float slot0_range_pct = log1pf(expf(raw0[2]));
    float slot0_conf4     = sigmoidf(raw0[3]);

    return {best_sc, cands[0].score, mean_sc, min_sc,
            slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4,
            fabsf(actual_d - comp0_delta_d)};
}

// ── MT1 orchestrator ─────────────────────────────────────────────────────────────

static MT1Result step_mt1(int ind_i, MT1Scratch& scratch,
                           float actual_d, const float in37[37],
                           int actual_day, float sigma) {
    if (actual_day < MT1_START_DAY) return {};

    // Adaptive per-industry floors from rolling buffers
    float acc_floor     = MT1_FLOOR_COLD / 2.f;  // cold-start
    float range_ceiling = 1e30f;
    if (scratch.rolling_count > 0) {
        float sum_a = 0.f, sum_r = 0.f;
        for (int k = 0; k < scratch.rolling_count; k++) {
            sum_a += scratch.rolling_actual[k];
            sum_r += scratch.rolling_residual[k];
        }
        acc_floor     = sum_a / (float)scratch.rolling_count / 2.f;
        range_ceiling = MT1_RANGE_CEIL_MULT * sum_r / (float)scratch.rolling_count;
    }

    // Step 4 component pools (dir, acc, rng, cfd)
    MT1CompResult cr[4];
    for (int p = 0; p < 4; p++)
        cr[p] = step_mt1_component(p, ind_i, scratch, actual_d, in37, actual_day, sigma,
                                    acc_floor, range_ceiling);

    // Step composite blend pool
    auto br = step_mt1_composite(ind_i, scratch, actual_d, in37, actual_day,
                                  acc_floor, range_ceiling);

    // Update rolling buffers AFTER scoring
    scratch.rolling_actual[scratch.rolling_head]   = fabsf(actual_d);
    scratch.rolling_residual[scratch.rolling_head] = br.comp0_residual;
    scratch.rolling_head = (scratch.rolling_head + 1) % MT1_ROLLING_DAYS;
    if (scratch.rolling_count < MT1_ROLLING_DAYS) scratch.rolling_count++;

    return {br.best_score, br.slot0_score, br.mean_score, br.min_score,
            br.slot0_conf, br.slot0_delta_t, br.slot0_range_pct, br.slot0_conf4,
            cr[0].best, cr[0].slot0, cr[0].mean, cr[0].min_v,
            cr[2].best, cr[2].slot0, cr[2].mean, cr[2].min_v,
            cr[1].best, cr[1].slot0, cr[1].mean, cr[1].min_v,
            cr[3].best, cr[3].slot0, cr[3].mean, cr[3].min_v};
}

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
        const MasterPortfolio* mp5[5], *mp10[10], *mp15[15];
        for (int k=0;k<n5;k++)  mp5[k]  = &state.portfolios[src_rank[k]];
        for (int k=0;k<n10;k++) mp10[k] = &state.portfolios[src_rank[k]];
        for (int k=0;k<n15;k++) mp15[k] = &state.portfolios[src_rank[k]];
        MasterPortfolio wp5={}, wp10={}, wp15={};
        wavg_mst_portfolio(mp5,  w5,  n5,  wp5);
        wavg_mst_portfolio(mp10, w10, n10, wp10);
        wavg_mst_portfolio(mp15, w15, n15, wp15);

        for (int k = 0; k < n_top; k++) {
            int slot = src_rank[k];
            if (slot < ELITE_POOL) {
                memcpy(scratch.new_elite(k), scratch.elite(slot), MT2NN_PARAMS * sizeof(float));
            } else {
                int mut_i = slot - ELITE_POOL, parent = mut_i / MUTATIONS_PER_PARENT;
                memcpy(scratch.new_elite(k), scratch.elite(parent), MT2NN_PARAMS * sizeof(float));
                apply_gaussian(scratch.new_elite(k), MT2NN_PARAMS, sigma, scratch.mut_seeds[mut_i]);
            }
            new_mports[k] = state.portfolios[slot];
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

    float elite_max_pts = -1e9f, elite_min_pts = 1e9f, elite_mean_pts = 0.f;
    for (int s = 0; s < ELITE_COUNT; s++) {
        float p = (pred_scores[s] - port_vals[s]) / 1e9f;
        if (p > elite_max_pts) elite_max_pts = p;
        if (p < elite_min_pts) elite_min_pts = p;
        elite_mean_pts += p;
    }
    elite_mean_pts /= ELITE_COUNT;

    if (injected_out) *injected_out = injected;
    return {best_pts_v, elite_max_pts, elite_min_pts, elite_mean_pts, ideal_pts};
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

static void save_mt1_all(const std::string& dir, int ind_i, const MT1Scratch& scratch) {
    const char* ind = g_ind_names[ind_i].c_str();
    char path[512];
    for (int p = 0; p < 4; p++) {
        for (int slot = 0; slot < MT1_COMP_ELITE; slot++) {
            snprintf(path, sizeof(path), "%s/mt1_%s_%s_elite_%d.bin",
                     dir.c_str(), ind, MT1_POOL_NAMES[p], slot);
            if (!save_bin(path, scratch.comp_elites[p] + (size_t)slot * MT1NN_PARAMS, MT1NN_PARAMS))
                log_msg(std::string("WARNING: could not save ") + path);
        }
        // Component pool history
        snprintf(path, sizeof(path), "%s/mt1_%s_%s_hist.bin", dir.c_str(), ind, MT1_POOL_NAMES[p]);
        FILE* f = fopen(path, "wb");
        if (f) {
            int meta[2] = {scratch.pool_hist_head[p], scratch.pool_hist_count[p]};
            fwrite(meta, sizeof(int), 2, f);
            fwrite(scratch.pool_hist[p], sizeof(float),
                   (size_t)HIST_DAYS * HIST_PER_DAY * MT1NN_PARAMS, f);
            fclose(f);
        }
    }
    // Composite blend history
    snprintf(path, sizeof(path), "%s/mt1_%s_comp_hist.bin", dir.c_str(), ind);
    FILE* fch = fopen(path, "wb");
    if (fch) {
        int meta[2] = {scratch.blend_hist_head, scratch.blend_hist_count};
        fwrite(meta, sizeof(int), 2, fch);
        fwrite(scratch.blend_hist, sizeof(float),
               (size_t)HIST_DAYS * HIST_PER_DAY * MT1NN_PARAMS, fch);
        fclose(fch);
    }
    // Re-injection buffer
    snprintf(path, sizeof(path), "%s/mt1_%s_comp_reinject.bin", dir.c_str(), ind);
    FILE* fri = fopen(path, "wb");
    if (fri) {
        fwrite(&scratch.reinject_count, sizeof(int), 1, fri);
        if (scratch.reinject_count > 0)
            fwrite(scratch.reinject_buf, sizeof(float),
                   (size_t)scratch.reinject_count * MT1NN_PARAMS, fri);
        fclose(fri);
    }
    // Range-inject buffer (top range elites for confidence cross-injection)
    snprintf(path, sizeof(path), "%s/mt1_%s_rng_inject.bin", dir.c_str(), ind);
    FILE* frni = fopen(path, "wb");
    if (frni) {
        fwrite(&scratch.rng_inject_count, sizeof(int), 1, frni);
        if (scratch.rng_inject_count > 0)
            fwrite(scratch.rng_inject_buf, sizeof(float),
                   (size_t)scratch.rng_inject_count * MT1NN_PARAMS, frni);
        fclose(frni);
    }
    // Composite best (production model)
    snprintf(path, sizeof(path), "%s/mt1_%s_comp_0.bin", dir.c_str(), ind);
    save_bin(path, scratch.comp0_buf, MT1NN_PARAMS);
}

static void load_or_init_mt1(const std::string& dir, const std::string& load_dir,
                               int ind_i, MT1Scratch& scratch) {
    const char* ind = g_ind_names[ind_i].c_str();
    PCG32 rng; rng.seed((uint64_t)(ind_i + N_IND) * 777777777ULL + 314159265ULL);
    char path[512];

    for (int p = 0; p < 4; p++) {
        for (int slot = 0; slot < MT1_COMP_ELITE; slot++) {
            float* e = scratch.comp_elite(p, slot);
            bool loaded = false;
            for (const std::string* sd : {&load_dir, &dir}) {
                if (sd->empty()) continue;
                snprintf(path, sizeof(path), "%s/mt1_%s_%s_elite_%d.bin",
                         sd->c_str(), ind, MT1_POOL_NAMES[p], slot);
                if (load_bin(path, e, MT1NN_PARAMS)) { loaded = true; break; }
            }
            if (!loaded) init_mt1_weights(e, rng);
        }
        // Component pool history
        bool hist_loaded = false;
        for (const std::string* sd : {&load_dir, &dir}) {
            if (sd->empty()) continue;
            snprintf(path, sizeof(path), "%s/mt1_%s_%s_hist.bin",
                     sd->c_str(), ind, MT1_POOL_NAMES[p]);
            FILE* f = fopen(path, "rb");
            if (!f) continue;
            int meta[2] = {};
            if (fread(meta, sizeof(int), 2, f) == 2) {
                scratch.pool_hist_head[p]  = std::max(0, std::min(meta[0], HIST_DAYS - 1));
                scratch.pool_hist_count[p] = std::max(0, std::min(meta[1], HIST_DAYS));
            }
            fread(scratch.pool_hist[p], sizeof(float),
                  (size_t)HIST_DAYS * HIST_PER_DAY * MT1NN_PARAMS, f);
            fclose(f);
            hist_loaded = true;
            break;
        }
        if (!hist_loaded) { scratch.pool_hist_head[p] = 0; scratch.pool_hist_count[p] = 0; }
    }

    // Composite blend history
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_comp_hist.bin", sd->c_str(), ind);
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        int meta[2] = {};
        if (fread(meta, sizeof(int), 2, f) == 2) {
            scratch.blend_hist_head  = std::max(0, std::min(meta[0], HIST_DAYS - 1));
            scratch.blend_hist_count = std::max(0, std::min(meta[1], HIST_DAYS));
        }
        fread(scratch.blend_hist, sizeof(float),
              (size_t)HIST_DAYS * HIST_PER_DAY * MT1NN_PARAMS, f);
        fclose(f);
        break;
    }

    // Re-injection buffer
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_comp_reinject.bin", sd->c_str(), ind);
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        int cnt = 0;
        if (fread(&cnt, sizeof(int), 1, f) == 1 && cnt > 0 && cnt <= MT1_REINJECT) {
            fread(scratch.reinject_buf, sizeof(float), (size_t)cnt * MT1NN_PARAMS, f);
            scratch.reinject_count = cnt;
        }
        fclose(f);
        break;
    }

    // Range-inject buffer
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_rng_inject.bin", sd->c_str(), ind);
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        int cnt = 0;
        if (fread(&cnt, sizeof(int), 1, f) == 1 && cnt > 0 && cnt <= MT1_RANGE_INJECT) {
            fread(scratch.rng_inject_buf, sizeof(float), (size_t)cnt * MT1NN_PARAMS, f);
            scratch.rng_inject_count = cnt;
        }
        fclose(f);
        break;
    }

    // Composite best (production model) — load into comp0_buf if available
    for (const std::string* sd : {&load_dir, &dir}) {
        if (sd->empty()) continue;
        snprintf(path, sizeof(path), "%s/mt1_%s_comp_0.bin", sd->c_str(), ind);
        if (load_bin(path, scratch.comp0_buf, MT1NN_PARAMS)) break;
    }
}

static void save_mt2_elites(const std::string& dir, const float* elite_buf) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        char p[512]; snprintf(p, sizeof(p), "%s/mt2_elite_%d.bin", dir.c_str(), slot);
        if (!save_bin(p, elite_buf + (size_t)slot * MT2NN_PARAMS, MT2NN_PARAMS))
            log_msg(std::string("WARNING: could not save ") + p);
    }
}

static void load_or_init_mt2(const std::string& dir, const std::string& load_dir,
                               float* elite_buf) {
    PCG32 rng; rng.seed(0xCAFED00DBEEF1234ULL);
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        float* e = elite_buf + (size_t)slot * MT2NN_PARAMS;
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
}


// ── MT binary log ────────────────────────────────────────────────────────────────

static constexpr uint32_t MT_LOG_MAGIC   = 0x4D543132u;  // 'MT12'
static constexpr uint32_t MT_LOG_VERSION = 3u;

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
    // MT2
    float    mt2_best_pts, mt2_slot0_pts, mt2_ideal_pts;
    uint8_t  mt2_injected;
    uint8_t  pad[3];
};
static_assert(sizeof(MTLogRecord) == 984, "MTLogRecord must be 984 bytes");

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
                           const IndResult* res, const MasterResult& mst) {
    fprintf(csv, "%d,%d", pass_num + 1, actual_day + 1);
    for (int i = 0; i < N_IND; i++)
        fprintf(csv, ",%+10.2f,%+10.2f,%+10.2f",
                res[i].elite_max_val, res[i].elite_min_val, res[i].elite_mean_val);
    fprintf(csv, ",%+.2f,%+.2f,%+.2f,%+.2f\n",
            mst.elite_max_pts, mst.elite_min_pts, mst.elite_mean_pts, mst.ideal_pts);
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
        "          [--workers N] [--master-only] [--preserve-stock-data] [--no-save]\n"
        "       %s --output DIR [--load-dir DIR] ...  (diagnostic/override)\n", prog, prog);
}

int main(int argc, char* argv[]) {
    // Parse args
    std::string output_dir, load_dir, account;
    int  start_day = -1, stop_day = -1, passes = 1, num_workers = 2;
    float sigma = 0.01f, master_sigma = -1.f, sigma_decay = 0.5f;
    bool master_only = false, preserve_stock = false;

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
        else if (arg == "--workers"  && a+1<argc) { num_workers=atoi(argv[++a]);}
        else if (arg == "--master-only") master_only = true;
        else if (arg == "--preserve-stock-data") preserve_stock = true;
        else if (arg == "--no-save") g_no_save = true;
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
    if (master_sigma < 0.f) master_sigma = sigma;

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
        fprintf(csv, ",mt2_elite_max_pts,mt2_elite_min_pts,mt2_elite_mean_pts,mt2_ideal_pts\n");
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
        float cur_sigma     = sigma        * powf(sigma_decay, (float)pass);
        float cur_mst_sigma = master_sigma * powf(sigma_decay, (float)pass);
        log_msg("===== PASS " + std::to_string(pass+1) + "/" + std::to_string(passes) +
                " | sigma=" + std::to_string(cur_sigma).substr(0,8) +
                " | master_sigma=" + std::to_string(cur_mst_sigma).substr(0,8) + " =====");

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
        // Load MT1 elites (12 per-industry × 4 component pools + composite) and MT2 once at pass start
        for (int i = 0; i < N_IND; i++)
            load_or_init_mt1(output_dir, load_dir, i, mt1_scratches[i]);
        load_or_init_mt2(output_dir, load_dir, mt2_scratch->elite_buf);
        // Init MT2 portfolio state
        mst->portfolios[0].cash = MST_STARTING_CASH;
        for (int i = 0; i < N_IND; i++) mst->portfolios[0].holdings[i] = 0.f;
        for (int s = 1; s < N_SLOTS; s++) mst->portfolios[s] = mst->portfolios[0];
        memset(mst->ind_val_hist, 0, sizeof(mst->ind_val_hist));
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

            // Build 444-feature vector (uses ind_val_hist from BEFORE today)
            float today444[444];
            build_master_features(mst->ind_val_hist, mst->ind_hist_count, today444);

            // Compute actual fractional returns from slot0 results
            float actual_perf[N_IND] = {};
            for (int i = 0; i < N_IND; i++)
                if (results[i].baseline > 0.f)
                    actual_perf[i] = results[i].slot0_score / results[i].baseline - 1.f;

            // MT1 step × 12 (actual_day >= MT1_START_DAY)
            MT1Result mt1_res[N_IND] = {};
            if (actual_day >= MT1_START_DAY) {
                for (int i = 0; i < N_IND; i++) {
                    const float* in37 = today444 + i * 37;
                    float actual_d_i  = results[i].slot0_score - results[i].baseline;
                    mt1_res[i] = step_mt1(i, mt1_scratches[i], actual_d_i,
                                          in37, actual_day, cur_mst_sigma);
                }
            }

            // MT2 step (actual_day >= MASTER_START_DAY)
            MasterResult master_res = {};
            bool mt2_injected = false;
            if (actual_day >= MASTER_START_DAY) {
                // Build 48-feature MT2 input from MT1 slot0 raw activations (no normalization)
                float in48[48];
                for (int i = 0; i < N_IND; i++) {
                    in48[i*4 + 0] = mt1_res[i].slot0_conf;
                    in48[i*4 + 1] = mt1_res[i].slot0_delta_t;
                    in48[i*4 + 2] = mt1_res[i].slot0_range_pct;
                    in48[i*4 + 3] = mt1_res[i].slot0_conf4;
                }
                master_res = step_mt2(*mst, *mt2_scratch, in48, actual_perf,
                                      actual_day, total_days, cur_mst_sigma, &mt2_injected);
            }

            // Append today's best-slot industry values to rolling ind_val_hist buffer
            for (int i = 0; i < N_IND; i++) {
                float today_v = results[i].baseline + results[i].best_delta;
                if (mst->ind_hist_count < IND_HIST_CAP) {
                    mst->ind_val_hist[i][mst->ind_hist_count] = today_v;
                } else {
                    memmove(mst->ind_val_hist[i], mst->ind_val_hist[i] + 1,
                            (IND_HIST_CAP - 1) * sizeof(float));
                    mst->ind_val_hist[i][IND_HIST_CAP - 1] = today_v;
                }
            }
            if (mst->ind_hist_count < IND_HIST_CAP) mst->ind_hist_count++;

            // Log CSV row
            if (csv) write_csv_row(csv, pass, actual_day, results, master_res);

            // Write MT binary log record (once MT1 is active)
            if (mt_log && actual_day >= MT1_START_DAY) {
                MTLogRecord rec{};
                rec.pass_num     = (uint32_t)pass;
                rec.actual_day   = (uint32_t)actual_day;
                for (int i = 0; i < N_IND; i++) {
                    rec.mt1_best[i]      = mt1_res[i].best_score;
                    rec.mt1_slot0[i]     = mt1_res[i].slot0_score;
                    rec.mt1_mean[i]      = mt1_res[i].mean_score;
                    rec.mt1_min[i]       = mt1_res[i].min_score;
                    rec.mt1_dir_best[i]  = mt1_res[i].best_dir;
                    rec.mt1_dir_slot0[i] = mt1_res[i].slot0_dir;
                    rec.mt1_dir_mean[i]  = mt1_res[i].mean_dir;
                    rec.mt1_dir_min[i]   = mt1_res[i].min_dir;
                    rec.mt1_rng_best[i]  = mt1_res[i].best_rng;
                    rec.mt1_rng_slot0[i] = mt1_res[i].slot0_rng;
                    rec.mt1_rng_mean[i]  = mt1_res[i].mean_rng;
                    rec.mt1_rng_min[i]   = mt1_res[i].min_rng;
                    rec.mt1_acc_best[i]  = mt1_res[i].best_acc;
                    rec.mt1_acc_slot0[i] = mt1_res[i].slot0_acc;
                    rec.mt1_acc_mean[i]  = mt1_res[i].mean_acc;
                    rec.mt1_acc_min[i]   = mt1_res[i].min_acc;
                    rec.mt1_cfd_best[i]  = mt1_res[i].best_cfd;
                    rec.mt1_cfd_slot0[i] = mt1_res[i].slot0_cfd;
                    rec.mt1_cfd_mean[i]  = mt1_res[i].mean_cfd;
                    rec.mt1_cfd_min[i]   = mt1_res[i].min_cfd;
                }
                rec.mt2_best_pts   = master_res.best_pts;
                rec.mt2_slot0_pts  = master_res.elite_mean_pts;  // slot0 pts proxy
                rec.mt2_ideal_pts  = master_res.ideal_pts;
                rec.mt2_injected   = mt2_injected ? 1u : 0u;
                write_mt_log_record(mt_log, rec);
            }

            // Periodic MT1/MT2 save (industry elites already saved inside step_industry each day)
            if (!g_no_save && (day_num % 50 == 49 || day_num == num_days - 1)) {
                log_msg("Saving MT1/MT2 elites to " + output_dir + " ...");
                for (int i = 0; i < N_IND; i++)
                    save_mt1_all(output_dir, i, mt1_scratches[i]);
                save_mt2_elites(output_dir, mt2_scratch->elite_buf);
            }
        }

        // Save MT1/MT2 after each pass (industry elites already saved by step_industry)
        if (!g_no_save) {
            log_msg("Pass " + std::to_string(pass+1) + " complete — saving MT1/MT2 elites");
            for (int i = 0; i < N_IND; i++)
                save_mt1_all(output_dir, i, mt1_scratches[i]);
            save_mt2_elites(output_dir, mt2_scratch->elite_buf);
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
