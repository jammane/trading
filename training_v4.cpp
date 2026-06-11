// training_v4.cpp — C++ port of training_v4.py
// Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
// Run:   ./build/training_v4_cpp --output models [--load-dir DIR] [--start-day N] [--stop-day N]
//        [--passes N] [--sigma F] [--master-sigma F] [--sigma-decay F] [--workers N]

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

namespace fs = std::filesystem;

// ── Universe ───────────────────────────────────────────────────────────────────

static constexpr int N_IND   = 12;
static constexpr int IND_SYMS = 12;
static constexpr int N_SYMS  = N_IND * IND_SYMS;   // 144

static const char* const IND_NAMES[N_IND] = {
    "tech_hardware","tech_software_ai","financials","consumer_discretionary",
    "consumer_services","health_care","industrials","consumer_staples",
    "energy","utilities","real_estate","materials"
};
static const char* const IND_SHORT[N_IND] = {
    "hardware ","software ","financial","discret  ","services ","health   ",
    "industrl ","staples  ","energy   ","utilitie ","land     ","materials"
};

static const char* const SYMS[N_IND][IND_SYMS] = {
    {"NVDA","AMD","MU","SMCI","MRVL","ON","AMAT","LRCX","KLAC","TSM","SWKS","MPWR"},
    {"PLTR","SNOW","DDOG","NET","CRWD","ZS","PANW","NOW","ADBE","CRM","FTNT","OKTA"},
    {"XYZ","PYPL","AFRM","UPST","MELI","COIN","GS","SCHW","C","COF","BX","APO"},
    {"TSLA","RCL","XPEV","LI","APTV","GM","LEA","WYNN","BKNG","ABNB","UBER","LYFT"},
    {"NFLX","ROKU","SPOT","META","IAC","PINS","DASH","RBLX","TTWO","LYV","MTCH","WBD"},
    {"MRNA","BNTX","IMVT","CRSP","ARWR","MYGN","NTRA","INMD","HIMS","BEAM","ACAD","BMRN"},
    {"BA","GE","CAT","DE","DAL","UAL","XPO","LUV","ALK","GNRC","BTU","STLD"},
    {"CELH","SFM","ELF","LULU","DECK","YETI","NKE","CROX","DKNG","PENN","MGM","CZR"},
    {"FANG","DVN","OXY","APA","AR","EQT","RRC","SM","SLB","COP","EOG","VLO"},
    {"ENPH","FSLR","SEDG","CWEN","VST","BE","BEP","DQ","CSIQ","JKS","HASI","NRG"},
    {"DHI","LEN","PHM","TOL","MTH","KBH","BZH","TMHC","LGIH","CSGP","Z","SKY"},
    {"NEM","AEM","FCX","SCCO","TECK","AA","SQM","WPM","AU","PAAS","GFI","CDE"}
};

// ── Training constants ─────────────────────────────────────────────────────────

static constexpr int   N_SLOTS             = 200;
static constexpr int   ELITE_COUNT         = 17;
static constexpr int   WAVG_COUNT          = 3;
static constexpr int   ELITE_POOL          = 20;
static constexpr int   MUTATIONS_PER_PARENT = 9;
static constexpr int   HIST_WINDOW         = 15;

static constexpr float IND_STARTING_CASH   = 25000.0f;
static constexpr float MST_STARTING_CASH   = 300000.0f;
static constexpr float IND_UNIT_PRICE      = 25000.0f;
static constexpr float MAX_SINGLE_STOCK_PCT = 0.60f;
static constexpr float SEC_FEE_RATE        = 0.0000278f;
static constexpr float FINRA_TAF_PER_SHARE = 0.000166f;
static constexpr float FINRA_TAF_MAX       = 8.30f;
static constexpr float SLIPPAGE_RATE       = 0.001f;

static constexpr int   STOCKNN_PARAMS = 921625;
static constexpr int   MASTERNN_PARAMS = 923560;

// ── Layer dimensions ───────────────────────────────────────────────────────────

static constexpr int STOCK_INJ_IN [14] = {180,185,190,195,200,205,210,215,220,225,230,235,240,245};
static constexpr int STOCK_INJ_OUT[14] = {125,130,135,140,145,150,155,160,165,170,175,180,185,190};
static constexpr int MAST_INJ_IN  [14] = {181,186,191,196,201,206,211,216,221,226,231,236,241,246};
static constexpr int MAST_INJ_OUT [14] = {125,130,135,140,145,150,155,160,165,170,175,180,185,190};

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

// Float offsets into the flat weight array for MasterNN
static constexpr int MAST_SEED_W  = 0;
static constexpr int MAST_SEED_B  = 7320;
static constexpr int MAST_INJ_W[14] = {
     7440, 30190, 54500, 80420,108000,137290,168340,201200,
    235920,272550,311140,351740,394400,439170
};
static constexpr int MAST_INJ_B[14] = {
    30065, 54370, 80285,107860,137145,168190,201045,235760,
    272385,310970,351565,394220,438985,485910
};
static constexpr int MAST_TODAY_W = 486100;
static constexpr int MAST_TODAY_B = 611800;
static constexpr int MAST_FLAT1_W = 612100;
static constexpr int MAST_FLAT1_B = 702100;
static constexpr int MAST_FLAT2_W = 702400;
static constexpr int MAST_FLAT2_B = 792400;
static constexpr int MAST_FC1_W   = 792700;
static constexpr int MAST_FC1_B   = 862900;
static constexpr int MAST_FC2_W   = 863134;
static constexpr int MAST_FC2_B   = 902446;
static constexpr int MAST_FC3_W   = 902614;
static constexpr int MAST_FC3_B   = 919750;
static constexpr int MAST_OUT_W   = 919852;
static constexpr int MAST_OUT_B   = 923524;

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

// MasterNN forward — weights[] is MASTERNN_PARAMS floats.
// hist15x61 is [15][61], today229 is [229].
// Output out36: [0..11] softmax alloc, [12..23] sigmoid depth, [24..35] sigmoid trigger.
static void master_forward(const float* W, const float* hist15x61,
                           const float* today229, float* out36) {
    float x[300], y[300], cat[430], fc1[234], fc2[168], fc3[102];

    // Seed: hist[0][61] → 120
    sgemv_relu(W + MAST_SEED_W, W + MAST_SEED_B, hist15x61, x, 120, 61);

    // Inject layers i=0..13
    int xsz = 120;
    for (int i = 0; i < 14; i++) {
        int in_sz  = MAST_INJ_IN[i];
        int out_sz = MAST_INJ_OUT[i];
        memcpy(cat, x, xsz * sizeof(float));
        memcpy(cat + xsz, hist15x61 + (i + 1) * 61, 61 * sizeof(float));
        sgemv_relu(W + MAST_INJ_W[i], W + MAST_INJ_B[i], cat, y, out_sz, in_sz);
        memcpy(x, y, out_sz * sizeof(float));
        xsz = out_sz;
    }

    // Today: cat(x[190], today[229]) = 419 → 300
    memcpy(cat, x, 190 * sizeof(float));
    memcpy(cat + 190, today229, 229 * sizeof(float));
    sgemv_relu(W + MAST_TODAY_W, W + MAST_TODAY_B, cat, x, 300, 419);

    // Flat
    sgemv_relu(W + MAST_FLAT1_W, W + MAST_FLAT1_B, x, y, 300, 300);
    sgemv_relu(W + MAST_FLAT2_W, W + MAST_FLAT2_B, y, x, 300, 300);

    // Funnel
    sgemv_relu(W + MAST_FC1_W, W + MAST_FC1_B, x,   fc1, 234, 300);
    sgemv_relu(W + MAST_FC2_W, W + MAST_FC2_B, fc1, fc2, 168, 234);
    sgemv_relu(W + MAST_FC3_W, W + MAST_FC3_B, fc2, fc3, 102, 168);

    // Output
    sgemv_only(W + MAST_OUT_W, W + MAST_OUT_B, fc3, out36, 36, 102);

    // Softmax for alloc[0..11]
    float mx = *std::max_element(out36, out36 + 12);
    float sm = 0.f;
    for (int i = 0; i < 12; i++) { out36[i] = expf(out36[i] - mx); sm += out36[i]; }
    for (int i = 0; i < 12; i++) out36[i] /= sm;

    // Sigmoid for depth[12..23] and trigger[24..35]
    for (int i = 12; i < 36; i++) out36[i] = sigmoidf(out36[i]);
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

struct MasterState {
    MasterPortfolio portfolios[N_SLOTS];
    SymHist         sym_hist[N_SYMS];   // all 144 symbols, indexed [ind*12+sym]
    float           flat_cos_hist[15];
    int             flat_cos_len{0};
    // elites removed — stored in MasterScratch (heap-allocated in main)
};

// ── Per-worker scratch (heap-allocated once per thread, ~162 MB total) ──────────

struct WorkerScratch {
    float*    elite_buf;    // [ELITE_POOL * STOCKNN_PARAMS] — loaded from disk per industry
    float*    new_elites;   // [ELITE_POOL * STOCKNN_PARAMS] — temp for selection reorder
    float*    wavg_buf;     // [3 * STOCKNN_PARAMS]
    float*    mut_buf;      // [STOCKNN_PARAMS]
    uint64_t  mut_seeds[N_SLOTS - ELITE_POOL];

    WorkerScratch() {
        size_t ep = (size_t)ELITE_POOL * STOCKNN_PARAMS;
        elite_buf  = new float[ep]();
        new_elites = new float[ep]();
        wavg_buf   = new float[3 * STOCKNN_PARAMS]();
        mut_buf    = new float[STOCKNN_PARAMS]();
    }
    ~WorkerScratch() {
        delete[] elite_buf; delete[] new_elites;
        delete[] wavg_buf;  delete[] mut_buf;
    }
    float* elite(int i)     { return elite_buf  + (size_t)i * STOCKNN_PARAMS; }
    float* new_elite(int i) { return new_elites + (size_t)i * STOCKNN_PARAMS; }
    float* wavg(int i)      { return wavg_buf   + (size_t)i * STOCKNN_PARAMS; }
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
    }
    ~MasterScratch() {
        delete[] elite_buf; delete[] new_elites;
        delete[] wavg_buf;  delete[] mut_buf;
    }
    float* elite(int i)     { return elite_buf  + (size_t)i * MASTERNN_PARAMS; }
    float* new_elite(int i) { return new_elites + (size_t)i * MASTERNN_PARAMS; }
    float* wavg(int i)      { return wavg_buf   + (size_t)i * MASTERNN_PARAMS; }
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
    kaiming_init(W + MAST_SEED_W, 120, 61, rng);
    for (int i = 0; i < 14; i++) {
        int in_sz = MAST_INJ_IN[i], out_sz = MAST_INJ_OUT[i];
        float bound = 1.0f / sqrtf((float)in_sz);
        for (int k = 0; k < out_sz * in_sz; k++)
            W[MAST_INJ_W[i] + k] = (rng.next_float() * 2.f - 1.f) * bound;
        for (int k = 0; k < out_sz; k++)
            W[MAST_INJ_B[i] + k] = (rng.next_float() * 2.f - 1.f) * bound;
    }
    kaiming_init(W + MAST_TODAY_W, 300, 419, rng);
    kaiming_init(W + MAST_FLAT1_W, 300, 300, rng);
    kaiming_init(W + MAST_FLAT2_W, 300, 300, rng);
    kaiming_init(W + MAST_FC1_W,   234, 300, rng);
    kaiming_init(W + MAST_FC2_W,   168, 234, rng);
    kaiming_init(W + MAST_FC3_W,   102, 168, rng);
    kaiming_init(W + MAST_OUT_W,    36, 102, rng);
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
};

// ── Forward declarations (needed because step_industry calls load/save defined later) ──
static void load_or_init_industry(const std::string& dir, const std::string& load_dir,
                                   int ind_i, float* elite_buf);
static void save_industry_elites(const std::string& dir, int ind_i, const float* elite_buf);

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
    const OHLCV* day_sym  = day.sym[ind_i];
    const OHLCV* fill_sym = fill ? fill->sym[ind_i] : day_sym;
    const char* const* symbols = SYMS[ind_i];

    // Compute num_past = minimum history length across symbols
    int num_past = HIST_WINDOW;
    for (int j = 0; j < IND_SYMS; j++)
        num_past = std::min(num_past, state.hist[j].len);

    if (day_num % 10 == 0 || day_num == num_days - 1)
        log_msg(std::string("[") + IND_SHORT[ind_i] + "] Day " +
                std::to_string(actual_day + 1) + "/" + std::to_string(total_avail) +
                " — running 200 models (history=" + std::to_string(num_past) + "/15 days warm)");

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

    // ── Score, flags, floor check ────────────────────────────────────────────
    float best_score  = *std::max_element(slot_scores, slot_scores + N_SLOTS);
    float best_delta  = best_score - baseline;
    float worst_delta = *std::min_element(slot_scores, slot_scores + N_SLOTS) - baseline;

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
        return {baseline, baseline, 0.f, 0.f, 0.f, 0};
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

        // Build sorted surviving list
        std::vector<std::pair<float,int>> surviving;
        for (int s = 0; s < N_SLOTS; s++) {
            if (inactive[s]) continue;
            if (slot_scores[s] < below_floor_thresh) continue;
            if (sel_scores[s] < survival_floor) continue;
            surviving.push_back({sel_scores[s], s});
        }
        if (surviving.empty()) {
            // relax: drop inactive filter
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

            // Portfolio wavg uses original slot indices (portfolios[0..199] are all valid)
            Portfolio wp5{}, wp10{}, wp15{};
            const Portfolio* p5[5], *p10[10], *p15[15];
            for (int k = 0; k < n5;  k++) p5[k]  = &state.portfolios[src_rank[k]];
            for (int k = 0; k < n10; k++) p10[k] = &state.portfolios[src_rank[k]];
            for (int k = 0; k < n15; k++) p15[k] = &state.portfolios[src_rank[k]];
            wavg_portfolio(p5,  w5_weights,  n5,  wp5);
            wavg_portfolio(p10, w10_weights, n10, wp10);
            wavg_portfolio(p15, w15_weights, n15, wp15);

            // Copy/regenerate top-n_top into new_elites[0..n_top-1].
            // src_rank[k] is a slot number (0-199): elites copy directly;
            // mutations (slots 20-199) must be regenerated from parent + original seed.
            Portfolio new_ports[ELITE_POOL];
            for (int k = 0; k < n_top; k++) {
                int slot = src_rank[k];
                if (slot < ELITE_POOL) {
                    memcpy(scratch.new_elite(k), scratch.elite(slot), STOCKNN_PARAMS * sizeof(float));
                } else {
                    int mut_i  = slot - ELITE_POOL;
                    int parent = mut_i / MUTATIONS_PER_PARENT;
                    memcpy(scratch.new_elite(k), scratch.elite(parent), STOCKNN_PARAMS * sizeof(float));
                    apply_gaussian(scratch.new_elite(k), STOCKNN_PARAMS, sigma, mut_seeds[mut_i]);
                }
                new_ports[k] = state.portfolios[slot];
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
                if (sl < ELITE_COUNT) elite_display += std::to_string(sl) + ".0";
                else if (sl < ELITE_POOL) {
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

    // Report: top_hold = slot0 holdings value, top_cash = slot0 cash
    float top_hold = 0.f;
    for (int j = 0; j < IND_SYMS; j++) {
        float price = fill_sym[j].valid ? fill_sym[j].close :
                      day_sym[j].valid  ? day_sym[j].close  : 0.f;
        top_hold += slot0_own.holdings[j] * price;
    }

    // Save updated elites back to disk
    save_industry_elites(models_dir, ind_i, scratch.elite_buf);

    IndResult res;
    res.baseline   = baseline;
    res.slot0_score = slot_scores[0];
    res.best_delta = best_delta;
    res.top_hold   = top_hold;
    res.top_cash   = slot0_own.cash;
    res.new_streak = new_streak;
    state.streak   = new_streak;
    return res;
}

// ── step_master ─────────────────────────────────────────────────────────────────

static float step_master(MasterState& state, MasterScratch& scratch,
                         const DayData& day, const DayData* fill,
                         const IndResult* ind_results,
                         int actual_day, int total_avail, int day_num, int num_days,
                         float sigma) {
    const DayData& fd = fill ? *fill : day;

    // Resolve actual_perf[N_IND]
    float actual_perf[N_IND] = {};
    for (int i = 0; i < N_IND; i++) {
        if (ind_results[i].baseline > 0.f)
            actual_perf[i] = ind_results[i].slot0_score / ind_results[i].baseline - 1.f;
    }

    // Pre-compute ind_stats: mean_deltas[15][5] per industry
    // mean_deltas[t][k]: t=0 most recent, t=14 oldest; k=0..4 delta channel
    float ind_mean_deltas[N_IND][15][5] = {};
    float ind_volatility[N_IND] = {};
    float ind_momentum[N_IND]   = {};

    for (int i = 0; i < N_IND; i++) {
        for (int t = 0; t < 15; t++) {
            float sum[5] = {}; int cnt = 0;
            for (int j = 0; j < IND_SYMS; j++) {
                const SymHist& h = state.sym_hist[i * IND_SYMS + j];
                if (h.len > t) {
                    const float* d = h.get(h.len - 1 - t).data;  // t steps back from newest
                    for (int k = 0; k < 5; k++) sum[k] += d[5 + k];
                    cnt++;
                }
            }
            if (cnt > 0) for (int k = 0; k < 5; k++) ind_mean_deltas[i][t][k] = sum[k] / cnt;
        }
        // volatility and momentum from mean_deltas[*][0] (delta close)
        float vals[15]; int vlen = 0;
        for (int t = 0; t < 15; t++) vals[vlen++] = ind_mean_deltas[i][t][0];
        float mean_v = 0.f;
        for (int t = 0; t < vlen; t++) mean_v += vals[t];
        mean_v /= vlen > 0 ? vlen : 1;
        float var_v = 0.f;
        for (int t = 0; t < vlen; t++) { float dv = vals[t] - mean_v; var_v += dv*dv; }
        float std_v = vlen > 0 ? sqrtf(var_v / vlen) : 0.f;
        float abs_mean = fabsf(mean_v);
        ind_volatility[i] = abs_mean > 1e-9f ? std_v / abs_mean : 0.f;
        float mean5 = 0.f;
        for (int t = 0; t < std::min(vlen, 5); t++) mean5 += vals[t];
        mean5 /= std::min(vlen, 5) > 0 ? std::min(vlen, 5) : 1;
        ind_momentum[i] = abs_mean > 1e-9f ? mean5 / mean_v : 1.f;
    }

    // flat_cos_hist padded to 15
    float fc_padded[15] = {};
    int fc_start = 15 - state.flat_cos_len;
    memcpy(fc_padded + fc_start, state.flat_cos_hist, state.flat_cos_len * sizeof(float));

    // Build master history: hist15x61, row 0 = oldest
    float hist_arr[15 * 61] = {};
    for (int row = 0; row < 15; row++) {
        // row 0 = oldest (t=14 in mean_deltas), row 14 = newest (t=0)
        int t = 14 - row;
        for (int i = 0; i < N_IND; i++)
            for (int k = 0; k < 5; k++)
                hist_arr[row * 61 + i * 5 + k] = ind_mean_deltas[i][t][k];
        hist_arr[row * 61 + 60] = fc_padded[row];
    }

    // Compute today_ind_data: delta aggs and ind_mean_today per industry
    float today_ind_aggs[N_IND][15] = {};
    float today_ind_mean[N_IND]     = {};

    for (int i = 0; i < N_IND; i++) {
        float dl[IND_SYMS][5] = {};
        for (int j = 0; j < IND_SYMS; j++) {
            int si = i * IND_SYMS + j;
            const OHLCV& d = day.sym[i][j];
            float raw_t[5] = {d.valid?d.open:0, d.valid?d.close:0,
                              d.valid?d.high:0, d.valid?d.low:0, d.valid?d.volume:0};
            const HistEntry* prev = state.sym_hist[si].newest();
            for (int k = 0; k < 5; k++) dl[j][k] = prev ? raw_t[k] - prev->data[k] : 0.f;
        }
        float sum_delta_close = 0.f;
        for (int k = 0; k < 5; k++) {
            float mx = dl[0][k], mn = dl[0][k], sm = 0.f;
            for (int j = 0; j < IND_SYMS; j++) {
                if (dl[j][k] > mx) mx = dl[j][k];
                if (dl[j][k] < mn) mn = dl[j][k];
                sm += dl[j][k];
            }
            today_ind_aggs[i][k * 3 + 0] = mx;
            today_ind_aggs[i][k * 3 + 1] = mn;
            today_ind_aggs[i][k * 3 + 2] = sm / IND_SYMS;
        }
        // ind_mean_today = mean delta_close across symbols
        for (int j = 0; j < IND_SYMS; j++) sum_delta_close += dl[j][1];
        today_ind_mean[i] = sum_delta_close / IND_SYMS;
    }
    float all_mean_today = 0.f;
    for (int i = 0; i < N_IND; i++) all_mean_today += today_ind_mean[i];
    all_mean_today /= N_IND;

    // Build today_arr [229]
    float today_arr[229] = {};
    for (int i = 0; i < N_IND; i++) {
        int base = i * 18;
        for (int k = 0; k < 15; k++) today_arr[base + k] = today_ind_aggs[i][k];
        today_arr[base + 15] = ind_volatility[i];
        today_arr[base + 16] = ind_momentum[i];
        float corr = fabsf(all_mean_today) > 1e-9f ? today_ind_mean[i] / all_mean_today : 1.f;
        today_arr[base + 17] = corr;
    }
    // State: [cash_norm, holdings[12]]
    float ref_cash = state.portfolios[0].cash;
    float ref_hold[N_IND];
    for (int i = 0; i < N_IND; i++) ref_hold[i] = state.portfolios[0].holdings[i];
    today_arr[216] = ref_cash / std::max(MST_STARTING_CASH, 1.f);
    for (int i = 0; i < N_IND; i++) today_arr[217 + i] = ref_hold[i] * IND_UNIT_PRICE;

    // Baseline
    float baseline = ref_cash;
    for (int i = 0; i < N_IND; i++) baseline += ref_hold[i] * IND_UNIT_PRICE;

    // Reset all portfolios
    for (int s = 0; s < N_SLOTS; s++) {
        state.portfolios[s].cash = ref_cash;
        for (int i = 0; i < N_IND; i++) state.portfolios[s].holdings[i] = ref_hold[i];
    }

    // Inference + trade loop — use per-master scratch (no thread-local statics)
    float pred_scores[N_SLOTS] = {};
    float slot_preds[N_SLOTS][N_IND] = {};
    float out36[36];
    float*    mast_mut_buf  = scratch.mut_buf;
    uint64_t* mast_mut_seeds = scratch.mut_seeds;
    {
        PCG32 seed_rng; seed_rng.seed((uint64_t)actual_day * 777017ULL + 99999ULL);
        for (int i = 0; i < N_SLOTS - ELITE_POOL; i++)
            mast_mut_seeds[i] = ((uint64_t)seed_rng.next() << 32) | seed_rng.next();
    }

    float floor_value = MST_STARTING_CASH * 0.02f;

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

        master_forward(W, hist_arr, today_arr, out36);
        for (int i = 0; i < N_IND; i++) slot_preds[slot][i] = out36[i];

        MasterPortfolio& port = state.portfolios[slot];

        // Liquidation: sorted ascending by alloc weight
        int sorted_inds[N_IND];
        for (int i = 0; i < N_IND; i++) sorted_inds[i] = i;
        std::sort(sorted_inds, sorted_inds + N_IND,
                  [&out36](int a, int b){ return out36[a] < out36[b]; });

        for (int si = 0; si < N_IND; si++) {
            int i = sorted_inds[si];
            float price = IND_UNIT_PRICE;
            if (out36[12 + i] <= 0.5f) continue;  // liq_trigger
            float cur_hold_v = port.holdings[i] * price;
            if (cur_hold_v <= floor_value) continue;
            float target_hold_v = out36[i] * MST_STARTING_CASH;
            float depth = out36[12 + i];
            float eff_tgt = target_hold_v + (1.f - depth) * std::max(0.f, cur_hold_v - target_hold_v);
            eff_tgt = std::max(eff_tgt, floor_value);
            float liq_v = std::max(0.f, std::min(cur_hold_v - eff_tgt, cur_hold_v - floor_value));
            float liq_units = liq_v / price;
            if (liq_units > 1e-9f) {
                port.holdings[i] -= liq_units;
                port.cash        += sell_net(liq_units, price);
            }
        }

        // Deployment
        for (int i = 0; i < N_IND; i++) {
            float price = IND_UNIT_PRICE;
            float cur_hold_v = port.holdings[i] * price;
            float target_hold_v = out36[i] * MST_STARTING_CASH;
            float diff = target_hold_v - cur_hold_v;
            if (diff <= 1e-6f) continue;
            float affordable = std::min(diff, port.cash);
            float units = affordable / price;
            if (units > 1e-9f) {
                port.holdings[i] += units;
                port.cash        -= units * price;
            }
        }

        // Apply daily returns
        for (int i = 0; i < N_IND; i++)
            port.holdings[i] *= (1.f + actual_perf[i]);
    }

    // Compute target_pct from ind_results
    float raw_deltas[N_IND] = {};
    bool have_deltas = false;
    for (int i = 0; i < N_IND; i++) {
        if (ind_results[i].baseline > 0.f) {
            raw_deltas[i] = ind_results[i].slot0_score - ind_results[i].baseline;
            have_deltas = true;
        }
    }

    float target_pct[N_IND] = {};
    float flat_cos = 0.f;
    float best_pred = 0.f;

    if (have_deltas) {
        float min_d = *std::min_element(raw_deltas, raw_deltas + N_IND);
        float total = 0.f;
        float shifted[N_IND];
        for (int i = 0; i < N_IND; i++) { shifted[i] = raw_deltas[i] - min_d + 1e-9f; total += shifted[i]; }
        for (int i = 0; i < N_IND; i++) target_pct[i] = shifted[i] / total;

        // Compute pred scores
        for (int s = 0; s < N_SLOTS; s++) {
            float dot=0, na=0, nb=0;
            for (int i = 0; i < N_IND; i++) {
                dot += slot_preds[s][i] * target_pct[i];
                na  += slot_preds[s][i] * slot_preds[s][i];
                nb  += target_pct[i]    * target_pct[i];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            pred_scores[s] = denom > 1e-9f ? dot / denom : 0.f;
            if (pred_scores[s] > best_pred) best_pred = pred_scores[s];
        }

        // flat_cos: uniform alloc vs target_pct
        float flat_alloc = 1.f / N_IND;
        float dot=0, na=0, nb=0;
        for (int i = 0; i < N_IND; i++) {
            dot += flat_alloc * target_pct[i];
            na  += flat_alloc * flat_alloc;
            nb  += target_pct[i] * target_pct[i];
        }
        float denom = sqrtf(na) * sqrtf(nb);
        flat_cos = denom > 1e-9f ? dot / denom : 0.f;
    }

    log_msg(std::string("[master  ] Day ") + std::to_string(actual_day + 1) +
            "/" + std::to_string(total_avail) +
            " | pred=" + std::to_string(best_pred).substr(0,6) +
            " flat=" + std::to_string(flat_cos).substr(0,6) +
            " | prod=$" + std::to_string((int)baseline));

    // Floor reset
    if (baseline < MST_STARTING_CASH * 0.9f) {
        for (int s = 0; s < N_SLOTS; s++) {
            state.portfolios[s].cash = MST_STARTING_CASH;
            for (int i = 0; i < N_IND; i++) state.portfolios[s].holdings[i] = 0.f;
        }
        return flat_cos;
    }

    MasterPortfolio slot0_own = state.portfolios[0];

    // Selection
    if (best_pred >= 0.50f) {
        float mean_ps = 0.f;
        for (int s = 0; s < N_SLOTS; s++) mean_ps += pred_scores[s];
        mean_ps /= N_SLOTS;
        float var_ps = 0.f;
        for (int s = 0; s < N_SLOTS; s++) {
            float d = pred_scores[s] - mean_ps; var_ps += d * d;
        }
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
        for (int k = 0; k < n_top; k++) {
            src_rank[k] = surviving[k].second;
            src_val[k]  = surviving[k].first;
        }
        for (int k = n_top; k < ELITE_COUNT; k++) {
            src_rank[k] = src_rank[0];
            src_val[k]  = src_val[0];
        }

        float w5_w[5], w10_w[10], w15_w[15];
        int n5 = std::min(n_top,5), n10 = std::min(n_top,10), n15 = std::min(n_top,15);
        normalize_weights(src_val, w5_w, n5);
        normalize_weights(src_val, w10_w, n10);
        normalize_weights(src_val, w15_w, n15);

        MasterPortfolio new_mports[ELITE_POOL];

        // Portfolio wavg from original slot positions (state.portfolios has 200 entries — valid)
        const MasterPortfolio* mp5[5], *mp10[10], *mp15[15];
        for (int k=0;k<n5;k++)  mp5[k]  = &state.portfolios[src_rank[k]];
        for (int k=0;k<n10;k++) mp10[k] = &state.portfolios[src_rank[k]];
        for (int k=0;k<n15;k++) mp15[k] = &state.portfolios[src_rank[k]];
        MasterPortfolio wp5={}, wp10={}, wp15={};
        wavg_mst_portfolio(mp5,  w5_w,  n5,  wp5);
        wavg_mst_portfolio(mp10, w10_w, n10, wp10);
        wavg_mst_portfolio(mp15, w15_w, n15, wp15);

        // Copy/regenerate into new_elites (mutations regenerated — elite_buf has only ELITE_POOL slots)
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

        // Weight wavg from new_elites[0..n-1] (consecutive indices, no OOB risk)
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
        // Diversity injection
        int half = ELITE_COUNT / 2;
        log_msg(std::string("[master  ] best_pred=") + std::to_string(best_pred).substr(0,6) +
                " below floor — injecting diversity");
        PCG32 div_rng; div_rng.seed((uint64_t)actual_day * 55555ULL + 77777ULL);
        for (int k = half; k < ELITE_COUNT; k++) {
            init_master_weights(scratch.mut_buf, div_rng);
            for (int p = 0; p < MASTERNN_PARAMS; p++)
                scratch.elite(k)[p] = 0.5f * scratch.elite(k - half)[p] + 0.5f * scratch.mut_buf[p];
            state.portfolios[k] = state.portfolios[k - half];
        }
    }

    state.portfolios[0] = slot0_own;

    // Update flat_cos history
    if (state.flat_cos_len < 15) {
        state.flat_cos_hist[state.flat_cos_len++] = flat_cos;
    } else {
        memmove(state.flat_cos_hist, state.flat_cos_hist + 1, 14 * sizeof(float));
        state.flat_cos_hist[14] = flat_cos;
    }

    return flat_cos;
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

static void save_industry_elites(const std::string& dir, int ind_i,
                                  const float* elite_buf) {
    for (int slot = 0; slot < ELITE_POOL; slot++) {
        std::string path = elite_path(dir, IND_NAMES[ind_i], slot);
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
            std::string p = elite_path(load_dir, IND_NAMES[ind_i], slot);
            loaded = load_bin(p, e, STOCKNN_PARAMS);
        }
        if (!loaded) {
            std::string p = elite_path(dir, IND_NAMES[ind_i], slot);
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

// ── Stock data loading ──────────────────────────────────────────────────────────

static std::vector<DayData> load_all_stock_data(const std::string& data_dir,
                                                  std::vector<std::string>& out_dates) {
    // Collect per-symbol maps
    std::map<std::string, OHLCV> sym_maps[N_IND][IND_SYMS];
    int loaded = 0;
    for (int i = 0; i < N_IND; i++) {
        for (int j = 0; j < IND_SYMS; j++) {
            std::string path = data_dir + "/" + SYMS[i][j] + ".json";
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

static void warmup_history(IndustryState* ind_states, MasterState& mst,
                            const std::vector<DayData>& all_days, int day_start) {
    if (day_start <= 0) return;
    int start = std::max(0, day_start - HIST_WINDOW);
    for (int d = start; d < day_start; d++) {
        for (int i = 0; i < N_IND; i++) {
            for (int j = 0; j < IND_SYMS; j++) {
                update_hist_sym(ind_states[i].hist[j], all_days[d].sym[i][j]);
                update_hist_sym(mst.sym_hist[i * IND_SYMS + j], all_days[d].sym[i][j]);
            }
        }
    }
    log_msg("Warmup: pre-loaded " + std::to_string(day_start - start) + " days of history");
}

// ── CSV logging ────────────────────────────────────────────────────────────────

static void write_csv_row(FILE* csv, int pass_num, int actual_day,
                           const IndResult* res, float flat_cos) {
    fprintf(csv, "%d,%d", pass_num + 1, actual_day + 1);
    for (int i = 0; i < N_IND; i++)
        fprintf(csv, ",%+10.2f", res[i].best_delta);
    fprintf(csv, ",%+.4f\n", flat_cos);
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
        const std::string& use_load = (ctx->day_num == 0) ? ctx->load_dir : std::string();
        while (true) {
            int i = ctx->next_ind.fetch_add(1, std::memory_order_relaxed);
            if (i >= N_IND) break;
            if (!ctx->master_only)
                ctx->results[i] = step_industry(i, ctx->ind_states[i], scratch,
                                                 ctx->models_dir, use_load,
                                                 *ctx->day_ptr, ctx->fill_ptr,
                                                 ctx->actual_day, ctx->total_avail,
                                                 ctx->day_num, ctx->num_days,
                                                 ctx->sigma, ctx->freeze, ctx->seq_flags);
            else
                ctx->results[i] = {ctx->ind_states[i].portfolios[0].cash,
                                   ctx->ind_states[i].portfolios[0].cash,
                                   0.f, 0.f, 0.f, 0};
        }
        ctx->work_done.release();
    }
}

// ── main ───────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --output DIR [--load-dir DIR] [--start-day N] [--stop-day N]\n"
        "          [--passes N] [--sigma F] [--master-sigma F] [--sigma-decay F]\n"
        "          [--workers N] [--master-only] [--preserve-stock-data]\n", prog);
}

int main(int argc, char* argv[]) {
    // Parse args
    std::string output_dir, load_dir;
    int  start_day = -1, stop_day = -1, passes = 1, num_workers = 2;
    float sigma = 0.01f, master_sigma = -1.f, sigma_decay = 0.5f;
    bool master_only = false, preserve_stock = false;

    for (int a = 1; a < argc; a++) {
        std::string arg(argv[a]);
        if (arg == "--output"    && a+1<argc) { output_dir   = argv[++a]; }
        else if (arg == "--load-dir" && a+1<argc) { load_dir = argv[++a]; }
        else if (arg == "--start-day"&& a+1<argc) { start_day= atoi(argv[++a]); }
        else if (arg == "--stop-day" && a+1<argc) { stop_day = atoi(argv[++a]); }
        else if (arg == "--passes"   && a+1<argc) { passes   = atoi(argv[++a]); }
        else if (arg == "--sigma"    && a+1<argc) { sigma    = atof(argv[++a]); }
        else if (arg == "--master-sigma"&&a+1<argc){master_sigma=atof(argv[++a]);}
        else if (arg == "--sigma-decay"&&a+1<argc){sigma_decay=atof(argv[++a]);}
        else if (arg == "--workers"  && a+1<argc) { num_workers=atoi(argv[++a]);}
        else if (arg == "--master-only") master_only = true;
        else if (arg == "--preserve-stock-data") preserve_stock = true;
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }
    if (output_dir.empty()) { print_usage(argv[0]); return 1; }
    if (master_sigma < 0.f) master_sigma = sigma;

    fs::create_directories(output_dir);
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

    // Allocate state on heap (small now — elites removed)
    auto ind_states  = std::make_unique<IndustryState[]>(N_IND);
    auto mst         = std::make_unique<MasterState>();
    auto mst_scratch = std::make_unique<MasterScratch>();  // ~162 MB for master elites

    // Open CSV log
    std::string csv_path = output_dir + "/training_log.csv";
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (csv) {
        fprintf(csv, "pass,day");
        for (int i = 0; i < N_IND; i++) fprintf(csv, ",%s", IND_NAMES[i]);
        fprintf(csv, ",flat_cos\n");
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
        // Load master elites once at pass start (no per-day reload needed for master)
        load_or_init_master(output_dir, load_dir, mst_scratch->elite_buf);
        mst->portfolios[0].cash = MST_STARTING_CASH;
        for (int i = 0; i < N_IND; i++) mst->portfolios[0].holdings[i] = 0.f;
        for (int s = 1; s < N_SLOTS; s++) mst->portfolios[s] = mst->portfolios[0];
        mst->flat_cos_len = 0;

        // Clear histories
        for (int i = 0; i < N_IND; i++)
            for (int j = 0; j < IND_SYMS; j++)
                ind_states[i].hist[j] = SymHist{};
        for (int si = 0; si < N_SYMS; si++) mst->sym_hist[si] = SymHist{};

        // Warmup history
        warmup_history(ind_states.get(), *mst, all_days, day_start);

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

            // Update histories (main thread is canonical, matching Python design)
            for (int i = 0; i < N_IND; i++) {
                for (int j = 0; j < IND_SYMS; j++) {
                    update_hist_sym(ind_states[i].hist[j], day_ptr->sym[i][j]);
                    update_hist_sym(mst->sym_hist[i * IND_SYMS + j], day_ptr->sym[i][j]);
                }
            }

            // Master step
            float flat_cos = step_master(*mst, *mst_scratch, *day_ptr, fill_ptr, results,
                                         actual_day, total_days, day_num, num_days,
                                         cur_mst_sigma);

            // Log CSV row
            if (csv) write_csv_row(csv, pass, actual_day, results, flat_cos);

            // Periodic master save (industry elites already saved inside step_industry each day)
            if (day_num % 50 == 49 || day_num == num_days - 1) {
                log_msg("Saving master elites to " + output_dir + " ...");
                save_master_elites(output_dir, mst_scratch->elite_buf);
            }
        }

        // Save master after each pass (industry elites already saved)
        log_msg("Pass " + std::to_string(pass+1) + " complete — saving master elites");
        save_master_elites(output_dir, mst_scratch->elite_buf);
    }

    // Shutdown workers
    wctx.shutdown.store(true);
    wctx.work_ready.release(num_workers);
    for (int w = 0; w < num_workers; w++) wctx.work_done.acquire();
    for (auto& t : workers) t.join();

    if (csv) fclose(csv);
    log_msg("Training complete.");
    return 0;
}
