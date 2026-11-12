"""race_models.py: every race architecture's numpy forward must match the trainer's C++ forward.

A committee member in paper trading has to predict exactly as it did in the race, or the race
says nothing about it. The weight files are raw float arrays validated by element count only, so a
layout drift that keeps the count loads silently and votes with plausible wrong numbers. These
tests compile the C++ forwards from the trainer's own headers and compare.
"""
import shutil
import subprocess

import numpy as np
import pytest

import race_models as rm

CPP = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "mt1_grad.h"
#include "mt1_backprop.h"
static std::vector<float> rd(const char* p) {
    std::vector<float> v; FILE* f = fopen(p, "rb"); float x;
    while (f && fread(&x, sizeof(float), 1, f) == 1) v.push_back(x);
    if (f) fclose(f);
    return v;
}
int main(int argc, char** argv) {
    const std::string a = argv[1];
    std::vector<float> W = rd(argv[2]), in = rd(argv[3]);
    double out = 0;
    if (a == "MT1") out = mt1net_forward(W.data(), in.data(), 0.f);
    else if (a == "MT1C") out = mt1cnet_forward(W.data(), in.data(), 0.f);
    else if (a == "MT1S") out = mt1s_forward(W.data(), in.data());
    else if (a == "MT1P_evo" || a == "MT1P_grad") {
        float sum = 0.f;
        for (int j = 0; j < MT1C_SYMS; j++) {
            const float o = mt1p_forward(W.data() + (size_t)j * MT1P_PARAMS, in.data() + j * MT1P_IN);
            sum += (a == "MT1P_evo") ? mt1_pred(o) : o;
        }
        out = sum;
    } else if (a == "TANH32") {
        MT1Backprop n; n.init({MT1C_IN, 32, 8, 1}, 1);
        if (W.size() != n.W.size()) return 4;
        n.W = W; out = n.forward(in.data());
    } else return 3;
    printf("%.9e\n", out);
    return 0;
}
'''


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    d = tmp_path_factory.mktemp("race_models")
    (d / "fwd.cpp").write_text(CPP)
    r = subprocess.run(["g++", "-std=c++20", "-O0", "-I", ".", "-o", str(d / "fwd"),
                        str(d / "fwd.cpp"), "-lm"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:2000]          # a header that no longer builds is a failure
    return d / "fwd"


def cpp(exe, tmp_path, arch, W, x):
    wf, xf = tmp_path / "w.bin", tmp_path / "x.bin"
    np.asarray(W, np.float32).tofile(wf)
    np.asarray(x, np.float32).tofile(xf)
    r = subprocess.run([str(exe), arch, str(wf), str(xf)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return float(r.stdout)


def close(a, b):
    return abs(a - b) <= 1e-4 * max(1.0, abs(b))


IN_WIDTH = {"MT1": 74, "MT1C": rm.C_IN, "MT1S": rm.C_IN, "TANH32": rm.C_IN}


@pytest.mark.parametrize("arch", ["MT1", "MT1C", "MT1S", "TANH32"])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_forward_matches_cpp(exe, tmp_path, arch, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.3, rm.PARAMS[arch]).astype(np.float32)
    x = rng.normal(0, 1, IN_WIDTH[arch]).astype(np.float32)
    m = rm.RaceModel({"MT1": "MT1 evo", "MT1C": "MT1C evo", "MT1S": "MT1S-sum evo",
                      "TANH32": "TANH32 grad"}[arch], W)
    got = m.predict(in74=x) if arch == "MT1" else m.predict(in146=x)
    want = cpp(exe, tmp_path, arch, W, x)
    assert close(got, want), f"{arch}: numpy {got} vs C++ {want}"


@pytest.mark.parametrize("entry,tag", [("MT1S-stk evo", "MT1P_evo"), ("MT1S-stk grad", "MT1P_grad")])
@pytest.mark.parametrize("seed", [0, 3])
def test_per_stock_forward_matches_cpp_in_its_own_units(exe, tmp_path, entry, tag, seed):
    """evo sums tanh(out) x $10,000 per stock, grad sums raw outputs -- each as it was trained."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.3, rm.PARAMS["MT1P"]).astype(np.float32)
    s = rng.normal(0, 1, (12, rm.P_IN)).astype(np.float32)
    got = rm.RaceModel(entry, W).predict(slices18=s)
    assert close(got, cpp(exe, tmp_path, tag, W, s.ravel()))


def test_every_input_reaches_the_mt1s_output(exe, tmp_path):
    """MT1S reads 12 symbol blocks plus cash and book; a slicing slip would drop a block."""
    rng = np.random.default_rng(5)
    W = rng.normal(0, 0.5, rm.MT1S_PARAMS).astype(np.float32)
    base = np.zeros(rm.C_IN, np.float32)
    b0 = rm.mt1s_forward(W, base)
    moved = sum(abs(rm.mt1s_forward(W, np.where(np.arange(rm.C_IN) == k, 5.0, 0.0).astype(np.float32))
                    - b0) > 1e-7 for k in range(rm.C_IN))
    assert moved > rm.C_IN * 0.5


def test_wrong_size_is_refused():
    with pytest.raises(ValueError):
        rm.RaceModel("MT1C evo", np.zeros(rm.MT1C_PARAMS - 1))


def test_slug_matches_the_trainers_file_names():
    # names as they appear in champion/ on a v5 run
    assert rm.slug("MT1 evo") == "MT1______evo"
    assert rm.slug("MT1 grdC") == "MT1______grdC"
    assert rm.slug("MT1C evo") == "MT1C_____evo"
    assert rm.slug("MT1S-stk evo") == "MT1S-stk_evo"
    assert rm.slug("TANH32 grad") == "TANH32___grad"


def test_mt1p_slices_ratio_and_raw_forms():
    in146 = np.arange(rm.C_IN, dtype=np.float32) / rm.C_IN
    close_ = np.full(12, 50.0)
    valid = np.ones(12, bool)
    valid[3] = False
    hold = np.full(12, 10.0)
    bq, sq = np.full(12, 20.0), np.full(12, 5.0)
    bf, sf, fl = np.full(12, 0.25), np.full(12, 0.75), np.full(12, 0.2)
    raw = rm.mt1p_slices(in146, close_, valid, 1000.0, hold, bq, bf, sf, sq, fl, raw=True)
    rat = rm.mt1p_slices(in146, close_, valid, 1000.0, hold, bq, bf, sf, sq, fl, raw=False)
    assert np.allclose(raw[0, :13], in146[:13]) and np.allclose(raw[0, 13:15], in146[156:158])
    assert np.allclose(raw[0, 15:], [20, 0.25, 0.75, 5, 0.2])
    assert np.allclose(rat[0, 15:], [20 / (1000 / 50), 0.25, 0.75, 5 / 10, 0.2])
    assert np.allclose(raw[3, 15:], [0, 0.5, 0.5, 0, 0.5])     # no valid bar -> no order


def test_champion_files_load_when_present():
    """On the droplet, every v5 champion file must load at its expected size."""
    import os
    d = "/root/ht_race_v5/champion"
    if not os.path.isdir(d):
        pytest.skip("no v5 champion store here")
    # The v5 store predates StockNN's sell_flag (MT1C 11177 weights, now 11945): it can never load
    # again, and the next full training replaces it. Skip it rather than read it.
    mt1c = os.path.join(d, f"race_{rm.slug('MT1C evo')}_energy.bin")
    if os.path.exists(mt1c) and os.path.getsize(mt1c) != 4 * rm.MT1C_PARAMS:
        pytest.skip("champion store predates sell_flag (race inputs 146 wide)")
    n = 0
    for f in sorted(os.listdir(d)):
        if not f.startswith("race_"):
            continue
        for e in rm.ARCH:
            pre = f"race_{rm.slug(e)}_"
            if f.startswith(pre):
                rm.RaceModel.load(d, e, f[len(pre):-4])
                n += 1
                break
    assert n == 120


def _bars():
    o = np.full(12, 50.0)
    h = np.full(12, 52.0)
    l_ = np.full(12, 48.0)
    c = np.full(12, 51.0)
    c[1] = 102.0
    h[1], l_[1], o[1] = 104.0, 96.0, 100.0
    valid = np.ones(12, bool)
    valid[2] = False
    return o, h, l_, c, valid


def test_cfeat146_hand_worked_symbol_and_industry_fields():
    o, h, l_, c, valid = _bars()
    hold = np.zeros(12)
    hold[0] = 100.0                                  # $5,100 at 51
    intent = np.zeros((12, 5))
    intent[0] = (30.7, 0.25, 0.9, 40.6, 0.8)         # 30 buyable of 30.7; sells 40 of 100
    intent[1] = (5.0, 1.7, -0.2, 0.0, 1.3)           # fractions clamp into [0, 1]
    x = rm.build_cfeat146(o, h, l_, c, valid, hold, 4900.0, intent)
    book = 4900.0 + 5100.0
    mean_close = (51.0 * 10 + 102.0) / 11            # valid closes only
    s0 = x[:13]
    assert np.allclose(s0[:4], [50 / 51, 52 / 51, 48 / 51, 51 / mean_close], rtol=1e-6)
    assert np.isclose(s0[6], 4.0 / ((100 + 153 + 52 + 48) / 7), rtol=1e-6)     # (H-L)/A
    assert np.isclose(s0[7], 5100.0 / book)                                   # position weight
    assert np.isclose(s0[8], 30.0 / 96.0)            # floor(30.7) of floor(4900/51)=96 affordable
    assert np.allclose(s0[9:13], [0.25, 0.9, 0.40, 0.8])                      # ... sell_flag
    assert np.allclose(x[13 + 9:13 + 11], [1.0, 0.0]) and x[13 + 12] == 1.0   # clamped
    assert np.allclose(x[26:30], [1, 1, 1, 1]) and not x[30:39].any()          # no valid bar
    assert np.isclose(x[156], 4900.0 / book)
    assert np.isclose(x[157], book / 25000.0)
    assert np.isclose(rm.build_cfeat146(o, h, l_, c, valid, hold, 4900.0, intent,
                                        book_base=10000.0)[157], 1.0)


def test_cfeat146_non_finite_intent_reads_as_no_opinion():
    o, h, l_, c, valid = _bars()
    intent = np.full((12, 5), np.nan)
    x = rm.build_cfeat146(o, h, l_, c, valid, np.zeros(12), 1000.0, intent)
    assert np.isfinite(x).all()
    assert np.allclose(x[9:11], 0.5) and x[8] == 0 and x[11] == 0 and x[12] == 0.5
