"""The trainer's fill prices, calibrated to paper (fees.buy_fill_price, training_v4.cpp's twin).

Paper's fill_check.csv (2026-10-13..11-10) showed the simulator pricing buys 0.43% cheaper than
the broker on balance: it filled a buy at or above the open at the open itself, where paper paid
+0.60% over it, and it charged 0.1% slippage beyond the limit on intraday limit fills, which never
filled worse than the limit (0 of 162). The rule now: a limit fills at the limit; a buy whose
limit is at or above the open fills at the open plus OPEN_BUY_PREMIUM, capped at the limit.
"""
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fees import OPEN_BUY_PREMIUM, buy_fill_price


def test_open_buy_pays_the_premium():
    assert buy_fill_price(50.0, 49.0, 52.0, 51.0) == pytest.approx(50.0 * (1 + OPEN_BUY_PREMIUM))


def test_open_buy_never_pays_more_than_its_limit():
    # The premium would put the price above the limit; the broker never fills above it.
    assert buy_fill_price(50.0, 49.0, 52.0, 50.1) == 50.1
    assert buy_fill_price(50.0, 49.0, 52.0, 50.0) == 50.0


def test_intraday_limit_fills_at_the_limit():
    assert buy_fill_price(50.0, 48.0, 52.0, 49.0) == 49.0


def test_no_fill_at_the_low_or_below_it():
    assert buy_fill_price(50.0, 49.0, 52.0, 49.0) == 0.0     # strict at the day's low
    assert buy_fill_price(50.0, 49.0, 52.0, 48.0) == 0.0


def test_premium_is_the_calibrated_value():
    # Changing it is a recalibration (fill_check.csv), not a tweak: say so in training_v4.cpp.
    assert OPEN_BUY_PREMIUM == pytest.approx(0.0070)


def test_cpp_twin_matches(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("g++ not available")
    src = Path("training_v4.cpp").read_text()
    block = (re.search(r"static constexpr float OPEN_BUY_PREMIUM\s*=[^;]+;", src).group(0) + "\n"
             + re.search(r"static inline float buy_fill_price\(.*?\n}\n", src, re.S).group(0))
    prog = tmp_path / "fill.cpp"
    prog.write_text("#include <algorithm>\n#include <cstdio>\n" + block + r"""
int main() {
    float o, l, h, lim;
    while (scanf("%f %f %f %f", &o, &l, &h, &lim) == 4)
        printf("%.6f\n", buy_fill_price(o, l, h, lim));
}
""")
    exe = tmp_path / "fill"
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe), str(prog)], check=True)
    rng = random.Random(20261112)
    cases = []
    for _ in range(2000):
        low = float(np.float32(rng.uniform(20, 90)))
        high = float(np.float32(low * rng.uniform(1.0, 1.06)))
        o = float(np.float32(rng.uniform(low, high)))
        lim = float(np.float32(rng.choice([low, high, o, o * 1.003, rng.uniform(low * 0.98, high * 1.02)])))
        cases.append((o, low, high, lim))
    out = subprocess.run([str(exe)], input="\n".join(" ".join(repr(x) for x in c) for c in cases) + "\n",
                         capture_output=True, text=True, check=True).stdout.split()
    for c, got in zip(cases, out, strict=True):
        assert float(got) == pytest.approx(buy_fill_price(*c), rel=1e-6, abs=1e-6), c
