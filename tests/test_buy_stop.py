"""The stop-loss a buy carries: STOP_LOSS_FRAC below the lower of its limit and the decision close.

It used to be 90% of the limit. A limit above the close fills near the next open, not at the limit,
so that stop could sit at the market: on paper (2026-10-14) SSRM's $35.50 limit filled at $32.31,
its $31.95 stop was already crossed when it went live, and Alpaca cancelled it. The trainer, the
Python simulator and production must use the same rule, or the models learn on stops the broker
never holds.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from training_lib import STOP_LOSS_FRAC, buy_stop_price

pv = pytest.importorskip("production_v2")


def test_limit_above_close_uses_the_close():
    assert buy_stop_price(35.50, 32.30) == pytest.approx(32.30 * STOP_LOSS_FRAC)


def test_limit_below_close_uses_the_limit():
    assert buy_stop_price(30.00, 32.30) == pytest.approx(30.00 * STOP_LOSS_FRAC)


def test_production_buy_carries_the_shared_stop():
    day = {"AAA": {"open": 50.0, "high": 60.0, "low": 40.0, "close": 45.0}}
    # buy_price_frac 0.9 puts the limit at $58, well above the $45 close
    o = pv.plan_industry_orders(["AAA"], {"AAA": (5.0, 0.9, 2.0, 0.0)}, {}, day, 10_000.0, {})
    buy = next(x for x in o if x["action"] == "buy")
    assert buy["price"] == pytest.approx(58.0)
    assert buy["stop"] == pytest.approx(round(45.0 * STOP_LOSS_FRAC, 2))


def test_no_copy_of_the_old_rule_survives():
    for path in ("training_lib.py", "production_v2.py", "inspect_trades.py", "training_v4.cpp"):
        src = Path(path).read_text()
        assert not re.search(r"buy_price\s*\*\s*0\.9", src), path


def test_cpp_twin_matches(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("g++ not available")
    src = Path("training_v4.cpp").read_text()
    frac = re.search(r"static constexpr float STOP_LOSS_FRAC = [^;]+;", src).group(0)
    fn = re.search(r"static inline float buy_stop_price\(.*?\n}\n", src, re.S).group(0)
    cases = [(35.50, 32.30), (30.00, 32.30), (50.0, 50.0), (61.5, 61.31)]
    prog = tmp_path / "stop.cpp"
    prog.write_text("#include <algorithm>\n#include <cstdio>\n" + frac + "\n" + fn
                    + "int main(){\n"
                    + "".join(f'printf("%.6f\\n", buy_stop_price({b}f, {c}f));\n'
                              for b, c in cases)
                    + "}\n")
    exe = tmp_path / "stop"
    subprocess.run([cxx, "-O2", "-o", str(exe), str(prog)], check=True)
    out = [float(x) for x in subprocess.run([str(exe)], capture_output=True, text=True,
                                            check=True).stdout.split()]
    for (b, c), got in zip(cases, out, strict=True):
        assert got == pytest.approx(buy_stop_price(b, c), rel=1e-6)
